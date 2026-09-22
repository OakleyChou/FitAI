"""LLM-as-Judge：用 LLM 代替人工判斷單一條 rubric 是否通過。

設計依據（Hamel Husain 的 LLM-as-Judge 方法論）：

**一個 judge 只問一個問題。** 不做「這個回答好不好」的萬能判官——
那種 judge 會被流暢度帶著走。每次呼叫只判定一條 rubric，準確率高得多，
失敗時也知道是哪一條沒過。

**先寫理由，再下判決。** JudgeVerdict 的欄位順序是 critique 在前、verdict 在後，
這不是排版問題：模型的推理發生在生成過程中，先吐 verdict 再解釋，
那個解釋只是事後合理化。附帶好處是 judge 判錯時你看得出它錯在哪。

**judge 用比受測模型更強的模型。** QA 用 gpt-4o-mini，judge 用 gpt-4o。

**judge 不做 L1 能做的事。** 簡體字、字數、引用標記格式用正則就好，
又快又準又免費。judge 只處理需要判斷力的部分。
"""

from typing import Literal

import instructor
from langfuse.openai import openai
from pydantic import BaseModel, Field

from app.config import settings
from evals.cache import JudgeCache, make_key

# 跟 app/llm.py 同樣的組合方式：langfuse 包過的 client 再交給 instructor，
# 這樣 token 用量會被記錄，同時拿到結構化輸出。
client = instructor.from_openai(openai.OpenAI(api_key=settings.openai_api_key))

JUDGE_MODEL = "gpt-4o"

JUDGE_SYSTEM_PROMPT = """你是一位嚴格的評審，負責判定一則 AI 營養教練的回答\
是否符合「單一一條」評分準則。

你會看到：
- <question>：使用者問的問題
- <evidence>：**當時 AI 實際看到的全部資料**。可能是以下兩種形態之一：
  - agent 架構：AI 自己選用的工具、呼叫參數、以及工具回傳的結果
  - 固定流程：事先檢索好的使用者飲食紀錄與營養資料庫結果（每筆有 [n] 編號）
- <answer>：AI 實際給出的回答
- <criterion>：這次要判定的唯一一條準則

判定規則：
- <criterion> 描述的是「通過時應有的狀態」。回答符合該描述就判 pass，
  不符合才判 fail。不要把「觀察到 criterion 描述的情況成立」誤當成違規。
- 只判定 <criterion> 這一條，不要評論回答的其他優缺點。
- 回答讀起來通順、有禮貌，跟這條準則通過與否無關。
- 判斷「數值是否正確」「有沒有編造」時，**一律以 <evidence> 為準**，
  不要用你自己記憶中的營養知識。
- <evidence> 就是 AI 當時能看到的全部。答案裡出現 <evidence> 沒有的食物或數字，
  就是編造；<evidence> 裡有而答案沒用到，那是另一回事，不要因此判 fail。
- 準則若涉及具體門檻（例如「每 100g 低於 20g 不算高蛋白」），嚴格照門檻判定。
- 先在 critique 寫下你的分析與依據，再給 verdict。
- 有疑慮而無法確定時判 fail——評審的作用是抓問題，不是給面子。
"""


class JudgeVerdict(BaseModel):
    """欄位順序即生成順序：先分析，後判決。"""

    critique: str = Field(description="判定的理由，引用 answer 或 context 中的具體內容")
    verdict: Literal["pass", "fail"] = Field(description="這條準則是否通過")


def judge_rubric(
    question: str,
    answer: str,
    criterion: str,
    evidence: str,
    cache: JudgeCache | None = None,
    model: str = JUDGE_MODEL,
) -> JudgeVerdict:
    """判定一則回答是否符合單一一條 rubric。

    ## evidence 必須是 AI 當時真正看到的東西

    這個參數原本是兩個（user_context / knowledge_context），由校準那端
    重新跑一次固定流程的檢索來產生。換成 agent 架構之後那個做法就錯了，
    而且錯得不會報錯：agent 自己決定呼叫哪個工具，judge 卻拿語意檢索的結果
    去對照。2026-09-22 的校準實際踩到——問「有什麼高蛋白低卡的食物」時，
    agent 用 rank_by_nutrient 拿到雞絞肉（27.4g），judge 拿到的卻是
    語意檢索撈回來的五筆蛋白（10.7g），於是把正確的答案判成「編造」。

    所以改成由呼叫端傳入**歸檔的**證據。跟 answer 同一個道理：
    重跑會得到不同的東西，judge 跟人工就不是在看同一份資料。

    Args:
        question: 原始問題。
        answer: AI 當初給出的回答（從歸檔的 CSV 讀，不要重跑 QA——
            同一題不同次執行的答案會變，重跑就無法跟人工標註對照）。
        criterion: 要判定的那一條 rubric。
        evidence: AI 當時看到的全部資料。agent 架構傳 CSV 的 tool_results 欄，
            固定流程傳當初歸檔的兩份 context。
        cache: 給了就會先查快取。judge 的輸出是所有輸入的函數，
            輸入沒變就沒有理由重算——校準會反覆執行，不快取會付好幾次錢。
        model: 判官模型。model 已納入快取 key，所以換模型會有自己的快取空間，
            不會跟別的模型的判定混在一起。

    Returns:
        JudgeVerdict，含理由與二元判定。
    """
    user_message = (
        f"<question>\n{question}\n</question>\n\n"
        f"<evidence>\n{evidence}\n</evidence>\n\n"
        f"<answer>\n{answer}\n</answer>\n\n"
        f"<criterion>\n{criterion}\n</criterion>"
    )

    key = None
    if cache is not None:
        # system prompt 也算進 key：改了 prompt 就該重判，
        # 靠 key 變動自動失效，比手動清快取可靠。
        key = make_key(
            model=model,
            system=JUDGE_SYSTEM_PROMPT,
            user=user_message,
        )
        cached = cache.get(key)
        if cached is not None:
            return JudgeVerdict(**cached)

    verdict = client.chat.completions.create(
        model=model,
        response_model=JudgeVerdict,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    if cache is not None and key is not None:
        cache.set(key, verdict.model_dump())

    return verdict
