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

# 跟 app/llm.py 同樣的組合方式：langfuse 包過的 client 再交給 instructor，
# 這樣 token 用量會被記錄，同時拿到結構化輸出。
client = instructor.from_openai(openai.OpenAI(api_key=settings.openai_api_key))

JUDGE_MODEL = "gpt-4o"

JUDGE_SYSTEM_PROMPT = """你是一位嚴格的評審，負責判定一則 AI 營養教練的回答\
是否符合「單一一條」評分準則。

你會看到：
- <question>：使用者問的問題
- <user_food_log>：當時提供給 AI 的使用者飲食紀錄
- <nutrition_reference>：當時提供給 AI 的營養資料庫檢索結果（每筆有 [n] 編號）
- <answer>：AI 實際給出的回答
- <criterion>：這次要判定的唯一一條準則

判定規則：
- <criterion> 描述的是「通過時應有的狀態」。回答符合該描述就判 pass，
  不符合才判 fail。不要把「觀察到 criterion 描述的情況成立」誤當成違規。
- 只判定 <criterion> 這一條，不要評論回答的其他優缺點。
- 回答讀起來通順、有禮貌，跟這條準則通過與否無關。
- 判斷「數值是否正確」時，以 <user_food_log> 和 <nutrition_reference> 為準，
  不要用你自己記憶中的營養知識。
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
    user_context: str,
    knowledge_context: str,
    criterion: str,
) -> JudgeVerdict:
    """判定一則回答是否符合單一一條 rubric。

    Args:
        question: 原始問題。
        answer: AI 當初給出的回答（從歸檔的 CSV 讀，不要重跑 QA——
            同一題不同次執行的答案會變，重跑就無法跟人工標註對照）。
        user_context: 當初的使用者飲食紀錄 context。
        knowledge_context: 當初的營養資料庫 context。
        criterion: 要判定的那一條 rubric。

    Returns:
        JudgeVerdict，含理由與二元判定。
    """
    user_message = (
        f"<question>\n{question}\n</question>\n\n"
        f"<user_food_log>\n{user_context}\n</user_food_log>\n\n"
        f"<nutrition_reference>\n{knowledge_context}\n</nutrition_reference>\n\n"
        f"<answer>\n{answer}\n</answer>\n\n"
        f"<criterion>\n{criterion}\n</criterion>"
    )

    return client.chat.completions.create(
        model=JUDGE_MODEL,
        response_model=JudgeVerdict,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
