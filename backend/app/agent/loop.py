"""Agent 主迴圈：ReAct，但有次數上限。

模型每一輪只決定「下一步」，看到工具結果才決定再下一步——這就是 ReAct
（Reason + Act）。加上 MAX_ITERATIONS 的上限，是為了讓成本、延遲、
可評估性都還在掌握中（見 tools.py 的架構決策）。

## 模型不會執行你的函式

它只回傳一個結構化的「我想呼叫 X，參數是 Y」。執行永遠是這裡的責任，
所以參數驗證、權限邊界、錯誤處理全都要在這一層做完。

## 三個必須做對的細節

1. 模型那則 tool_calls 訊息要先 append 回歷史，才能接 role="tool" 的結果。
   順序錯了 API 會直接拒絕。
2. 模型一次可能要求多個工具，每一個都要回一則結果（靠 tool_call_id 配對），
   少回一個就報錯。
3. 最後一輪要傳 tool_choice="none"，強制模型產出文字。
   沒有這層保險，模型可能用完次數還在要求呼叫工具，你手上什麼都沒有。
"""

import inspect
import json
from dataclasses import dataclass, field
from typing import Any

from langfuse.decorators import observe

from app.agent.schemas import SERVER_PROVIDED_ARGS, TOOL_REGISTRY, TOOL_SCHEMAS
from app.config import settings
from app.rag.embedding import client as openai_client

MAX_ITERATIONS = 3

# 工具結果寫進紀錄時截短。完整結果仍然會送給模型，這裡只影響除錯用的紀錄——
# summarize_food_logs 的輸出可能上千字，不截短會讓 trace 難以閱讀。
RESULT_PREVIEW_CHARS = 160

SYSTEM_PROMPT = """你是一位專業的營養教練，正在跟使用者對話。

你有幾個工具可以查詢使用者的飲食紀錄與衛福部食藥署的營養成分資料庫。
先判斷需要哪些資訊，呼叫工具取得，再根據結果回答。

回答規則：
- 一律用「你」稱呼使用者，不要說「使用者」。
- 問題只要提到「我」「今天」「這週」等涉及使用者本人狀況的字眼，
  **一律先呼叫工具查紀錄再回答**。即使後半句聽起來像一般知識問題
  （例如「我今天蛋白質不夠的話，推薦什麼補？」），前半句的「我今天」
  就表示你必須先查當天紀錄，確認實際缺口之後才給建議。
- 所有具體的營養數值都必須來自工具回傳的結果。
  工具查不到就如實說明資料不足，絕對不要引用你自己記憶中的營養數據。
- **推薦具體食物時，食物名稱也必須來自工具結果**，不只是數值。
  沒有呼叫工具就不要列出食物清單——列一串沒查過的食物名稱，
  跟編造數值一樣是沒有根據的回答。
- 工具回傳的合計與平均已經算好，直接引用，不要自己重新加總。
- 營養目標標示為「推估」時，回答要說明那是依體重估算的參考值。
- 使用者沒說份量時，用常見單位估算並講明假設（例如一杯豆漿約 300ml）。
- 用繁體中文，語氣自然精簡，控制在 200 字內。

資料範圍：
- 營養資料庫只有衛福部食藥署的**一般食材**，沒有連鎖餐飲的品項。
  使用者問「星巴克拿鐵的熱量」這類問題時，說明資料庫沒有該品牌的資料，
  可以改提供類似的一般食材數值當參考，但要講清楚那是替代參考而非該品項。

安全邊界：
- 你不是醫師。使用者問到疾病診斷、用藥、保健食品該不該吃、
  極端節食等醫療範疇的問題時，明確說明這超出你能提供的範圍，
  建議諮詢醫師或營養師。不要給出模稜兩可的建議。
- 一般的飲食均衡建議不在此限，可以正常回答。

功能範圍：
- 使用者描述剛吃了什麼（例如「我剛吃了排骨便當」）時，
  說明你目前只能查詢與分析，無法代為記錄，請他們使用飲食記錄功能新增。
  不要假裝已經記錄，也不要追問細節。
"""


@dataclass
class ToolCall:
    """一次工具呼叫的紀錄。

    存下來是為了讓 eval 能評估「路徑」而不只是最終答案——
    選對工具答對、和選錯工具湊巧答對，是兩種不同的系統健康度，
    只看答案分不出來。
    """

    name: str
    arguments: dict[str, Any]
    result_preview: str
    error: str | None = None


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    hit_limit: bool = False


def _execute(name: str, arguments: dict[str, Any], user_id: int) -> tuple[str, str | None]:
    """執行一個工具呼叫，回傳 (給模型看的結果字串, 錯誤訊息或 None)。

    工具拋例外時不讓它往上炸——把錯誤訊息當成工具結果回給模型，
    讓它有機會用正確的參數重試。直接拋出去的話使用者只會看到 500，
    而模型其實有能力自己修正（例如 nutrient 打錯欄位名）。
    """
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return f"錯誤：沒有名為 {name!r} 的工具。", f"unknown tool: {name}"

    # user_id 由伺服器端注入，不接受模型指定——模型能指定的話，
    # 一句「查 user 2 的紀錄」就會洩漏別人的資料。
    safe_args = {k: v for k, v in arguments.items() if k not in SERVER_PROVIDED_ARGS}

    # 只有查使用者資料的工具需要 user_id；search_nutrition 和 rank_by_nutrient
    # 查的是公開的營養資料庫，硬塞會 TypeError。用簽章判斷而不是另外維護一份
    # 名單，加新工具時不會忘記同步。
    if "user_id" in inspect.signature(func).parameters:
        safe_args["user_id"] = user_id

    try:
        result = func(**safe_args)
    except TypeError as e:
        # 參數名稱或數量不對，通常是模型編了不存在的參數
        return f"錯誤：參數不正確（{e}）。請檢查工具的參數說明後重試。", str(e)
    except ValueError as e:
        # 工具自己的驗證失敗（例如 nutrient 不在白名單），訊息本身就是給模型的提示
        return f"錯誤：{e}", str(e)

    if isinstance(result, str):
        return result, None
    # FoodItem 的清單轉成 JSON 給模型讀。空清單也要明確講，
    # 否則模型容易當成「沒查到就自己補」。
    if not result:
        return "查無符合條件的資料。", None
    return json.dumps([item.__dict__ for item in result], ensure_ascii=False), None


@observe()
def run_agent(question: str, user_id: int) -> AgentResult:
    """跑一輪 ReAct 迴圈回答問題。

    Args:
        question: 使用者的自然語言問題。
        user_id: 要查誰的資料。由呼叫端提供，不經過模型。

    Returns:
        AgentResult，含答案與完整的工具呼叫紀錄。
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    calls: list[ToolCall] = []
    # (工具, 參數) → 結果。模型拿到不滿意的結果時有時會原封不動再問一次，
    # 那只會白花錢又把一輪次數用掉。
    #
    # 重複時回傳**快取的原結果**而不是「剛才查過了」之類的提示：
    # 提示訊息會讓模型以為查詢失敗或無資料（實測過，它接著就改用自己的記憶回答）。
    # 把同樣的結果再給它一次，語意上也更誠實——這個查詢的答案就是這個。
    cache: dict[str, str] = {}

    for iteration in range(1, MAX_ITERATIONS + 1):
        is_last = iteration == MAX_ITERATIONS
        response = openai_client.chat.completions.create(
            model=settings.openai_model_cheap,
            messages=messages,
            tools=TOOL_SCHEMAS,
            # 最後一輪禁止呼叫工具，強迫它給出文字答案
            tool_choice="none" if is_last else "auto",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return AgentResult(
                answer=message.content or "",
                tool_calls=calls,
                iterations=iteration,
                hit_limit=False,
            )

        # 先把模型的 tool_calls 訊息放回歷史，才能接 role="tool" 的結果
        messages.append(message)

        for tc in message.tool_calls:
            name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            signature = f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
            if signature in cache:
                content, error = cache[signature], "duplicate call"
            else:
                content, error = _execute(name, arguments, user_id)
                cache[signature] = content

            calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    result_preview=content[:RESULT_PREVIEW_CHARS],
                    error=error,
                )
            )
            # 每個 tool_call_id 都必須有對應的結果，少一個 API 會報錯
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    # 迴圈跑完仍沒拿到文字答案。理論上最後一輪的 tool_choice="none"
    # 已經擋住這條路，留著是防止 API 行為改變時無聲失敗。
    return AgentResult(
        answer="抱歉，我查了幾次仍無法整理出答案，可以換個方式問嗎？",
        tool_calls=calls,
        iterations=MAX_ITERATIONS,
        hit_limit=True,
    )


if __name__ == "__main__":
    # 跑法：
    #   poetry run python -m app.agent.loop              ← 跑預設的七題
    #   poetry run python -m app.agent.loop "我昨天吃了什麼"  ← 只跑指定的問題
    import sys

    from langfuse.decorators import langfuse_context

    DEFAULT_QUESTIONS = [
        "豆漿的熱量是多少？",
        "我今天吃了什麼？總共多少熱量？",
        "有什麼高蛋白低卡的食物推薦？",
        "我這週跟上週比，熱量有下降嗎？",
        "我今天蛋白質不夠的話，推薦什麼補？",
        "我該不該吃減肥藥？",
        "我剛吃了一個排骨便當",
    ]

    questions = sys.argv[1:] or DEFAULT_QUESTIONS
    verbose = len(sys.argv) > 1  # 指定單題時印出完整的工具結果，方便除錯

    for q in questions:
        result = run_agent(q, user_id=1)
        print("=" * 74)
        print(f"Q: {q}")
        path = " → ".join(f"{c.name}({c.arguments})" for c in result.tool_calls) or "（未呼叫工具）"
        print(f"路徑（{result.iterations} 輪）: {path}")
        if verbose:
            for i, call in enumerate(result.tool_calls, 1):
                print(f"  [{i}] error={call.error}")
                print(f"      → {call.result_preview}")
        print("-" * 74)
        print(result.answer)
        print()

    langfuse_context.flush()
