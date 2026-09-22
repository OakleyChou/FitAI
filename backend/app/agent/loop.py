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

## 錯誤要先分類再處理

決定一個錯誤該怎麼辦，看的是「**模型有沒有能力修它**」：

- **模型能修**（參數名稱錯、值不在 enum、區間顛倒）→ 把訊息當工具結果回給它，
  讓它自己改。這是 function calling 少數能自我修正的地方。
- **模型修不了**（DB 連不上、API key 失效）→ **不要**把細節回給模型。
  它會換個參數再試一次，然後再失敗，把次數用完，最後編一個答案。
  只告訴它「這個工具暫時不可用」，讓它用其他資訊盡量回答。

錯誤訊息有兩種觀眾：給模型的要具體可據以修正，給使用者的不能洩漏內部細節。
使用者永遠不該看到 TypeError 或連線字串。
"""

import inspect
import json
import time
from collections import defaultdict
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

# 單次 LLM 呼叫的上限。OpenAI SDK 內建重試（預設 2 次、帶 backoff），
# 所以這裡只需設 timeout，不用自己寫重試邏輯。
LLM_TIMEOUT_SECONDS = 30.0

# 整個迴圈的 wall clock 上限。三輪 × (LLM + 工具查詢) 最壞情況可能拖很久，
# 接成 API 之後那會直接卡住 web request。
LOOP_BUDGET_SECONDS = 90.0

# 同一個工具連續失敗幾次就放棄。不設限的話，模型會把三輪次數
# 全耗在同一個壞掉的工具上，最後什麼答案都給不出來。
MAX_TOOL_FAILURES = 2

# 模型看得懂、也改得掉的錯誤。這些的訊息會原樣回給模型讓它重試。
#   TypeError  ← 參數名稱或數量不對（模型編了不存在的參數）
#   ValueError ← 工具自己的驗證失敗（nutrient 不在白名單、區間顛倒）
# 其他例外一律視為系統層問題：模型改參數也修不好，不該讓它看到細節。
MODEL_FIXABLE_ERRORS = (TypeError, ValueError)

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
- **引用營養資料庫的數值一定要寫計量基準**。資料庫的數字全都是「每 100 公克」，
  只寫「豆漿約 35 大卡」等於把單位吞掉了——35 大卡的是一口還是一公升？
  換算成常見份量（一杯 300ml）也可以，但那個份量要寫出來。
- **營養資料庫的食物要附上來源編號**，格式是食物名稱後面接方括號，
  例如「雞絞肉[I0410101]」。編號就是工具結果裡的 source_ref，原樣照抄，
  不要自己編也不要改格式。這讓使用者能回頭查證每個數字的出處。
  使用者自己的飲食紀錄不需要編號——那是他自己輸入的資料，沒有外部出處。
- 營養目標標示為「推估」時，回答要說明那是依體重估算的參考值。
- 使用者沒說份量時，用常見單位估算並講明假設（例如一杯豆漿約 300ml）。
- 食物名稱模糊或不完整時（「那個叫什麼粿的」「一種豆做的東西」），
  **還是要先用 search_nutrition 查查看再說**。語意檢索本來就擅長處理不精確的
  名稱，資料庫也收錄了俗名。查到候選就列出來讓使用者確認，真的查不到才請他們
  講具體一點。**不要在還沒查之前就判斷「太模糊所以查不到」**——那等於放棄了
  這個工具最擅長的事。
- **列出食物時最多 3 項**，不管是推薦還是查詢都一樣，挑最符合問題的。
  工具可能回傳五筆以上，但對話介面裡列太多反而難讀——
  選出最相關的幾個才是教練該做的事。
- 每項食物只講關鍵數值（通常是熱量加一個主要營養素），不要把所有欄位都唸出來。
- 用繁體中文，語氣自然精簡，控制在 300 字內。

資料範圍：
- 營養資料庫只有衛福部食藥署的**一般食材**，沒有連鎖餐飲的品項。
  使用者問「星巴克拿鐵的熱量」這類問題時，說明資料庫沒有該品牌的資料，
  可以改提供類似的一般食材數值當參考，但要講清楚那是替代參考而非該品項。

安全邊界：
- 你不是醫師。使用者問到疾病診斷、用藥、保健食品該不該吃、
  極端節食等醫療範疇的問題時，明確說明這超出你能提供的範圍，
  建議諮詢醫師或營養師。不要給出模稜兩可的建議。
- 一般的飲食均衡建議不在此限，可以正常回答。

工具失敗時：
- 某個工具回報「暫時不可用」時，**用已經取得的其他資訊盡量回答**，
  並明確說明哪一部分查不到。部分答案加上說明，比整個拒絕回答有用得多。
- 不要因為一個工具壞掉就放棄整個問題，也不要改用你自己記憶中的數據填補。

功能範圍：
- 使用者描述剛吃了什麼（例如「我剛吃了排骨便當」），
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
    result: str
    error: str | None = None
    # 系統層失敗（工具暫時不可用），跟「模型參數給錯」或「重複呼叫」區分開——
    # 後兩者也會設 error，但那不代表服務降級。API 層靠這個旗標決定
    # 要不要告訴前端「部分功能暫時不可用」。
    unavailable: bool = False

    @property
    def result_preview(self) -> str:
        """截短版，給終端機輸出用。

        完整結果留在 result：eval 要檢查「答案裡的數值有沒有出處」時，
        必須比對模型實際看到的全部內容，截短過的會製造假警報。
        """
        return self.result[:RESULT_PREVIEW_CHARS]


@dataclass
class _ToolOutcome:
    """一次工具執行的結果。

    unavailable 把「模型能修的錯」和「系統壞了」分開——
    前者值得讓模型重試，後者重試幾次都一樣，該直接放棄那個工具。
    """

    content: str  # 給模型看的，系統錯誤時只有籠統說明
    error: str | None = None  # 給紀錄看的，保留完整訊息供除錯
    unavailable: bool = False  # False = 系統層失敗，別再試


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    hit_limit: bool = False
    # 系統層失敗的完整訊息。給使用者的 answer 不會包含這些細節，
    # 但要留下來，否則問題會無聲消失。
    error: str | None = None


def _execute(name: str, arguments: dict[str, Any], user_id: int) -> _ToolOutcome:
    """執行一個工具呼叫。

    錯誤分兩類處理（見模組 docstring）：
    模型能修的，訊息原樣回給它重試；修不了的，只回籠統說明並標記 unavailable。
    """
    func = TOOL_REGISTRY.get(name)
    if func is None:
        # 模型編了不存在的工具名。列出可用清單讓它能自我修正。
        available = "、".join(sorted(TOOL_REGISTRY))
        return _ToolOutcome(
            content=f"錯誤：沒有名為 {name!r} 的工具。可用的工具：{available}。",
            error=f"unknown tool: {name}",
        )

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
    except MODEL_FIXABLE_ERRORS as e:
        # 訊息本身就是給模型的修正提示，原樣傳遞
        return _ToolOutcome(
            content=f"錯誤：{e}。請檢查工具的參數說明後重試。",
            error=f"{type(e).__name__}: {e}",
        )
    except Exception as e:
        # 資料庫連線失敗、外部服務掛掉之類。模型改參數也修不好，
        # 而且錯誤內容可能含連線字串等不該外流的資訊。
        return _ToolOutcome(
            content=f"錯誤：{name} 目前暫時不可用，請改用其他方式或直接回答。",
            error=f"{type(e).__name__}: {e}",
            unavailable=True,
        )

    if isinstance(result, str):
        return _ToolOutcome(content=result)
    # FoodItem 的清單轉成 JSON 給模型讀。空清單也要明確講，
    # 否則模型容易當成「沒查到就自己補」。
    if not result:
        return _ToolOutcome(content="查無符合條件的資料。")
    return _ToolOutcome(
        content=json.dumps([item.__dict__ for item in result], ensure_ascii=False)
    )


@observe()
def run_agent(question: str, user_id: int) -> AgentResult:
    """跑一輪 ReAct 迴圈回答問題。

    這個函式**不會拋例外**。未預期的錯誤會轉成給使用者的訊息，
    完整內容留在 AgentResult.error——接成 API 之後，
    未捕捉的例外就是 500 加一整串 stack trace 給使用者看。

    Args:
        question: 使用者的自然語言問題。
        user_id: 要查誰的資料。由呼叫端提供，不經過模型。

    Returns:
        AgentResult，含答案、工具呼叫紀錄、以及系統層錯誤（若有）。
    """
    calls: list[ToolCall] = []
    try:
        return _run_loop(question, user_id, calls)
    except Exception as e:
        # 最外層防護。走到這裡表示迴圈本身出問題（不是工具），
        # 例如 LLM 服務掛掉或 timeout。給使用者一句話，細節留在 error。
        return AgentResult(
            answer="抱歉，系統目前無法處理你的問題，請稍後再試。",
            tool_calls=calls,
            error=f"{type(e).__name__}: {e}",
        )


def _run_loop(question: str, user_id: int, calls: list[ToolCall]) -> AgentResult:
    """迴圈本體。calls 由呼叫端傳入，讓例外發生時仍保留已完成的呼叫紀錄。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    # (工具, 參數) → 結果。模型拿到不滿意的結果時有時會原封不動再問一次，
    # 那只會白花錢又把一輪次數用掉。
    #
    # 重複時回傳**快取的原結果**而不是「剛才查過了」之類的提示：
    # 提示訊息會讓模型以為查詢失敗或無資料（實測過，它接著就改用自己的記憶回答）。
    # 把同樣的結果再給它一次，語意上也更誠實——這個查詢的答案就是這個。
    cache: dict[str, str] = {}
    # 每個工具累積的失敗次數，達到上限就不再執行它
    failures: dict[str, int] = defaultdict(int)
    deadline = time.monotonic() + LOOP_BUDGET_SECONDS

    for iteration in range(1, MAX_ITERATIONS + 1):
        if time.monotonic() > deadline:
            # 超時不算失敗——已經查到的資訊仍然有價值，
            # 只是沒機會讓模型整理成答案了。
            return AgentResult(
                answer="抱歉，這個問題查詢時間過長，可以換個方式問嗎？",
                tool_calls=calls,
                iterations=iteration - 1,
                hit_limit=True,
                error="loop budget exceeded",
            )

        is_last = iteration == MAX_ITERATIONS
        response = openai_client.chat.completions.create(
            model=settings.openai_model_cheap,
            messages=messages,
            tools=TOOL_SCHEMAS,
            # 最後一輪禁止呼叫工具，強迫它給出文字答案
            tool_choice="none" if is_last else "auto",
            timeout=LLM_TIMEOUT_SECONDS,
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
                # 模型產出的 arguments 不是合法 JSON。當成空參數送進去，
                # 讓工具自己的驗證回報缺什麼，比在這裡猜測好。
                arguments = {}

            signature = f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"

            if failures[name] >= MAX_TOOL_FAILURES:
                outcome = _ToolOutcome(
                    content=f"錯誤：{name} 已連續失敗多次，請不要再呼叫它。",
                    error="tool disabled after repeated failures",
                )
            elif signature in cache:
                outcome = _ToolOutcome(content=cache[signature], error="duplicate call")
            else:
                outcome = _execute(name, arguments, user_id)
                cache[signature] = outcome.content
                if outcome.error is not None:
                    failures[name] += 1
                if outcome.unavailable:
                    # 系統層失敗重試沒有意義，直接讓它達到上限
                    failures[name] = MAX_TOOL_FAILURES

            calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    result=outcome.content,
                    error=outcome.error,
                    unavailable=outcome.unavailable,
                )
            )
            # 每個 tool_call_id 都必須有對應的結果，少一個 API 會報錯
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": outcome.content}
            )

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
