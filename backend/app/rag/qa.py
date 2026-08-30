"""飲食問答（RAG QA 第一版）。

架構是**固定流程**：每個問題都同時取兩份 context，不做路由判斷。
    ① 使用者飲食紀錄  ← SQL 查 food_logs
    ② 營養知識         ← 向量檢索 knowledge_chunks
兩份都塞進 prompt，讓 LLM 自己判斷該用哪些。

為什麼不先分類問題再決定走哪條路？
第一版要的是「行為可預測」——固定流程沒有「分類錯誤」這種失敗模式，
明天做 eval 時變因單純。等有了 eval 基準，再談 router 或 tool use 才有依據
可以比較「換架構到底有沒有變好」。

代價是每個問題都付兩邊的成本、prompt 較長。以目前的量級可以接受。
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from langfuse.decorators import observe
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import FoodLog, User

# 這裡要的是純文字回答，不是結構化輸出，所以不能用 app.llm 那個
# instructor 包過的 client（它預期每次呼叫都帶 response_model）。
# 借用 embedding.py 裡那個「只有 langfuse 包過」的 client，一樣有 token 追蹤。
# TODO: client 建在 embedding.py 裡名不符實，之後可抽成 app/openai_client.py
from app.rag.embedding import client as openai_client
from app.rag.retrieval import RetrievedChunk, search

LOCAL_TZ = ZoneInfo("Asia/Taipei")

# 每公斤體重的每日蛋白質建議攝取量（克）。
# User 資料表目前只有 daily_calories_goal，沒有三大營養素的目標，
# 所以蛋白質目標用體重推估。1.6 g/kg 是常見的增肌區間下緣。
# 這是暫時的權宜作法，prompt 裡會註明是推估值，避免 LLM 講得像官方標準。
PROTEIN_G_PER_KG = 1.6

# 第一版先放模組常數，不進 PromptVersion 資料表。
# 理由：明天做 eval 時這段 prompt 會被大量修改，放在程式碼裡改得快；
# 等它穩定下來再搬進 PromptVersion（9/08 排程本來就有這項）。
SYSTEM_PROMPT = """你是一位專業的營養教練，正在跟使用者對話。

你會收到三個區塊：
- <user_food_log>：這位使用者今天實際吃的東西與營養目標，是事實。
- <nutrition_reference>：從衛福部食藥署營養成分資料庫檢索出來的一般知識，
  每筆前面有 [n] 編號。這只是參考資料，不代表使用者吃過這些食物。
- <question>：要回答的問題。

回答規則：
- 先判斷問題類型：
  (a) 一般營養知識（「X 的熱量是多少」）→ 只用 <nutrition_reference> 回答，
      不要提及使用者的飲食紀錄，也不要說明紀錄裡缺什麼。
  (b) 關於使用者自己的問題（「我」「今天」「還差多少」）→ 以 <user_food_log> 為主。
- 一律用「你」稱呼使用者，不要說「使用者」。
- 引用 <nutrition_reference> 的營養數值時，必須在句末標註來源編號，例如「約 35 大卡 [1]」。
- 無法標註來源的具體營養數值一律不要寫出來。若參考資料不足以回答，就說明資料不足，
  絕對不要引用你自己記憶中的營養數據。
- <user_food_log> 裡的合計數字已經算好，直接使用，不要自己重新加總。
- 營養目標標示為「推估」時，回答要說明那是依體重估算的參考值。
- 使用者沒說份量時，用常見單位估算並講明假設（例如一杯豆漿約 300ml）。
- 用繁體中文，語氣自然精簡，控制在 200 字內。

回答範例：

Q：豆漿的熱量是多少？
A：無糖豆漿每 100 公克約 35 大卡 [1]，一杯 300ml 大約 105 大卡。

Q：我今天蛋白質吃夠了嗎？
A：你今天吃了 64g，目標約 96g（依體重推估），還差 32g。可以再補一份雞胸肉或豆製品。

Q：有什麼高鈣的食物？
A：目前檢索到的資料裡沒有適合的高鈣食物，我無法給你有依據的建議。
"""


@dataclass
class QAResult:
    """一次問答的完整結果，包含中間產物，方便除錯與之後做 eval。"""

    question: str
    answer: str
    user_context: str
    knowledge_context: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)


def _local_day_range_utc(days_ago: int = 0) -> tuple[datetime, datetime]:
    """某一天（台北時間）對應到資料庫的 UTC naive 起訖時間。

    這裡有個容易錯的地方：food_logs.logged_at 是 timestamp without time zone，
    存進去的是 UTC 的 naive datetime。但使用者說的「今天」是台北時間的今天，
    兩者差 8 小時——直接拿 UTC 的 00:00 當界線，
    會把台灣時間早上 8 點前記錄的早餐算成前一天。
    """
    day = datetime.now(LOCAL_TZ).date() - timedelta(days=days_ago)
    start_local = datetime.combine(day, time.min, tzinfo=LOCAL_TZ)
    end_local = start_local + timedelta(days=1)
    to_db = lambda dt: dt.astimezone(timezone.utc).replace(tzinfo=None)  # noqa: E731
    return to_db(start_local), to_db(end_local)


def _build_user_context(db: Session, user_id: int) -> str:
    """把使用者今天的飲食紀錄與目標整理成一段文字。

    刻意輸出「人看得懂的文字」而不是 JSON：LLM 讀自然語言的表現比讀巢狀結構好，
    而且之後在 Langfuse 上檢視 trace 時，人也比較容易一眼看出餵了什麼進去。
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        return "（查無此使用者）"

    start, end = _local_day_range_utc()
    logs = (
        db.query(FoodLog)
        .filter(
            FoodLog.user_id == user_id,
            FoodLog.logged_at >= start,
            FoodLog.logged_at < end,
        )
        .order_by(FoodLog.logged_at)
        .all()
    )

    lines = [f"日期：{datetime.now(LOCAL_TZ).date()}（台北時間）"]

    if not logs:
        lines.append("今日尚無飲食紀錄。")
    else:
        lines.append("今日已記錄的餐點：")
        for log in logs:
            # 存的是 naive UTC，要轉回台北時間才是使用者認知的用餐時間
            local_time = log.logged_at.replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ)
            lines.append(
                f"- {local_time:%H:%M} {log.raw_input_text or '（無描述）'}："
                f"熱量 {log.calories:.0f} 大卡、蛋白質 {log.protein_g:.0f}g、"
                f"碳水 {log.carbs_g:.0f}g、脂肪 {log.fat_g:.0f}g"
            )
        # 加總在 Python 端算，不讓 LLM 自己加——跟 extract_food_log 同樣的理由
        lines.append(
            f"今日合計：熱量 {sum(x.calories for x in logs):.0f} 大卡、"
            f"蛋白質 {sum(x.protein_g for x in logs):.0f}g、"
            f"碳水 {sum(x.carbs_g for x in logs):.0f}g、"
            f"脂肪 {sum(x.fat_g for x in logs):.0f}g"
        )

    if user.daily_calories_goal:
        lines.append(f"每日熱量目標：{user.daily_calories_goal:.0f} 大卡")
    if user.current_weight_kg:
        lines.append(
            f"每日蛋白質目標：約 {user.current_weight_kg * PROTEIN_G_PER_KG:.0f}g"
            f"（推估，依體重 {user.current_weight_kg:.0f}kg × {PROTEIN_G_PER_KG} g/kg）"
        )

    return "\n".join(lines)


def _build_knowledge_context(question: str, top_k: int) -> tuple[str, list[RetrievedChunk]]:
    """向量檢索營養知識庫，組成 context 字串。

    同時回傳原始 chunk list，讓呼叫端可以檢查「到底撈到了什麼」——
    QA 答錯時第一件要確認的就是檢索有沒有撈對，沒有這份資料無從判斷。
    """
    chunks = search(question, top_k=top_k)
    if not chunks:
        return "（無相關資料）", []
    # 每筆加 [n] 編號，讓模型回答時能標註數值出處。
    # 這不只是可讀性——有編號才能在 eval 階段用程式自動檢查「這個數字有沒有來源」，
    # 不必人工逐句比對。沒有引用標記的營養數值就是模型自己編的。
    text = "\n".join(f"[{i}] {chunk.content}" for i, chunk in enumerate(chunks, start=1))
    return text, chunks


@observe()
def answer_question(question: str, user_id: int, top_k: int = 5) -> QAResult:
    """回答一個飲食相關問題。

    Args:
        question: 使用者的自然語言問題。
        user_id: 要查誰的飲食紀錄。
        top_k: 從營養知識庫檢索幾筆。

    Returns:
        QAResult，含最終答案與兩份 context（除錯與 eval 用）。
    """
    db = SessionLocal()
    try:
        user_context = _build_user_context(db, user_id)
    finally:
        db.close()

    knowledge_context, chunks = _build_knowledge_context(question, top_k)

    # 兩份 context 一定要明確分開。混在一起的話，
    # LLM 很容易把知識庫裡的食物講成「你今天吃了柴魚片」。
    #
    # 用 XML 標籤而不是【】中文括號：標籤在訓練語料裡的出現頻率高得多，
    # 模型對它的邊界感更強，context 變長時比較不會「串台」。
    # 問題放在最後也是刻意的——長 context 下指令緊接在結尾比較不會被忽略。
    user_message = (
        f"<user_food_log>\n{user_context}\n</user_food_log>\n\n"
        f"<nutrition_reference>\n{knowledge_context}\n</nutrition_reference>\n\n"
        f"<question>\n{question}\n</question>"
    )

    response = openai_client.chat.completions.create(
        model=settings.openai_model_cheap,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return QAResult(
        question=question,
        answer=response.choices[0].message.content or "",
        user_context=user_context,
        knowledge_context=knowledge_context,
        retrieved=chunks,
    )


if __name__ == "__main__":
    # poetry run python -m app.rag.qa
    # 這五題就是問句盤點的結果，涵蓋「純向量 / 純 SQL / 兩者都要」三種類型
    from langfuse.decorators import langfuse_context

    QUESTIONS = [
        "豆漿的熱量是多少？",
        "我今天吃了什麼？總共多少熱量？",
        "我今天蛋白質吃夠了嗎？",
        "有什麼高蛋白低卡的食物推薦？",
        "我吃的那個叫什麼粿的熱量多少？",
    ]

    for q in QUESTIONS:
        result = answer_question(q, user_id=1)
        print(f"\n{'=' * 70}\nQ: {q}\n{'-' * 70}\n{result.answer}")
        print(f"[檢索到] {', '.join(c.content[:18] for c in result.retrieved)}")

    langfuse_context.flush()
