"""Agent 可以呼叫的工具。介面定義階段——簽章與 docstring 已定，實作待 9/01。

## 架構決策：有限迴圈的 tool calling

依 8/26 的問句盤點（5 題中 2 題純向量、2 題純 SQL、1 題混合），
選擇「給 LLM 工具、但迴圈上限 3 輪」而不是純路由或自主 agent：

- **純路由不夠**：分類器處理不了沒預想過的問題，加一種問題就要改分類邏輯。
- **自主 agent 太過**：路徑本身變成變因，eval 會變得難以處理；而且錯誤會累積
  （每步 95% 正確，5 步後只剩 77%）。步數無上限等於成功率無下限。
- **有限迴圈剛好**：保留「模型自己決定用哪個工具」的彈性，
  又因為上限固定，成本、延遲、可評估性都還在掌握中。

## 全部唯讀

第一版不提供寫入工具（例如「幫我記錄這餐」）。讀取工具出錯最多是答錯，
寫入工具出錯會污染資料庫。等工具選擇的行為穩定、eval 涵蓋足夠再考慮。

## docstring 就是 prompt

用 function calling 時，LLM 看到的工具說明來自這些 docstring 和型別註記——
它們會被轉成 JSON schema 送進 prompt。所以下面每一句「什麼時候該用這個工具」
都是寫給模型看的指令，不是寫給人看的註解。

工具選錯時，第一個要改的是 docstring，不是 system prompt。
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import Float, cast

from app.database import SessionLocal
from app.models import FoodLog, KnowledgeChunk, User
from app.rag.ingest import SOURCE_FOOD_COMPOSITION

# 這兩個是 qa 模組的私有函式（底線開頭），這裡跨模組引用是刻意的權宜：
# 邏輯已經寫好且驗證過，重寫一份只會製造兩份會分歧的實作。
# 之後若 agent 成為主要入口，該把它們從 qa 搬到共用模組再一起改名。
from app.rag.qa import LOCAL_TZ, _build_user_context, _local_day_range_utc
from app.rag.retrieval import search

# rank_by_nutrient 允許排序的欄位白名單。
# 模型會編出 "protein" 這種不存在的值；直接把它串進查詢既查不到、
# 也是注入風險。用白名單擋掉，並在錯誤訊息裡列出可用值讓模型能自我修正。
ALLOWED_NUTRIENTS = {
    "protein_g",
    "carbs_g",
    "fat_g",
    "fiber_g",
    "sugar_g",
    "sodium_mg",
}


@dataclass
class FoodItem:
    """一筆營養資料庫的食物，附上判斷用的數值。

    回傳 dataclass 而不是純文字，是因為 agent 可能需要接著比較或計算。
    組成給 LLM 讀的文字是呼叫端的事，工具本身回傳結構化資料。
    """

    name: str
    category: str
    calories: float
    protein_g: float | None
    carbs_g: float | None
    fat_g: float | None
    source_ref: str


def _totals(logs: list[FoodLog]) -> tuple[float, float, float, float]:
    """加總一批紀錄的熱量與三大營養素。

    加總一律在 Python 端算，不讓 LLM 自己加——跟 extract_food_log 同樣的理由：
    多項數字相加時模型會漏算或算錯，而這種錯誤在文字裡看不出來。
    """
    return (
        sum(x.calories for x in logs),
        sum(x.protein_g for x in logs),
        sum(x.carbs_g for x in logs),
        sum(x.fat_g for x in logs),
    )


def _to_food_item(meta: dict, source_ref: str | None) -> FoodItem:
    """把 knowledge_chunks 的 meta 轉成 FoodItem。

    search_nutrition 和 rank_by_nutrient 共用——前者拿到的是 RetrievedChunk、
    後者是 ORM 物件，但兩邊的 meta 結構相同，所以轉換邏輯只寫一份。

    數值用 .get() 而不是 meta[...]：缺值時 meta 存的是 None（不是缺 key），
    但萬一之後 schema 變動，.get() 至少不會直接 KeyError。
    """
    return FoodItem(
        name=meta.get("name") or "（未知名稱）",
        category=meta.get("category") or "",
        calories=meta.get("calories") or 0.0,
        protein_g=meta.get("protein_g"),
        carbs_g=meta.get("carbs_g"),
        fat_g=meta.get("fat_g"),
        source_ref=source_ref or "",
    )


def query_food_logs(user_id: int, days_back: int = 0) -> str:
    """查詢使用者某一天的飲食紀錄與營養目標。

    使用者問到「我」「今天」「昨天」「這幾天」吃了什麼、吃了多少、
    有沒有達標時，用這個工具。它回傳的是**事實**——使用者實際吃過的東西。

    Args:
        user_id: 使用者 ID。
        days_back: 往前推幾天。0 是今天、1 是昨天，依此類推。
            日期以台北時間為準。

    Returns:
        一段文字，含當天每一餐的品項與營養素、當日合計、以及每日目標。
        合計是程式算好的，直接引用即可，不要自己重新加總。
        當天沒有紀錄時會明確說明。

    這個工具回傳字串而不是 FoodItem：它的輸出是要給模型讀的敘述，
    不是要拿來排序或計算的結構化資料。
    """
    db = SessionLocal()
    try:
        result = _build_user_context(db, user_id, days_back)
        return result
    
    finally:
        db.close()


def search_nutrition(query: str, top_k: int = 5) -> list[FoodItem]:
    """用語意相似度在衛福部食藥署營養成分資料庫裡找食物。

    使用者提到某個**具體食物**、想知道它的營養成分時用這個工具。
    它擅長處理模糊或不精確的名稱——例如使用者說「洋薏仁」，
    資料庫裡登記的是「大麥仁」，語意檢索仍然找得到。

    **不要用這個工具回答「最高」「最低」「前幾名」這類問題。**
    語意相似度不會比較數字大小：查「高蛋白食物」會撈到名稱裡有「蛋白」
    的食物（例如蛋白質僅 10.7g 的雞蛋白），而不是蛋白質含量真正高的食物。
    那類問題請用 rank_by_nutrient。

    Args:
        query: 食物名稱或描述。使用者的原話即可，不需要先清理。
        top_k: 回傳幾筆，預設 5。

    Returns:
        依相似度排序的食物清單，最相近的在前。

    """
    chunks = search(query, top_k=top_k)
    return [_to_food_item(chunk.meta, chunk.source_ref) for chunk in chunks]


def rank_by_nutrient(
    nutrient: str,
    max_calories: float | None = None,
    min_amount: float | None = None,
    limit: int = 5,
) -> list[FoodItem]:
    """依某個營養素的含量排序，找出含量最高的食物，可加上熱量上限。

    使用者問「有什麼**高蛋白**的食物」「**低卡**又有飽足感的」
    「哪些食物**纖維最多**」這類**比較或排序**的問題時，用這個工具。

    它做的是資料庫的數值排序，不是語意比對，所以結果保證符合數值條件。
    這是 search_nutrition 做不到的事。

    Args:
        nutrient: 要排序的營養素，可用值：
            "protein_g"（蛋白質）、"carbs_g"（碳水）、"fat_g"（脂肪）、
            "fiber_g"（膳食纖維）、"sugar_g"（糖）、"sodium_mg"（鈉）。
        max_calories: 每 100 公克的熱量上限，用來表達「低卡」。
            使用者說「低卡」時可用 150 大卡當預設門檻。
        min_amount: 該營養素每 100 公克的最低含量。
            使用者說「高蛋白」時，慣例門檻是 20 公克。
        limit: 回傳幾筆，預設 5。

    Returns:
        依該營養素含量由高到低排序的食物清單。
        沒有食物符合條件時回傳空清單——這時要如實告訴使用者查無符合的食物，
        不要改用你自己記憶中的營養知識來回答。

    """
    if nutrient not in ALLOWED_NUTRIENTS:
        # 錯誤訊息列出可用值，讓模型能自我修正後重試
        raise ValueError(f"不支援的營養素 {nutrient!r}，可用值：{sorted(ALLOWED_NUTRIENTS)}")

    # JSONB 取值 → 文字 → 數字。少了 .astext 會拿 JSONB 型別去比較，結果不是你要的。
    amount = cast(KnowledgeChunk.meta[nutrient].astext, Float)
    calories = cast(KnowledgeChunk.meta["calories"].astext, Float)

    db = SessionLocal()
    try:
        query = db.query(KnowledgeChunk).filter(
            # 只搜營養資料庫，之後灌知識文章進來才不會混進結果
            KnowledgeChunk.source == SOURCE_FOOD_COMPOSITION,
            # NULL 在 ORDER BY DESC 會排最前面（Postgres 預設 NULLS FIRST），
            # 不擋掉的話「蛋白質最高的前 5 名」會全是沒測過蛋白質的食物
            amount.isnot(None),
        )

        if max_calories is not None:
            query = query.filter(calories <= max_calories)
        if min_amount is not None:
            query = query.filter(amount >= min_amount)

        query = query.order_by(amount.desc()).limit(limit)
        return [_to_food_item(c.meta, c.source_ref) for c in query.all()]
    finally:
        db.close()


def summarize_food_logs(user_id: int, start_days_back: int, end_days_back: int = 0) -> str:
    """統計使用者一段期間的飲食紀錄：逐日數字與整體平均。

    使用者問到**跨天的比較、平均或趨勢**時用這個工具，例如
    「我這週跟上週比熱量有下降嗎」「最近三天平均吃多少蛋白質」
    「這週哪一天吃最多」。

    **不要用 query_food_logs 呼叫很多次來湊。** 那樣會很快用完可用的工具呼叫次數，
    而且多天的數字要你自己加總——加總請交給這個工具，它算好才給你。

    Args:
        user_id: 使用者 ID。
        start_days_back: 區間起點，往前推幾天（較早的那端）。
            例如「最近七天」是 start_days_back=6、end_days_back=0。
        end_days_back: 區間終點，往前推幾天（較晚的那端），預設 0（今天）。

    Returns:
        一段文字，含區間內每一天的熱量與三大營養素、區間總計與日平均，
        以及有紀錄的天數（沒吃或忘記記錄的日子會拉低平均，所以要標明）。

    """
    if start_days_back < end_days_back:
        raise ValueError(
            f"區間顛倒：start_days_back({start_days_back}) 應該 >= "
            f"end_days_back({end_days_back})，數字越大代表越早"
        )

    # 一次查完整個區間，不要一天查一次——N 次查詢慢，而且結果還要自己併。
    # 起點取較早那天的開始，終點取較晚那天的結束。
    start, _ = _local_day_range_utc(start_days_back)
    _, end = _local_day_range_utc(end_days_back)

    db = SessionLocal()
    try:
        # 先確認使用者存在。少了這段，「查無此人」跟「這個人沒記錄」
        # 會產生一模一樣的輸出，而模型該給的回應完全不同——
        # 前者要說系統找不到帳號，後者要鼓勵開始記錄。
        if db.query(User).filter(User.id == user_id).first() is None:
            return f"查無 user_id={user_id} 這位使用者。"

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
    finally:
        db.close()

    today = datetime.now(LOCAL_TZ).date()
    return _summarize_logs(
        logs,
        first_day=today - timedelta(days=start_days_back),
        last_day=today - timedelta(days=end_days_back),
        span_days=start_days_back - end_days_back + 1,
    )


def _summarize_logs(
    logs: list[FoodLog], first_day: date, last_day: date, span_days: int
) -> str:
    """把一批飲食紀錄整理成逐日與區間統計的文字。純函式，不碰資料庫。

    刻意跟查詢拆開：分組、加總、平均都是純邏輯，拆出來就能用假資料寫單元測試，
    不必起 Postgres、也不受「相對日期的測試資料會漂移」影響
    （見 docs/pitfalls.md 第 3 條）。

    Args:
        logs: 已經篩選好區間的紀錄。
        first_day / last_day: 區間的頭尾（台北日期），只用於顯示。
        span_days: 區間總天數，用來算涵蓋率。
    """
    # 依「使用者認知的那一天」分組。logged_at 存的是 naive UTC，
    # 直接取 .date() 會把台北時間早上 8 點前的紀錄算到前一天。
    by_day: dict[date, list[FoodLog]] = defaultdict(list)
    for log in logs:
        local_day = log.logged_at.replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ).date()
        by_day[local_day].append(log)

    header = f"期間：{first_day} ~ {last_day}（台北時間，共 {span_days} 天）"
    if not by_day:
        return f"{header}\n這段期間完全沒有飲食紀錄。"

    lines = [header, "逐日紀錄："]
    for day in sorted(by_day):
        kcal, protein, carbs, fat = _totals(by_day[day])
        lines.append(
            f"- {day}：熱量 {kcal:.0f} 大卡、蛋白質 {protein:.0f}g、"
            f"碳水 {carbs:.0f}g、脂肪 {fat:.0f}g（{len(by_day[day])} 筆）"
        )

    kcal, protein, carbs, fat = _totals(logs)
    recorded_days = len(by_day)
    lines.append(
        f"期間總計：熱量 {kcal:.0f} 大卡、蛋白質 {protein:.0f}g、"
        f"碳水 {carbs:.0f}g、脂肪 {fat:.0f}g"
    )

    # 平均的分母用「有紀錄的天數」而不是「區間天數」。
    # 使用者忘記記錄的日子不該被當成吃了 0 大卡——那會讓平均嚴重偏低，
    # 而模型看不出這件事，只會照著錯的數字給建議。
    lines.append(
        f"日平均（以有紀錄的 {recorded_days} 天計）：熱量 {kcal / recorded_days:.0f} 大卡、"
        f"蛋白質 {protein / recorded_days:.0f}g、碳水 {carbs / recorded_days:.0f}g、"
        f"脂肪 {fat / recorded_days:.0f}g"
    )
    # 涵蓋率一定要講出來，模型才知道這個平均可信到什麼程度
    lines.append(f"涵蓋率：{span_days} 天中有 {recorded_days} 天留下紀錄。")

    return "\n".join(lines)
