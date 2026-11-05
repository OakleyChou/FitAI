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
    # calories 原本不在清單裡，但「熱量最低的點心」是這個工具最自然的用法。
    # 2026-11-05 實測：agent 呼叫 nutrient="calories" 被擋掉，拿不到任何資料，
    # 然後**編了三個食物與三個 source_ref**（其中 I0410101 是雞絞肉的真實編號，
    # 被挪用到「草莓蛋糕」上）。工具少一個選項，代價是一個看起來完全可稽核的假答案。
    "calories",
    "protein_g",
    "carbs_g",
    "fat_g",
    "fiber_g",
    "sugar_g",
    "sodium_mg",
}

# 排序方向。原本只支援由高到低，結果問「有什麼低醣的東西」時，模型照樣呼叫
# 這個工具、拿到碳水最高的前三名，然後在答案裡宣稱那些是低醣食物——
# 不會拒答，只會給一個反過來的答案，而且講得很有自信。
ALLOWED_ORDERS = {"desc", "asc"}

# 食藥署的 18 個食物分類，用來讓「點心」「調味料」這類條件表達得出來。
#
# 沒有這個篩選時，問「剩 380 大卡可以吃什麼點心」會退化成
# 「任何符合熱量上限的東西」——實測推薦過五味醬、和風沙拉醬，
# 後來又推薦過牛後腿腱子心、豬腦、牛肚。
# 路徑對、數值對、額度也守住，但沒有人會把豬腦當點心。
#
# 值必須跟資料庫裡的字串完全相同（取自 meta->>'category'），
# 所以這裡寫死而不是每次查詢去 distinct——分類是資料集的固定結構，
# 不是會變動的資料。
ALLOWED_CATEGORIES = {
    "穀物類",
    "澱粉類",
    "堅果及種子類",
    "水果類",
    "蔬菜類",
    "藻類",
    "菇類",
    "豆類",
    "肉類",
    "魚貝類",
    "蛋類",
    "乳品類",
    "油脂類",
    "糖類",
    "飲料類",
    "調味料及香辛料類",
    "糕餅點心類",
    "加工調理食品及其他類",
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
    # 這三個原本沒有，但 rank_by_nutrient 可以依它們排序——
    # 不回傳的話，模型拿到「纖維最多的前五名」卻看不到纖維值，
    # 只能略過不提或自己編一個。而 prompt 要求數值必須來自工具結果，
    # 所以它會略過，使用者就看不出排序的依據。
    fiber_g: float | None = None
    sugar_g: float | None = None
    sodium_mg: float | None = None


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
        fiber_g=meta.get("fiber_g"),
        sugar_g=meta.get("sugar_g"),
        sodium_mg=meta.get("sodium_mg"),
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
    order: str = "desc",
    category: str | None = None,
    max_calories: float | None = None,
    min_amount: float | None = None,
    limit: int = 5,
) -> list[FoodItem]:
    """依某個營養素的含量排序（由高到低或由低到高），可加上熱量上限。

    使用者問「有什麼**高蛋白**的食物」「**低卡**又有飽足感的」
    「哪些食物**纖維最多**」「有什麼**低醣**的可以換」這類**比較或排序**的
    問題時，用這個工具。

    它做的是資料庫的數值排序，不是語意比對，所以結果保證符合數值條件。
    這是 search_nutrition 做不到的事。

    Args:
        nutrient: 要排序的營養素，可用值：
            "protein_g"（蛋白質）、"carbs_g"（碳水）、"fat_g"（脂肪）、
            "fiber_g"（膳食纖維）、"sugar_g"（糖）、"sodium_mg"（鈉）。
        order: 排序方向。**這個參數決定你拿到的是最高還是最低，不要用錯。**
            "desc"（預設）由高到低——使用者問「高蛋白」「纖維最多」「最高」時用。
            "asc" 由低到上——使用者問「低醣」「低鈉」「最少」「想換成比較低的」
            時用。用錯方向會拿到完全相反的清單，而它看起來仍然很合理。
        category: 只在這個食物分類裡排序。**使用者講了食物類型就一定要用。**
            沒有篩分類的話，「可以吃什麼點心」會退化成「任何符合熱量上限的東西」，
            實測推薦過豬腦與牛肚——數值全對，但沒有人會把豬腦當點心。
            可用值（食藥署分類，必須完全相同）：
            "穀物類"、"澱粉類"、"堅果及種子類"、"水果類"、"蔬菜類"、"藻類"、
            "菇類"、"豆類"、"肉類"、"魚貝類"、"蛋類"、"乳品類"、"油脂類"、
            "糖類"、"飲料類"、"調味料及香辛料類"、"糕餅點心類"、
            "加工調理食品及其他類"。
            對應常見說法："點心"→"糕餅點心類"；"飲料"→"飲料類"；
            "水果"→"水果類"；"青菜"→"蔬菜類"；"肉"→"肉類"；
            "海鮮"/"魚"→"魚貝類"；"調味料"/"醬料"→"調味料及香辛料類"。
            使用者只說「東西」「食物」而沒指定類型時，不要傳這個參數。
        max_calories: 每 100 公克的熱量上限，用來表達「低卡」。
            使用者說「低卡」時可用 150 大卡當預設門檻。
        min_amount: 該營養素每 100 公克的最低含量。
            使用者說「高蛋白」時，慣例門檻是 20 公克。
        limit: 回傳幾筆，預設 5。

    Returns:
        依該營養素含量排序的食物清單，方向由 order 決定。
        沒有食物符合條件時回傳空清單——這時要如實告訴使用者查無符合的食物，
        不要改用你自己記憶中的營養知識來回答。

    """
    if nutrient not in ALLOWED_NUTRIENTS:
        # 錯誤訊息列出可用值，讓模型能自我修正後重試
        raise ValueError(f"不支援的營養素 {nutrient!r}，可用值：{sorted(ALLOWED_NUTRIENTS)}")
    if order not in ALLOWED_ORDERS:
        raise ValueError(f"不支援的排序方向 {order!r}，可用值：{sorted(ALLOWED_ORDERS)}")
    if category is not None and category not in ALLOWED_CATEGORIES:
        # 模型會自己發明「點心類」「飲料」這種近似值。白名單擋掉並列出可用值，
        # 讓它能自我修正——錯誤訊息是模型唯一的回饋管道。
        raise ValueError(
            f"不支援的食物分類 {category!r}，可用值：{sorted(ALLOWED_CATEGORIES)}"
        )

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

        if category is not None:
            # 用 JSONB 的文字比對而不是 LIKE：分類是固定的列舉值，
            # 模糊比對只會讓「肉類」撈到「加工調理食品及其他類」。
            query = query.filter(KnowledgeChunk.meta["category"].astext == category)
        if max_calories is not None:
            query = query.filter(calories <= max_calories)
        if min_amount is not None:
            query = query.filter(amount >= min_amount)

        query = query.order_by(amount.desc() if order == "desc" else amount.asc())
        query = query.limit(limit)
        return [_to_food_item(c.meta, c.source_ref) for c in query.all()]
    finally:
        db.close()


ALLOWED_MEAL_METRICS = {"total", "per_100kcal"}

# 排序「餐」時可用的營養素。跟 ALLOWED_NUTRIENTS 不同：
# food_logs 只存熱量與三大營養素，沒有纖維、糖、鈉。
ALLOWED_MEAL_NUTRIENTS = {"calories", "protein_g", "carbs_g", "fat_g"}


def rank_my_meals(
    user_id: int,
    nutrient: str = "protein_g",
    metric: str = "total",
    days_back: int = 6,
    order: str = "desc",
    limit: int = 3,
) -> str:
    """把**使用者自己吃過的餐點**依某個營養素排序。

    使用者問「我**吃過**什麼蛋白質最高的」「我**哪一餐**熱量最高」
    「我吃過的東西裡哪個蛋白質密度高」這類**針對自己紀錄**的排序問題時，
    用這個工具。

    **不要用 rank_by_nutrient 回答這類問題。** 那個查的是公開營養資料庫，
    回答的會是使用者沒吃過的食物——實測發生過：問「我吃過什麼蛋白質密度最高的
    食物」，agent 用 rank_by_nutrient 回了柴魚片、小魚干，然後在答案裡
    說成「你過去一週吃過的」。兩次工具呼叫都成功、文字也通順，
    只有看路徑才發現它答錯了問題。

    ## 這個工具的兩個資料層限制（回答時要如實說明）

    1. **排序的單位是「餐」，不是「食物」。** `food_logs` 存的是每一餐的加總
       （「鮪魚蛋吐司加一杯無糖豆漿」是一筆，蛋白質 22g 是整餐的）。
       逐項明細沒有存進資料庫，所以「哪樣**食物**最高」嚴格說答不出來。
    2. **沒有每 100 公克的密度。** 紀錄裡沒有餐點重量，所以算不出
       「每 100 公克含多少蛋白質」。能算的是 metric="per_100kcal"
       （每 100 大卡含多少），那是另一種密度——營養素相對於熱量的密度。

    Args:
        user_id: 使用者 ID。
        nutrient: 要排序的營養素。可用值："calories"（熱量）、
            "protein_g"（蛋白質）、"carbs_g"（碳水）、"fat_g"（脂肪）。
            紀錄裡沒有纖維、糖、鈉，問那些要說明資料不足。
        metric: 排序依據。
            "total"（預設）按整餐的絕對量排——回答「哪一餐蛋白質最多」。
            "per_100kcal" 按每 100 大卡的含量排——回答「密度最高」。
            使用者說「密度」「比例」「每卡」時用 per_100kcal；
            說「最多」「最高」而沒提密度時用 total。
        days_back: 往前看幾天，預設 6（含今天共七天）。
        order: "desc"（預設）由高到低；"asc" 由低到高。
        limit: 回傳幾筆，預設 3。
            刻意跟 system prompt 的「列出食物時最多 3 項」對齊——
            工具回傳 5 筆的話模型會把 5 筆都列出來，然後超過字數上限。
            **不要給模型比它被允許使用的更多資料。**

    Returns:
        一段文字，列出排序後的餐點、日期、數值，並附上上述限制的說明。
        沒有紀錄時明確說明——那時要鼓勵使用者開始記錄，不要改去查公開資料庫。

    """
    if nutrient not in ALLOWED_MEAL_NUTRIENTS:
        raise ValueError(
            f"飲食紀錄裡沒有 {nutrient!r} 這個欄位，可用值："
            f"{sorted(ALLOWED_MEAL_NUTRIENTS)}。"
            f"紀錄只存熱量與三大營養素，微量營養素要說明資料不足。"
        )
    if metric not in ALLOWED_MEAL_METRICS:
        raise ValueError(f"不支援的排序依據 {metric!r}，可用值：{sorted(ALLOWED_MEAL_METRICS)}")
    if order not in ALLOWED_ORDERS:
        raise ValueError(f"不支援的排序方向 {order!r}，可用值：{sorted(ALLOWED_ORDERS)}")
    if days_back < 0:
        raise ValueError(f"days_back 不能是負數（收到 {days_back}）")

    start, _ = _local_day_range_utc(days_back)
    _, end = _local_day_range_utc(0)

    db = SessionLocal()
    try:
        # 先確認使用者存在。少了這段，「查無此人」跟「這個人沒記錄」
        # 會產生一樣的輸出，而模型該給的回應完全不同。
        if db.query(User).filter(User.id == user_id).first() is None:
            return f"查無 user_id={user_id} 這位使用者。"

        logs = (
            db.query(FoodLog)
            .filter(
                FoodLog.user_id == user_id,
                FoodLog.logged_at >= start,
                FoodLog.logged_at < end,
            )
            .all()
        )
    finally:
        db.close()

    return _rank_meals(logs, nutrient, metric, order, limit, days_back)


def _meal_value(log: FoodLog, nutrient: str, metric: str) -> float | None:
    """一餐在這個排序依據下的數值。算不出來時回 None。

    per_100kcal 在熱量為 0 時沒有意義（除以零）。回 None 讓呼叫端排除它，
    而不是回 0——0 會被當成「密度很低」排到 asc 的最前面，
    那是一個看起來合理的錯誤答案。
    """
    amount = getattr(log, nutrient)
    if metric == "total":
        return amount
    if not log.calories:
        return None
    return amount / log.calories * 100.0


def _rank_meals(
    logs: list[FoodLog],
    nutrient: str,
    metric: str,
    order: str,
    limit: int,
    days_back: int,
) -> str:
    """把紀錄依指定依據排序成文字。純函式，不碰資料庫。

    跟查詢拆開的理由同 _summarize_logs：排序、密度換算、排除零熱量
    都是純邏輯，拆出來就能用假資料測，不必起 Postgres。
    """
    label = {
        "calories": "熱量",
        "protein_g": "蛋白質",
        "carbs_g": "碳水",
        "fat_g": "脂肪",
    }[nutrient]
    unit = "大卡" if nutrient == "calories" else "公克"
    metric_label = f"{label}總量" if metric == "total" else f"每 100 大卡的{label}"

    header = f"排序依據：{metric_label}（{'由高到低' if order == 'desc' else '由低到高'}），範圍：最近 {days_back + 1} 天"

    if not logs:
        return f"{header}\n這段期間沒有任何飲食紀錄，無法排序。"

    scored = [(log, _meal_value(log, nutrient, metric)) for log in logs]
    usable = [(log, value) for log, value in scored if value is not None]
    skipped = len(scored) - len(usable)

    if not usable:
        return f"{header}\n這段期間的紀錄都無法計算{metric_label}（共 {skipped} 筆）。"

    usable.sort(key=lambda pair: pair[1], reverse=(order == "desc"))

    lines = [header, f"共 {len(usable)} 筆紀錄，列出前 {min(limit, len(usable))} 筆："]
    for i, (log, value) in enumerate(usable[:limit], start=1):
        local_day = log.logged_at.replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ)
        suffix = unit if metric == "total" else f"{unit}/100大卡"
        lines.append(
            f"{i}. {local_day:%m/%d %H:%M} {log.raw_input_text or '（無描述）'}："
            f"{metric_label} {value:.1f} {suffix}"
            f"（整餐熱量 {log.calories:.0f} 大卡）"
        )

    if skipped:
        lines.append(f"（排除 {skipped} 筆熱量為 0、無法計算密度的紀錄）")

    # 這兩行一定要帶出去。模型看到一份排序清單會自然地說成「你吃過的食物排名」，
    # 而這裡排的是「餐」——說錯了就等於把限制講成了功能。
    lines.append(
        "【資料限制】以上排序的單位是**一餐**，不是單一食物。"
        "紀錄存的是每餐加總（例如「鮪魚蛋吐司加一杯無糖豆漿」是一筆），"
        "逐項明細沒有保留，所以無法回答「哪樣食物」最高。"
    )
    if metric == "per_100kcal":
        lines.append(
            "【資料限制】這裡的密度是「每 100 大卡」，不是「每 100 公克」——"
            "紀錄裡沒有餐點重量。"
        )

    return "\n".join(lines)


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
