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

from dataclasses import dataclass


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

    TODO(9/01): 包裝 qa._build_user_context，把寫死的「今天」改成吃 days_back。
    """
    raise NotImplementedError


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

    TODO(9/01): 包裝 retrieval.search，把 RetrievedChunk 的 meta 轉成 FoodItem。
    """
    raise NotImplementedError


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

    TODO(9/01): 用 JSONB 查詢實作：
        WHERE (meta->>'calories')::float < max_calories
          AND (meta->>nutrient)::float >= min_amount
        ORDER BY (meta->>nutrient)::float DESC
    """
    raise NotImplementedError
