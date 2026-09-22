"""工具的 JSON Schema，以及名稱到函式的派送表。

送給模型的 `tools` 參數需要 JSON Schema。這裡**手寫**而不是從型別註記自動產生：
四個工具的規模下，手寫一份常數比引入自動化更好讀，而且每個欄位的 description
都是寫給模型看的指令，值得逐字斟酌——自動產生反而會讓人忽略這件事。

description 的內容濃縮自 tools.py 的 docstring。兩邊要同步維護是這個做法的代價，
換來的是可以針對模型的實際行為單獨調整說明，不必動到給人讀的文件。
"""

from typing import Any, Callable

from app.agent.tools import (
    ALLOWED_NUTRIENTS,
    ALLOWED_ORDERS,
    query_food_logs,
    rank_by_nutrient,
    search_nutrition,
    summarize_food_logs,
)

# 名稱 → 實際函式。迴圈拿到模型的呼叫請求後靠這張表派送。
# 用明確的 dict 而不是 globals() 查找：只有列在這裡的函式可被呼叫，
# 模型編出不存在的工具名時會直接查不到，而不是意外執行到別的東西。
TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "query_food_logs": query_food_logs,
    "search_nutrition": search_nutrition,
    "rank_by_nutrient": rank_by_nutrient,
    "summarize_food_logs": summarize_food_logs,
}

# user_id 不放進 schema——它由伺服器端從登入狀態決定，不該讓模型指定。
# 模型能指定 user_id 的話，一句「查 user 2 的紀錄」就會洩漏別人的資料。
SERVER_PROVIDED_ARGS = {"user_id"}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_food_logs",
            "description": (
                "查詢使用者某一天的飲食紀錄與營養目標。"
                "使用者問到「我」「今天」「昨天」吃了什麼、吃了多少、有沒有達標時使用。"
                "回傳的是事實——使用者實際吃過的東西，其中的合計數字已經算好，"
                "直接引用即可，不要自己重新加總。"
                "只查單一一天；跨天的比較或平均請改用 summarize_food_logs。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "往前推幾天。0 是今天、1 是昨天，依台北時間計算。",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_nutrition",
            "description": (
                "用語意相似度在衛福部食藥署營養成分資料庫裡找食物。"
                "使用者提到某個具體食物、想知道它的營養成分時使用。"
                "擅長處理模糊或不精確的名稱（使用者說「洋薏仁」，資料庫登記「大麥仁」也找得到）。"
                "【重要】不要用這個工具回答「最高」「最低」「前幾名」「高蛋白」「低卡」"
                "這類比較或排序的問題——語意相似度不會比較數字大小，"
                "查「高蛋白食物」會撈到名稱裡有「蛋白」的雞蛋白（蛋白質僅 10.7 公克），"
                "而不是蛋白質真正高的食物。那類問題一律用 rank_by_nutrient。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "食物名稱或描述，使用者的原話即可，不需要先清理。",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "回傳幾筆，預設 5。",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_by_nutrient",
            "description": (
                "依某個營養素的含量排序，找出含量最高的食物，可加上熱量上限。"
                "使用者問「有什麼高蛋白的食物」「低卡又有飽足感的」「哪些食物纖維最多」"
                "這類比較或排序的問題時使用。"
                "它做的是資料庫的數值排序而非語意比對，結果保證符合數值條件。"
                "【重要】order 參數決定由高到低還是由低到高，問「低醣」「低鈉」"
                "這類問題時必須傳 asc，否則會拿到完全相反的結果。"
                "查無符合條件的食物時會回傳空清單——這時要如實告知使用者，"
                "不要改用你自己記憶中的營養知識來回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nutrient": {
                        "type": "string",
                        # enum 讓模型只能從這幾個值裡選。少了它，模型很容易
                        # 寫出 "protein" 這種看似合理但不存在的欄位名。
                        "enum": sorted(ALLOWED_NUTRIENTS),
                        "description": "要排序的營養素欄位名。",
                    },
                    "order": {
                        "type": "string",
                        "enum": sorted(ALLOWED_ORDERS),
                        "description": (
                            "排序方向，決定拿到的是最高還是最低。"
                            "desc（預設）由高到低，用於「高蛋白」「纖維最多」；"
                            "asc 由低到高，用於「低醣」「低鈉」「最少」。"
                            "用錯方向會拿到完全相反的清單，而它看起來仍然很合理。"
                        ),
                    },
                    "max_calories": {
                        "type": "number",
                        "description": (
                            "每 100 公克的熱量上限，用來表達「低卡」。"
                            "使用者說「低卡」時可用 150 大卡當門檻。"
                        ),
                    },
                    "min_amount": {
                        "type": "number",
                        "description": (
                            "該營養素每 100 公克的最低含量。"
                            "使用者說「高蛋白」時，慣例門檻是 20 公克。"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "回傳幾筆，預設 5。",
                    },
                },
                "required": ["nutrient"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_food_logs",
            "description": (
                "統計使用者一段期間的飲食紀錄：逐日數字、期間總計與日平均。"
                "使用者問到跨天的比較、平均或趨勢時使用，例如"
                "「我這週跟上週比熱量有下降嗎」「最近三天平均吃多少蛋白質」。"
                "【重要】不要用 query_food_logs 呼叫很多次來湊——"
                "那會用完可用的工具呼叫次數，而且多天的數字要你自己加總；"
                "這個工具會算好才給你。"
                "日平均的分母是「有紀錄的天數」，輸出會一併標明涵蓋率。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_days_back": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "區間起點，往前推幾天（較早的那端，數字較大）。"
                            "例如「最近七天」是 start_days_back=6、end_days_back=0；"
                            "「上一週」是 start_days_back=13、end_days_back=7。"
                        ),
                    },
                    "end_days_back": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "區間終點，往前推幾天（較晚的那端，數字較小），預設 0。",
                    },
                },
                "required": ["start_days_back"],
            },
        },
    },
]
