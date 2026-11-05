from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum

class User(BaseModel):
    id: int
    email: str
    height_cm: float
    current_weight_kg: float
    target_weight_kg: Optional[float] = None
    daily_calories_goal: Optional[float] = None
    created_at: datetime


class InputType(str, Enum):
    text = "text"
    photo = "photo"


class FoodLog(BaseModel):
    id: int
    user_id: int
    logged_at: datetime
    calories: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)
    raw_input_text: Optional[str] = None
    input_type: InputType = InputType.text

    model_config = {"from_attributes": True}  # 從SQLAlchemy物件轉換成回應格式


class FoodLogCreate(BaseModel):
    """新增飲食紀錄的 request body —— 使用者只需要傳自然語言文字，
    營養素由後端呼叫 LLM（app.llm.extract_food_log）解析出來，不是使用者自己填。
    """

    user_id: int
    raw_text: str


class FoodLogUpdate(BaseModel):
    """更新飲食紀錄的 request body —— 全部欄位選填，只更新有傳的欄位。"""

    calories: Optional[float] = Field(default=None, ge=0)
    protein_g: Optional[float] = Field(default=None, ge=0)
    carbs_g: Optional[float] = Field(default=None, ge=0)
    fat_g: Optional[float] = Field(default=None, ge=0)



class ExtractedItemInfo(BaseModel):
    """新增紀錄時回報的逐項明細。

    `grounded` 是重點：它區分「數值來自食藥署資料庫」與「LLM 估的」。
    不給前端這個欄位的話，可靠的數字和猜測在畫面上看起來一模一樣，
    使用者就不知道哪一項需要核對。
    """

    name: str
    portion_g: float
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    grounded: bool
    source_ref: Optional[str] = None
    db_name: Optional[str] = None
    note: str = ""


class FoodLogCreated(FoodLog):
    """POST /food_logs 的回應：存好的紀錄 + 這次擷取的逐項明細。

    比 FoodLog 多的欄位都是**不落地**的——food_logs 表只存合計。
    不另外存是刻意的：那需要加資料表欄位，而這個專案用 create_all 沒有
    migration（見 pitfalls，create_all 不會修改既有表結構）。
    而且明細最有價值的時刻就是剛記錄完、使用者還記得自己吃了什麼的時候。
    """

    items: list[ExtractedItemInfo] = Field(default_factory=list)
    grounded_count: int = 0

    model_config = {"from_attributes": True}


class MealEntry(BaseModel):
    """當日摘要裡的一餐。時間已經換算成台北時間的 HH:MM。"""

    id: int
    logged_at_local: str
    description: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


class NutrientTotals(BaseModel):
    calories: float = 0
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0


class DaySummary(BaseModel):
    """某一天的飲食摘要。

    計算由後端負責（見 app/nutrition.py），前端只顯示。
    讓前端自己篩日期加總的話，同一個數字會有兩份實作，
    而只有後端那份有測試守著。
    """

    date: str
    meals: list[MealEntry] = Field(default_factory=list)
    totals: NutrientTotals = Field(default_factory=NutrientTotals)
    calories_goal: Optional[float] = None
    protein_goal_g: Optional[float] = None
    # 可以是負數——吃超過了要讓使用者看到超了多少。
    calories_remaining: Optional[float] = None
    # 蛋白質目標是依體重推估的，不是使用者設定的。前端要標明，
    # 否則使用者會以為那是他自己設過的數字。
    protein_goal_is_estimated: bool = False


class ChatRequest(BaseModel):
    """對話請求。

    刻意**不含 user_id**——那由伺服器端從登入狀態決定（見 app/deps.py）。
    讓客戶端指定的話，一句「查 user 2 的紀錄」就會洩漏別人的資料。
    """

    question: str = Field(min_length=1, max_length=500, description="使用者的問題")


class ToolCallInfo(BaseModel):
    """一次工具呼叫的摘要，給前端除錯或顯示「查詢中」用。

    不含完整的工具結果——那可能上千字，而且是內部細節。
    """

    name: str
    arguments: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """對話回應。

    degraded 的用意：agent 有部分工具失敗、但仍產出了有用的降級回答時，
    仍然回 200（而不是 503），因為那個回答對使用者有價值——
    回 503 前端會當成失敗，反而把它丟掉。前端可以依這個旗標
    顯示「部分功能暫時不可用」的提示。
    """

    answer: str
    degraded: bool = False
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)


class FeedbackSource(str, Enum):
    food_log = "food_log"
    chat = "chat"


class FeedbackKind(str, Enum):
    too_high = "too_high"
    too_low = "too_low"
    wrong_match = "wrong_match"
    other = "other"


class FeedbackCreate(BaseModel):
    """回報一筆「怪怪的」。

    刻意**不含 user_id**——那由伺服器端決定，跟 /chat 同樣的理由。

    `expected_min` / `expected_max` 是這張表最重要的欄位：**eval 的判準就是它**。
    要區間而不是單一數字，因為營養估算沒有唯一正確答案，
    逼使用者給精確值只會讓他亂填。
    """

    source: FeedbackSource
    raw_input: str = Field(min_length=1, max_length=500)
    kind: FeedbackKind
    system_value: Optional[float] = Field(default=None, ge=0)
    expected_min: Optional[float] = Field(default=None, ge=0)
    expected_max: Optional[float] = Field(default=None, ge=0)
    note: Optional[str] = Field(default=None, max_length=1000)
    context: dict = Field(default_factory=dict)


class Feedback(BaseModel):
    id: int
    user_id: int
    created_at: datetime
    source: str
    raw_input: str
    kind: str
    system_value: Optional[float] = None
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None
    note: Optional[str] = None
    context: dict = Field(default_factory=dict)
    converted_to_eval: bool = False

    model_config = {"from_attributes": True}
