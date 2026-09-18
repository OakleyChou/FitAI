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
