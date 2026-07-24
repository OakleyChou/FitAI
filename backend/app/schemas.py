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

