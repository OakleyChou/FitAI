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

