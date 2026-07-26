from datetime import datetime
from typing import Optional
from sqlalchemy import Enum as SqlEnum
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base
from app.schemas import InputType


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True)
    height_cm: Mapped[float] = mapped_column()
    current_weight_kg: Mapped[float] = mapped_column()
    target_weight_kg: Mapped[Optional[float]] = mapped_column(nullable=True)
    daily_calories_goal: Mapped[Optional[float]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class FoodLog(Base):
    __tablename__ = "food_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    logged_at: Mapped[datetime]

    calories: Mapped[float] = mapped_column()
    protein_g: Mapped[float] = mapped_column()
    carbs_g: Mapped[float] = mapped_column()
    fat_g: Mapped[float] = mapped_column()
    raw_input_text: Mapped[Optional[str]] = mapped_column(nullable=True)
    input_type: Mapped[InputType] = mapped_column(SqlEnum(InputType))
  

class WeightLog(Base):
    __tablename__ = "weight_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    weight_kg: Mapped[float] = mapped_column()
    logged_at: Mapped[datetime] = mapped_column()


class PromptVersion(Base):
    """Prompt 版本紀錄。name 用來區分不同用途的 prompt（例如
    "food_extraction_system_prompt"），同一個 name 底下可以有多筆不同
    version，is_active 標記目前生效的是哪一筆——換版本時新增一筆新資料、
    把舊的 is_active 改 False，而不是直接覆蓋舊內容。
    """

    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    version: Mapped[int]
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    is_active: Mapped[bool] = mapped_column(default=True)
