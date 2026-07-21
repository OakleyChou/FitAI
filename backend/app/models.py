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
