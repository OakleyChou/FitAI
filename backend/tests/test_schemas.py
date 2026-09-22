from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas import FoodLog, InputType, User


def test_user_minimal_fields():
    user = User(
        id=1,
        email="alice@example.com",
        height_cm=165.0,
        current_weight_kg=60.0,
        created_at=datetime(2026, 7, 17),
    )
    assert user.target_weight_kg is None
    assert user.daily_calories_goal is None


def test_user_all_fields():
    user = User(
        id=1,
        email="alice@example.com",
        height_cm=165.0,
        current_weight_kg=60.0,
        target_weight_kg=55.0,
        daily_calories_goal=1800,
        created_at=datetime(2026, 7, 17),
    )
    assert user.target_weight_kg == 55.0
    assert user.daily_calories_goal == 1800


def test_user_missing_required_field():
    with pytest.raises(ValidationError):
        User(  # type: ignore[call-arg]
            id=1,
            height_cm=165.0,
            current_weight_kg=60.0,
            created_at=datetime(2026, 7, 17),
        )


def test_food_log_valid():
    log = FoodLog(
        id=1,
        user_id=1,
        logged_at=datetime(2026, 7, 17, 12, 30),
        calories=650.0,
        protein_g=35.0,
        carbs_g=70.0,
        fat_g=20.0,
    )
    assert log.raw_input_text is None
    # 兩種寫法都斷言，是因為 InputType 繼承 str——
    # 它既等於 enum 成員，也等於字串。後者是 JSON 序列化時實際會拿到的形式。
    assert log.input_type == InputType.text
    assert log.input_type == "text"


def test_food_log_negative_calories_rejected():
    with pytest.raises(ValidationError):
        FoodLog(
            id=1,
            user_id=1,
            logged_at=datetime(2026, 7, 17, 12, 30),
            calories=-100.0,
            protein_g=35.0,
            carbs_g=70.0,
            fat_g=20.0,
        )


def test_food_log_invalid_input_type_rejected():
    with pytest.raises(ValidationError):
        FoodLog(
            id=1,
            user_id=1,
            logged_at=datetime(2026, 7, 17, 12, 30),
            calories=650.0,
            protein_g=35.0,
            carbs_g=70.0,
            fat_g=20.0,
            input_type="voice",  # type: ignore[arg-type]
        )


def test_food_log_with_photo_input():
    log = FoodLog(
        id=1,
        user_id=1,
        logged_at=datetime(2026, 7, 17, 12, 30),
        calories=650.0,
        protein_g=35.0,
        carbs_g=70.0,
        fat_g=20.0,
        raw_input_text="一碗雞肉飯",
        input_type=InputType.photo,
    )
    assert log.input_type == InputType.photo
    assert log.raw_input_text == "一碗雞肉飯"
