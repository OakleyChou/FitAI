"""當日摘要的計算與端點測試。

`summarize_day` 是純函式，所以時區換算與加總可以不開資料庫就驗證。
端點測試另外守一件單元測試看不到的事：**路由順序**。
"""

from datetime import date, datetime, timezone

import pytest

from app.models import FoodLog, User
from app.nutrition import summarize_day
from app.schemas import InputType

DAY = date(2026, 10, 15)


def _log(hour_utc: int, calories=500.0, protein=30.0, text="測試餐", log_id=1) -> FoodLog:
    return FoodLog(
        id=log_id,
        user_id=1,
        # 資料庫存的是 naive UTC（見 pitfalls #33）
        logged_at=datetime(2026, 10, 15, hour_utc, 0, tzinfo=timezone.utc).replace(
            tzinfo=None
        ),
        calories=calories,
        protein_g=protein,
        carbs_g=50.0,
        fat_g=20.0,
        raw_input_text=text,
        input_type=InputType.text,
    )


def _user(goal=2000.0, weight=60.0) -> User:
    return User(
        id=1,
        email="t@example.com",
        height_cm=170.0,
        current_weight_kg=weight,
        daily_calories_goal=goal,
    )


# ── 純函式 ────────────────────────────────────────────────────


def test_合計與剩餘額度():
    s = summarize_day([_log(1, 420.0), _log(4, 680.0), _log(11, 520.0)], _user(), DAY)

    assert s.totals.calories == 1620
    assert s.calories_goal == 2000
    assert s.calories_remaining == 380


def test_吃超過時剩餘額度是負數():
    """不要夾到 0——超了多少是使用者最需要知道的資訊。"""
    s = summarize_day([_log(1, 2500.0)], _user(goal=2000.0), DAY)

    assert s.calories_remaining == -500


def test_時間換算成台北時間():
    """UTC 00:00 是台北時間早上 8 點。

    直接顯示資料庫的值會讓早餐看起來在凌晨吃的。
    """
    s = summarize_day([_log(0)], _user(), DAY)

    assert s.meals[0].logged_at_local == "08:00"


def test_蛋白質目標是推估值並標記():
    """體重 60kg × 1.6 = 96g。使用者沒設定過這個數字，前端要標明。"""
    s = summarize_day([], _user(weight=60.0), DAY)

    assert s.protein_goal_g == pytest.approx(96.0)
    assert s.protein_goal_is_estimated is True


def test_沒有紀錄不是錯誤():
    """「今天還沒吃東西」是正常狀態，合計為 0、額度等於目標。"""
    s = summarize_day([], _user(), DAY)

    assert s.meals == []
    assert s.totals.calories == 0
    assert s.calories_remaining == 2000


def test_查無使用者時目標為None():
    s = summarize_day([_log(1)], None, DAY)

    assert s.totals.calories == 500
    assert s.calories_goal is None
    assert s.calories_remaining is None
    assert s.protein_goal_is_estimated is False


# ── 端點 ──────────────────────────────────────────────────────


def test_summary端點不會被當成food_log_id(client, user):
    """路由順序的回歸測試。

    FastAPI 依宣告順序比對路由。`/food_logs/{food_log_id}` 若宣告在
    `/food_logs/summary` 之前，這個請求會先命中它，然後 "summary"
    轉 int 失敗回 422——而錯誤訊息只會說「路徑參數不是合法整數」，
    完全看不出是路由順序的問題。
    """
    r = client.get("/food_logs/summary")

    assert r.status_code == 200, f"被 /{{food_log_id}} 攔截了：{r.text}"
    assert "totals" in r.json()


def test_summary回傳使用者目標(client, user):
    r = client.get("/food_logs/summary")

    body = r.json()
    assert body["calories_goal"] == 1800.0  # conftest 的 user fixture
    assert body["protein_goal_is_estimated"] is True


def test_summary的user_id不由客戶端決定(client, user):
    """跟 /chat 一樣：讓客戶端指定 user_id 就會洩漏別人的資料。"""
    r = client.get("/food_logs/summary", params={"user_id": 999})

    # 999 被忽略，仍然回 user 1 的目標
    assert r.json()["calories_goal"] == 1800.0
