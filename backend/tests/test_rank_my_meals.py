"""`rank_my_meals` 的排序邏輯。純函式，不碰資料庫。

跟 `_summarize_logs` 同樣的理由拆開：排序、密度換算、排除零熱量都是純邏輯，
拆出來就能用假資料測，不必起 Postgres，也不受「相對日期的測試資料會漂移」影響。
"""

from datetime import datetime, timezone

import pytest

from app.agent.tools import _meal_value, _rank_meals
from app.models import FoodLog
from app.schemas import InputType


def _log(text: str, calories: float, protein: float, hour_utc: int = 4) -> FoodLog:
    return FoodLog(
        id=1,
        user_id=1,
        # 資料庫存的是 naive UTC（見 pitfalls #33）
        logged_at=datetime(2026, 11, 5, hour_utc, 0, tzinfo=timezone.utc).replace(
            tzinfo=None
        ),
        calories=calories,
        protein_g=protein,
        carbs_g=0.0,
        fat_g=0.0,
        raw_input_text=text,
        input_type=InputType.text,
    )


# ── 單餐的數值 ────────────────────────────────────────────────


def test_total是整餐的絕對量():
    assert _meal_value(_log("x", 500, 30), "protein_g", "total") == 30


def test_per_100kcal是每一百大卡的含量():
    """500 大卡、30 公克蛋白質 → 每 100 大卡 6 公克。

    這是「密度」問題真正能回答的版本——紀錄裡沒有餐點重量，
    算不出每 100 公克，但每 100 大卡算得出來。
    """
    assert _meal_value(_log("x", 500, 30), "protein_g", "per_100kcal") == pytest.approx(6.0)


def test_零熱量的餐算不出密度():
    """回 None 而不是 0。

    回 0 的話它會被當成「密度很低」排到 asc 的最前面——
    一個看起來合理的錯誤答案。
    """
    assert _meal_value(_log("x", 0, 10), "protein_g", "per_100kcal") is None
    # total 不受影響，0 大卡但有蛋白質是可能的紀錄
    assert _meal_value(_log("x", 0, 10), "protein_g", "total") == 10


# ── 排序 ──────────────────────────────────────────────────────


def test_依總量由高到低排序():
    logs = [_log("早餐", 400, 20), _log("午餐", 700, 45), _log("晚餐", 500, 30)]

    out = _rank_meals(logs, "protein_g", "total", "desc", 5, 6)

    # 午餐(45) 要排在第一
    assert out.index("午餐") < out.index("晚餐") < out.index("早餐")


def test_依密度排序的結果跟總量不同():
    """這是 metric 參數存在的理由。

    午餐蛋白質總量最高（45g），但高蛋白綜合蛋密度更高（20/150*100=13.3）。
    問「最多」和問「密度最高」該得到不同答案。
    """
    logs = [_log("午餐", 700, 45), _log("高蛋白綜合蛋", 150, 20)]

    by_total = _rank_meals(logs, "protein_g", "total", "desc", 5, 6)
    by_density = _rank_meals(logs, "protein_g", "per_100kcal", "desc", 5, 6)

    assert by_total.index("午餐") < by_total.index("高蛋白綜合蛋")
    assert by_density.index("高蛋白綜合蛋") < by_density.index("午餐")


def test_asc由低到高():
    logs = [_log("多的", 700, 45), _log("少的", 400, 10)]

    out = _rank_meals(logs, "protein_g", "asc", "asc", 5, 6)

    assert out.index("少的") < out.index("多的")


def test_limit會截斷():
    logs = [_log(f"第{i}餐", 500, i * 10) for i in range(1, 6)]

    out = _rank_meals(logs, "protein_g", "total", "desc", 2, 6)

    assert "列出前 2 筆" in out
    assert "第5餐" in out and "第4餐" in out
    assert "第1餐" not in out


def test_零熱量的紀錄被排除並回報():
    logs = [_log("正常", 500, 30), _log("壞資料", 0, 10)]

    out = _rank_meals(logs, "protein_g", "per_100kcal", "desc", 5, 6)

    assert "排除 1 筆" in out
    assert "壞資料" not in out


def test_沒有紀錄時明確說明():
    """不要回空字串。

    空輸出會讓模型以為工具壞了然後改去查公開資料庫——
    那正是這個工具要防的事。
    """
    out = _rank_meals([], "protein_g", "total", "desc", 5, 6)

    assert "沒有任何飲食紀錄" in out


def test_全部都算不出密度時明確說明():
    out = _rank_meals([_log("x", 0, 5)], "protein_g", "per_100kcal", "desc", 5, 6)

    assert "都無法計算" in out


# ── 資料限制一定要帶出去 ──────────────────────────────────────


def test_輸出一定附上單位是餐的說明():
    """模型看到一份排序清單會自然地說成「你吃過的食物排名」。

    而這裡排的是「餐」。工具不講清楚，模型就會把限制講成功能——
    那正是 H 類缺口最初的症狀（把資料庫食物說成使用者吃過的）。
    """
    out = _rank_meals([_log("午餐", 700, 45)], "protein_g", "total", "desc", 5, 6)

    assert "一餐" in out
    # 要指路，不能只說「做不到」。只講限制的話模型會直接回「查不到」，
    # 而其實換一個工具就答得出來。
    assert "rank_my_foods" in out


def test_不可再聲稱逐項明細沒有保留():
    """這句話 2026-11-06 起是假的，而它是**隨工具結果送給模型的**。

    限制的敘述跟程式一樣會過期，差別是它過期時不會報錯：
    模型會把那句話當事實轉述給使用者，變成一個很有自信的錯誤說明。
    用否定斷言釘住它——只斷言「有講限制」的測試擋不住這種情況，
    因為過期的限制敘述也是一段限制敘述。
    """
    for metric in ("total", "per_100kcal"):
        out = _rank_meals([_log("午餐", 700, 45)], "protein_g", metric, "desc", 5, 6)

        assert "逐項明細沒有保留" not in out
        assert "明細沒有保留" not in out


def test_密度模式額外說明是每100大卡():
    out = _rank_meals([_log("午餐", 700, 45)], "protein_g", "per_100kcal", "desc", 5, 6)

    assert "每 100 大卡" in out
    assert "沒有餐點重量" in out


def test_時間顯示為台北時間():
    """UTC 04:00 是台北時間 12:00。"""
    out = _rank_meals([_log("午餐", 700, 45, hour_utc=4)], "protein_g", "total", "desc", 5, 6)

    assert "12:00" in out
