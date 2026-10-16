"""當日飲食摘要的計算。純函式，不碰資料庫。

## 為什麼不讓前端自己算

前端拿得到 `GET /food_logs` 的清單，自己篩日期、自己加總，看起來更省事。
但那會讓「今天吃了多少」「還剩多少額度」這些計算有**兩份實作**，
而只有後端那份有測試和 eval 守著。兩份遲早會分岔，而分岔的那天
使用者在對話頁看到「還剩 380 大卡」、在總覽頁看到別的數字。

通則：**可以被兩個地方計算的東西，就只能有一個地方計算它。**

## 為什麼是純函式

輸入是已經查好的 logs 與 user，不自己開 session。這樣：
單元測試不需要資料庫、時區換算與加總邏輯可以單獨驗證、
而「怎麼查」留給呼叫端決定（HTTP 端點用 DI 的 session，工具用自己的）。
"""

from datetime import date, datetime, timezone

from app.models import FoodLog, User
from app.rag.qa import LOCAL_TZ, PROTEIN_G_PER_KG
from app.schemas import DaySummary, MealEntry, NutrientTotals


def summarize_day(logs: list[FoodLog], user: User | None, day: date) -> DaySummary:
    """把某一天的紀錄與使用者目標整理成結構化摘要。

    Args:
        logs: 已經篩選過日期的紀錄，呼叫端負責查詢與排序。
        user: 使用者，用來取得目標。查無此人時目標欄位為 None。
        day: 這批紀錄屬於哪一天（台北時間），只用於回報。

    Returns:
        DaySummary。沒有紀錄時 meals 是空清單、合計全為 0——
        刻意不回 404：「今天還沒吃東西」是一個正常狀態，不是錯誤。
    """
    meals = [
        MealEntry(
            id=log.id,
            # 存的是 naive UTC，要轉回台北時間才是使用者認知的用餐時間。
            # 直接顯示資料庫的值會讓早餐看起來在凌晨吃的（見 pitfalls #33）。
            logged_at_local=log.logged_at.replace(tzinfo=timezone.utc)
            .astimezone(LOCAL_TZ)
            .strftime("%H:%M"),
            description=log.raw_input_text or "（無描述）",
            calories=log.calories,
            protein_g=log.protein_g,
            carbs_g=log.carbs_g,
            fat_g=log.fat_g,
        )
        for log in logs
    ]

    # 加總一律在 Python 端算，不讓 LLM 自己加——跟 extract_food_log 同樣的理由。
    totals = NutrientTotals(
        calories=sum(x.calories for x in logs),
        protein_g=sum(x.protein_g for x in logs),
        carbs_g=sum(x.carbs_g for x in logs),
        fat_g=sum(x.fat_g for x in logs),
    )

    calories_goal = user.daily_calories_goal if user else None
    # 蛋白質目標是推估值（體重 × 1.6 g/kg），不是使用者設定的。
    # 常數從 qa 匯入而不是在這裡再寫一次：公式散在兩處的話，
    # 改了一邊沒改另一邊，對話頁和總覽頁就會報不同的目標。
    protein_goal_g = (
        user.current_weight_kg * PROTEIN_G_PER_KG
        if user and user.current_weight_kg
        else None
    )

    return DaySummary(
        date=day.isoformat(),
        meals=meals,
        totals=totals,
        calories_goal=calories_goal,
        protein_goal_g=protein_goal_g,
        # 可以是負數——吃超過了就該讓使用者看到超了多少，不要夾到 0。
        calories_remaining=(
            calories_goal - totals.calories if calories_goal is not None else None
        ),
        protein_goal_is_estimated=protein_goal_g is not None,
    )


def local_today() -> date:
    """台北時間的今天。伺服器時區可能不是台北，不能用 date.today()。"""
    return datetime.now(LOCAL_TZ).date()
