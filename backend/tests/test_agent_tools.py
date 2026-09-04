"""`_summarize_logs` 的單元測試。

這是專案第一個真正的純邏輯測試。之所以測得起來，是因為
`summarize_food_logs` 把「查資料庫」和「算統計」拆開了——
純函式可以餵假資料，不必起 Postgres，也不受「相對日期的測試資料會漂移」
影響（見 docs/pitfalls.md 第 3 條）。

測的是三個設計決策，不是實作細節：
1. 日平均的分母用「有紀錄的天數」而不是「區間天數」
2. 分組依台北日期，不是 UTC 日期
3. 空區間與區間顛倒的行為
"""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.agent.tools import _summarize_logs, summarize_food_logs
from app.models import FoodLog

LOCAL_TZ = ZoneInfo("Asia/Taipei")


def _log(taipei_time: str, calories: float) -> FoodLog:
    """建一筆假紀錄。

    傳入台北時間，轉成資料庫實際存的 naive UTC——
    這樣測試裡看到的時間跟使用者認知一致，讀起來不用心算時差。
    """
    local = datetime.fromisoformat(taipei_time).replace(tzinfo=LOCAL_TZ)
    return FoodLog(
        logged_at=local.astimezone(timezone.utc).replace(tzinfo=None),
        calories=calories,
        protein_g=calories / 20,
        carbs_g=calories / 10,
        fat_g=calories / 30,
        raw_input_text="測試餐點",
    )


def test_日平均分母用有紀錄的天數而非區間天數():
    """四天的區間裡只有兩天有紀錄，平均要除以 2 不是除以 4。

    這是刻意的設計：使用者忘記記錄的日子不該被當成吃了 0 大卡。
    除以 4 會得到 750，模型會據此說「你吃太少了」——但那是假的。
    """
    logs = [_log("2026-09-01 12:00", 1000), _log("2026-09-02 12:00", 2000)]

    out = _summarize_logs(
        logs, first_day=date(2026, 9, 1), last_day=date(2026, 9, 4), span_days=4
    )

    assert "1500" in out, "日平均應為 3000/2=1500，不是 3000/4=750"
    assert "750" not in out
    # 涵蓋率一定要寫出來，模型才知道這個平均可信到什麼程度
    assert "4 天中有 2 天" in out


def test_分組依台北日期而非UTC日期():
    """台北時間早上 7 點的紀錄，要算在當天。

    這筆的 naive UTC 是 2026-09-02 23:00（前一天），
    直接取 .date() 會歸到 9/02——那正是「早餐憑空消失」的成因。
    """
    logs = [_log("2026-09-03 07:00", 500)]

    out = _summarize_logs(
        logs, first_day=date(2026, 9, 3), last_day=date(2026, 9, 3), span_days=1
    )

    assert "2026-09-03" in out
    assert "2026-09-02" not in out


def test_空區間不會除以零():
    """沒有任何紀錄時要走專門的分支，不能讓 kcal/0 炸掉。"""
    out = _summarize_logs(
        [], first_day=date(2026, 9, 1), last_day=date(2026, 9, 4), span_days=4
    )

    assert "沒有飲食紀錄" in out


def test_逐日小計各自獨立():
    """同一天的多筆要合併，不同天的不能混在一起。"""
    logs = [
        _log("2026-09-01 08:00", 400),
        _log("2026-09-01 12:00", 600),
        _log("2026-09-02 12:00", 300),
    ]

    out = _summarize_logs(
        logs, first_day=date(2026, 9, 1), last_day=date(2026, 9, 2), span_days=2
    )

    assert "1000" in out, "9/01 應合併成 400+600=1000"
    assert "1300" in out, "區間總計應為 1300"


def test_區間顛倒要報錯():
    """start_days_back 是較早的那端，比 end 小代表參數寫反了。

    模型很可能把 start/end 理解成「時間順序的開始與結束」而傳反，
    所以要擋下來並在訊息裡說明數字的意義，讓它能自我修正。
    """
    with pytest.raises(ValueError, match="區間顛倒"):
        summarize_food_logs(user_id=1, start_days_back=0, end_days_back=3)
