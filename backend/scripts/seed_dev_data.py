"""開發用假資料：讓 QA 有東西可以查。

跑法（在 backend/ 底下）：
    poetry run python -m scripts.seed_dev_data

冪等：每次執行會先刪掉 user 1 的所有 food_logs 再重寫，重跑不會累積。
之後 P2 階段寫 README 時，這支腳本就是「clone 下來怎麼跑起來」的一環。
"""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.database import SessionLocal
from app.models import FoodLog, User
from app.schemas import InputType

LOCAL_TZ = ZoneInfo("Asia/Taipei")
DEV_USER_ID = 1

# (幾天前, 幾點, 描述, 熱量, 蛋白質, 碳水, 脂肪)
# 數值是常見台式餐點的粗估，不需精確——這是拿來測 QA 流程的，不是營養資料
MEALS = [
    (0, 8, "鮪魚蛋吐司加一杯無糖豆漿", 420, 22, 45, 16),
    (0, 12, "雞胸肉便當，配燙青菜", 680, 42, 78, 18),
    (1, 8, "蛋餅加米漿", 480, 14, 62, 19),
    (1, 12, "牛肉麵", 720, 38, 85, 24),
    (1, 19, "滷肉飯加燙青菜", 620, 20, 80, 22),
    (2, 8, "火腿三明治配拿鐵", 450, 18, 48, 20),
    (2, 12, "排骨飯", 850, 32, 95, 35),
    (2, 19, "水餃十顆", 560, 24, 62, 22),
    (3, 9, "無糖豆漿配飯糰", 510, 16, 74, 15),
    (3, 13, "麻辣燙", 690, 28, 66, 32),
    (3, 20, "鹹酥雞小份加珍奶", 980, 22, 98, 52),
]


def _to_db_datetime(days_ago: int, hour: int) -> datetime:
    """把「幾天前的台北時間幾點」轉成資料庫存的 naive UTC datetime。

    food_logs.logged_at 是 timestamp without time zone，存的是 UTC 但沒有 tzinfo，
    所以這裡算完要 .replace(tzinfo=None) 把時區資訊拿掉才對得上。
    """
    day = datetime.now(LOCAL_TZ).date() - timedelta(days=days_ago)
    local = datetime.combine(day, time(hour=hour), tzinfo=LOCAL_TZ)
    return local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def seed() -> int:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == DEV_USER_ID).first()
        if user is None:
            raise ValueError(f"找不到 user_id={DEV_USER_ID}，請先建立測試使用者")

        # 補上目標值——QA 要回答「吃夠了嗎」需要比較基準
        user.daily_calories_goal = 2000.0
        if not user.current_weight_kg:
            user.current_weight_kg = 70.0

        # 先刪再寫，保持冪等
        db.query(FoodLog).filter(FoodLog.user_id == DEV_USER_ID).delete()

        logs = [
            FoodLog(
                user_id=DEV_USER_ID,
                logged_at=_to_db_datetime(days_ago, hour),
                calories=kcal,
                protein_g=protein,
                carbs_g=carbs,
                fat_g=fat,
                raw_input_text=text,
                input_type=InputType.text,
            )
            for days_ago, hour, text, kcal, protein, carbs, fat in MEALS
        ]
        db.add_all(logs)
        db.commit()
        return len(logs)
    finally:
        db.close()


if __name__ == "__main__":
    count = seed()
    print(f"seeded {count} food logs for user {DEV_USER_ID}")
