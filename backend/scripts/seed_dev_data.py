"""開發用假資料：讓 QA 與 agent 工具有東西可以查。

跑法（在 backend/ 底下）：
    poetry run python -m scripts.seed_dev_data

冪等：每次執行會先刪掉 user 1 的所有 food_logs 再重寫，重跑不會累積。
之後 P2 階段寫 README 時，這支腳本就是「clone 下來怎麼跑起來」的一環。

資料是刻意設計的，不是隨手填的假數字：

- **跨兩週且有明顯落差**（近七天約 1700 大卡、前七天約 2300 大卡），
  這樣「我這週跟上週比熱量有下降嗎」才有可驗證的答案。
- **刻意空兩天**（第 4 天與第 9 天沒有任何紀錄），用來驗證
  summarize_food_logs 的日平均是拿「有紀錄的天數」當分母，
  而不是拿「區間天數」——後者會把忘記記錄的日子算成吃了 0 大卡。
"""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.database import SessionLocal
from app.models import FoodLog, User
from app.schemas import InputType

LOCAL_TZ = ZoneInfo("Asia/Taipei")
DEV_USER_ID = 1

# (幾天前, 幾點, 描述, 熱量, 蛋白質, 碳水, 脂肪)
# 數值是常見台式餐點的粗估，不需精確——這是拿來測流程的，不是營養資料。
MEALS = [
    # ── 近七天：控制飲食中，約 1700 大卡/天 ──────────────────
    (0, 8, "鮪魚蛋吐司加一杯無糖豆漿", 420, 22, 45, 16),
    (0, 12, "雞胸肉便當，配燙青菜", 680, 42, 78, 18),
    (0, 19, "烤鯖魚定食，白飯半碗", 520, 34, 42, 22),
    (1, 8, "蛋餅加無糖豆漿", 480, 18, 52, 19),
    (1, 12, "自助餐：糙米飯、清蒸魚、炒時蔬", 620, 36, 72, 16),
    (1, 19, "雞肉沙拉佐油醋", 500, 32, 30, 24),
    (2, 8, "水煮蛋兩顆加黑咖啡", 450, 20, 38, 22),
    (2, 12, "牛肉燴飯", 700, 34, 88, 20),
    (2, 19, "豆腐鍋，少油", 580, 30, 52, 26),
    (3, 9, "無糖豆漿配茶葉蛋", 400, 24, 34, 16),
    (3, 13, "雞腿便當（去皮）", 650, 40, 74, 18),
    (3, 19, "清燙牛肉麵，麵吃一半", 620, 35, 62, 22),
    # 第 4 天刻意沒有紀錄——測日平均的分母
    (5, 8, "地瓜配希臘優格", 510, 22, 68, 14),
    (5, 12, "海鮮炒麵", 690, 32, 84, 24),
    (5, 19, "滷雞腿加燙青菜", 540, 38, 40, 22),
    (6, 8, "全麥吐司夾蛋", 430, 20, 46, 17),
    (6, 12, "排骨湯麵", 660, 30, 80, 22),
    (6, 19, "蒸蛋加味噌湯配糙米飯", 600, 28, 76, 18),
    # ── 前七天：飲食較放縱，約 2300 大卡/天 ──────────────────
    (7, 8, "培根蛋餅加大冰奶", 620, 18, 66, 30),
    (7, 12, "炸雞腿便當", 850, 38, 92, 34),
    (7, 20, "鹹酥雞加珍奶", 900, 24, 96, 46),
    (8, 8, "美式早餐：鬆餅、培根、柳橙汁", 580, 16, 74, 24),
    (8, 12, "麻辣鍋", 920, 42, 78, 52),
    (8, 19, "義大利肉醬麵", 780, 32, 96, 30),
    # 第 9 天刻意沒有紀錄
    (10, 8, "燒餅油條加豆漿", 650, 16, 72, 32),
    (10, 12, "牛肉麵加滷味", 880, 44, 94, 34),
    (10, 20, "披薩兩片配可樂", 850, 30, 98, 36),
    (11, 8, "火腿三明治配拿鐵", 600, 20, 62, 30),
    (11, 12, "排骨飯", 900, 34, 98, 38),
    (11, 19, "水餃十五顆", 820, 34, 92, 34),
    (12, 9, "蘿蔔糕加蛋配米漿", 560, 14, 78, 22),
    (12, 13, "咖哩豬排飯", 950, 36, 104, 42),
    (12, 20, "滷味拼盤配啤酒", 800, 30, 68, 40),
    (13, 8, "起司蛋餅加紅茶", 630, 18, 70, 30),
    (13, 12, "炸醬麵加小菜", 870, 32, 102, 34),
    (13, 19, "燒肉吃到飽", 880, 52, 60, 48),
]


def _to_db_datetime(days_ago: int, hour: int) -> datetime:
    """把「幾天前的台北時間幾點」轉成資料庫存的 naive UTC datetime。

    food_logs.logged_at 是 timestamp without time zone，存的是 UTC 但沒有 tzinfo，
    所以這裡算完要 .replace(tzinfo=None) 把時區資訊拿掉才對得上。
    順序不能顛倒：先 astimezone 換算時刻，再拔掉標籤。
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
