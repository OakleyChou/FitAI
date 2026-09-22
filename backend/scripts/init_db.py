"""建立資料庫結構：啟用 pgvector、建表、建開發用使用者。

跑法（在 backend/ 底下）：
    poetry run python -m scripts.init_db

在這之前，建表這件事只存在於一次性的互動指令裡——沒有任何程式碼做得到，
所以換一台電腦（或別人 clone 下來）就卡在第一步。

**這支腳本是冪等的**，重複執行不會出錯也不會重複建立，
所以可以放心放進啟動流程。

## 為什麼不是 Alembic

專案有 alembic 的依賴但沒有 migrations。現階段 schema 還在變動，
用 create_all 直接對齊 model 定義比較快。等 schema 穩定、或有正式資料
需要保留時，再導入 migration——那時 create_all 就不夠用了，
因為它只會「建立不存在的表」，不會修改既有的表結構。
"""

from sqlalchemy import text

import app.models  # noqa: F401  import 就會讓 model 註冊到 Base.metadata
from app.database import Base, SessionLocal, engine
from app.llm import FOOD_EXTRACTION_PROMPT_NAME
from app.models import PromptVersion, User
from app.prompts import FOOD_EXTRACTION_SYSTEM_PROMPT

DEV_USER_ID = 1
DEV_USER_EMAIL = "test@example.com"


def init() -> None:
    with engine.begin() as conn:
        # pgvector 要先啟用，Vector 欄位才建得起來。
        # IF NOT EXISTS 讓這行可以重複執行。
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # create_all 只會建立「不存在」的表，已存在的完全不動（包含欄位有變動時）。
    Base.metadata.create_all(engine)

    # 建開發用使用者。seed_dev_data 需要這個人存在才能掛飲食紀錄，
    # 沒有他的話新的 clone 會卡在「找不到 user_id=1」。
    db = SessionLocal()
    try:
        if db.query(User).filter(User.id == DEV_USER_ID).first() is None:
            db.add(
                User(
                    id=DEV_USER_ID,
                    email=DEV_USER_EMAIL,
                    height_cm=170.0,
                    current_weight_kg=60.0,
                    target_weight_kg=57.0,
                    daily_calories_goal=2000.0,
                )
            )
            db.commit()

        # 寫入基準 prompt。少了這筆，POST /food_logs 會直接 500——
        # 它要靠 get_active_prompt 拿系統提示詞，而那是從資料庫讀的。
        # 這個 bug 在 2026-09-22 的整合測試中踩到：prompt 原本只存在
        # 開發機的資料庫裡，沒有任何程式碼或文件記錄它的內容。
        #
        # 只在「這個 name 完全沒有任何版本」時才寫，所以不會蓋掉
        # 之後用 add_prompt_version 加上去的新版本。
        has_prompt = (
            db.query(PromptVersion)
            .filter(PromptVersion.name == FOOD_EXTRACTION_PROMPT_NAME)
            .first()
            is not None
        )
        if not has_prompt:
            db.add(
                PromptVersion(
                    name=FOOD_EXTRACTION_PROMPT_NAME,
                    version=1,
                    content=FOOD_EXTRACTION_SYSTEM_PROMPT,
                    is_active=True,
                )
            )
            db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    init()
    tables = ", ".join(sorted(Base.metadata.tables))
    print(f"資料庫初始化完成。資料表：{tables}")
    print(f"開發用使用者：id={DEV_USER_ID} ({DEV_USER_EMAIL})")
    print(f"基準 prompt：{FOOD_EXTRACTION_PROMPT_NAME}")
