"""pytest 的共用設定。

`conftest.py` 是 pytest 的慣例檔名：放在這裡的 fixture 不用 import，
同目錄（含子目錄）的測試檔直接用參數名就拿得到。

## 為什麼測試要有自己的資料庫

整合測試要能在**沒有 Docker、沒有 Postgres**的情況下跑完，否則它會變成
「開發者有空才跑」的那種測試，也就等於沒有測試。這裡用 SQLite in-memory
取代真正的 Postgres：建表、寫入、查詢、刪除的行為足以驗證端點的契約。

代價要講清楚：**這層測不到 Postgres 特有的東西**。JSONB 查詢、pgvector 的
相似度搜尋、statement_timeout——這些在 SQLite 上根本不存在，所以
knowledge_chunks 那張表這裡不建，相關功能靠 eval 在真的資料庫上驗證。
知道一層測試「測不到什麼」，跟知道它測到什麼一樣重要。
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import FoodLog, User


@pytest.fixture
def db_session():
    """每個測試一個乾淨的資料庫。

    StaticPool + check_same_thread=False 是 SQLite in-memory 的必要組合：
    預設每開一條連線就是一個全新的空資料庫，而 TestClient 會在另一個
    執行緒跑端點——不固定成同一條連線的話，測試寫進去的資料端點看不到。
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # 只建用得到的表。knowledge_chunks 有 JSONB 與 Vector 欄位，
    # SQLite 建不起來，而這層測試也不需要它。
    Base.metadata.create_all(engine, tables=[User.__table__, FoodLog.__table__])

    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db_session):
    """接上測試資料庫的 TestClient。

    dependency_overrides 是 FastAPI 為測試準備的替換點：
    端點寫的還是 Depends(get_db)，但拿到的是這裡給的 session。
    正因為連線是從 DI 拿的而不是在端點裡自己建，才換得掉——
    這是依賴注入在測試上的實際回報。

    raise_server_exceptions=False 讓未捕捉的例外走 app 的 exception handler，
    而不是直接往測試拋——否則測不到「全域 handler 有沒有把細節擋下來」。
    """
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        # 覆寫是全域狀態，不清掉會汙染後面的測試。
        app.dependency_overrides.clear()


@pytest.fixture
def user(db_session) -> User:
    """一個存在的使用者。food_logs 的 user_id 指向 users.id。"""
    u = User(
        id=1,
        email="dev@example.com",
        height_cm=170.0,
        current_weight_kg=70.0,
        target_weight_kg=65.0,
        daily_calories_goal=1800.0,
    )
    db_session.add(u)
    db_session.commit()
    return u
