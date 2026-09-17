from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# 建立連線引擎。
#
# connect_timeout 不能省：psycopg 預設會一直等到作業系統放棄，而且 localhost
# 會依序試 ::1 和 127.0.0.1 兩個位址、各等一次。資料庫掛掉時，單一次查詢
# 就可能卡住一分鐘以上——在 agent 迴圈裡那會直接吃光整個時間預算，
# 使用者看到的是「查詢逾時」而不是「某個功能暫時不可用」。
#
# 通則：每一個對外呼叫都要有 timeout。沒設的地方遲早會變成卡住整個請求的那一個。
#
# pool_pre_ping 讓連線在借出前先 ping 一次，回收掉已經斷掉的舊連線
# （例如資料庫重啟過）。代價是每次查詢多一次極輕量的往返。
# statement_timeout 管的是另一件事：連得上、但查詢跑不完。
# connect_timeout 對「沒有索引的查詢掃全表」完全沒有幫助——那是兩種
# 不同的失敗模式，各需要各的上限。單位是毫秒，由 Postgres 端強制執行，
# 超過就中斷該查詢並回錯誤，不會拖著整個連線。
engine = create_engine(
    settings.database_url,
    connect_args={
        "connect_timeout": 5,
        "options": "-c statement_timeout=10000",
    },
    pool_pre_ping=True,
)

# session產生器
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()  # 用完自動關閉
