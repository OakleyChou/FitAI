import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.errors import register_error_handlers
from app.routers import chat, food_logs, health

# 全域錯誤處理會把完整 traceback 寫進 log，但沒有設定 logging 的話
# 那些訊息不會出現在任何地方——降級不等於隱藏。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許任何來源 # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 例外處理要在 router 之前註冊，才能涵蓋所有端點
register_error_handlers(app)

app.include_router(health.router)
app.include_router(food_logs.router)
app.include_router(chat.router)
