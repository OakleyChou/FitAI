# Python 版本跟 pyproject 的 ^3.11 對齊。
# slim 而不是完整版：映像小很多，而且這個專案不需要編譯工具鏈
# （psycopg 用 binary 版、pgvector 是純 Python）。
FROM python:3.11-slim

WORKDIR /app

# 讓輸出直接進 log 不被緩衝，容器環境才看得到即時訊息
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir poetry==1.8.3

# 先只複製依賴定義再安裝——這樣只要 pyproject/poetry.lock 沒變，
# Docker 就會重用這層的快取。程式碼改動不會觸發重新安裝依賴，
# 重建時間從幾分鐘降到幾秒。
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-interaction --no-ansi

COPY . .

EXPOSE 8000

# --host 0.0.0.0 不能省：預設的 127.0.0.1 只監聽容器內部，
# 從外面連進來會被拒絕。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
