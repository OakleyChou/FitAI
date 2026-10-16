# 前端刻意用獨立的映像，不共用 backend 的。
#
# 這不只是乾淨問題：這個映像裡**沒有** backend 的程式碼，所以前端
# 不可能 import 到業務邏輯——那條架構規則由環境強制執行，不靠自律。
FROM python:3.11-slim

WORKDIR /app

# 依賴先複製再安裝，程式碼後複製：改前端程式碼時不會讓這層失效，
# rebuild 從幾分鐘變幾秒（跟 backend/Dockerfile 同樣的理由）。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# --server.address=0.0.0.0 是容器裡的必要設定。Streamlit 預設只綁 localhost，
# 那在容器內表示「只有容器自己連得到」——從主機打 8501 會直接被拒絕，
# 而且錯誤訊息完全看不出是綁定位址的問題。
#
# --server.headless=true 讓它不要嘗試開瀏覽器（容器裡沒有瀏覽器）。
# gatherUsageStats=false 關掉 Streamlit 的匿名使用統計回傳。
# 作品集專案不需要它，而且少一個對外連線就少一個卡住啟動的可能。
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
