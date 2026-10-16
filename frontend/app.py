"""FitAI Coach 前端入口。

跑法（在 frontend/ 底下）：
    pip install -r requirements.txt
    streamlit run app.py

或用容器（在專案根目錄）：
    docker compose up -d

## 這支程式刻意很薄

它只做三件事：設定頁面、顯示後端連線狀態、把路由交給 views/。
業務邏輯一行都沒有——那全部在後端，由 58 個測試與 8 題 eval 守著。

`from app.agent.loop import run_agent` 這種寫法在這裡會直接 ImportError：
前端容器裡沒有 backend 的程式碼（見 requirements.txt）。
**「前端不碰業務邏輯」這條規則由環境強制，不靠自律。**
"""

import streamlit as st

import api
from views import chat, log, today

st.set_page_config(
    page_title="FitAI Coach",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _sidebar_status() -> None:
    """側邊欄顯示後端狀態。

    這不只是裝飾：這個專案最常見的「壞掉」就是後端沒起來或資料庫沒灌，
    而那在問答頁只會表現成一句看不出原因的錯誤訊息。
    把連線狀態放在固定位置，第一步除錯就不用猜。
    """
    with st.sidebar:
        st.title("🥗 FitAI Coach")
        st.caption("飲食紀錄 + RAG + 工具呼叫 agent")

        try:
            info = api.health()
            st.success(f"後端正常 · v{info.get('version', '?')}", icon="✅")
        except api.ApiError as exc:
            st.error(f"後端異常：{exc.message}", icon="🚨")
            st.caption(f"API：`{api.API_BASE_URL}`")
            with st.expander("怎麼修"):
                st.markdown(
                    "1. 啟動 Docker Desktop\n"
                    "2. 專案根目錄執行 `docker compose up -d`\n"
                    "3. 首次還需要灌資料："
                    "`docker compose exec api python -m scripts.init_db`、"
                    "`... seed_dev_data`、`... -m app.rag.ingest`"
                )

        st.divider()
        st.caption(
            "**已知限制**：沒有認證，所有人共用 user_id=1；"
            "工具全部唯讀，agent 不會代為寫入紀錄。"
        )


PAGES = {
    "飲食問答": chat.render,
    "記錄飲食": log.render,
    "今日總覽": today.render,
}


def main() -> None:
    _sidebar_status()

    # 用 radio 而不是 st.navigation：radio 的選擇自動存在 session_state 裡，
    # 在任何 Streamlit 版本都一樣，而且側邊欄的版面比較好控制。
    with st.sidebar:
        choice = st.radio("頁面", list(PAGES), label_visibility="collapsed")

    PAGES[choice]()


main()
