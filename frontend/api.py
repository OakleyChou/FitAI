"""後端 API 的唯一出入口。

前端所有的 HTTP 呼叫都集中在這裡，頁面只呼叫這裡的函式。這樣有三個好處：
base URL 和 timeout 只設定一處、錯誤處理只寫一次、換後端介面只改一個檔案。

## Timeout 必須由外而內遞增

後端的時間預算（見 `backend/app/agent/loop.py` 與 `database.py`）：

    LOOP_BUDGET_SECONDS = 90    agent 迴圈上限
    LLM_TIMEOUT_SECONDS = 30    單次 LLM 呼叫
    statement_timeout   = 10    單次資料庫查詢

前端的 timeout 若小於 90，會出現這種情形：

    t=30   前端放棄 → 使用者看到「連線逾時」
    t=45   後端跑完 → 完美的答案回給一個沒有人在聽的連線

**使用者看到失敗，但系統其實成功了**，而且那次 LLM 的錢照付。
所以 CHAT_TIMEOUT 必須大於後端的迴圈預算。

## 錯誤處理只寫一條路徑

後端所有錯誤都是同一個形狀（見 `backend/app/errors.py`）：

    {"error": {"code": "service_unavailable", "message": "..."}}

形狀一致的回報就是在這裡兌現的——一個 `_raise_for_error` 處理所有端點的
所有錯誤，不需要為每個端點寫一套。
"""

import os

import requests
import streamlit as st

# 容器內用服務名（compose 的 api service），本機開發用 localhost。
# 同一份程式碼服務兩種跑法，靠環境變數切換，不用改程式。
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")

# 問答要等 agent 跑完迴圈，必須大於後端的 LOOP_BUDGET_SECONDS（90 秒）。
CHAT_TIMEOUT = 120.0

# 單純的資料庫讀寫。新增紀錄要呼叫一次 LLM 解析，所以比純查詢寬一些。
WRITE_TIMEOUT = 60.0
READ_TIMEOUT = 15.0

# 目前後端沒有認證，`get_current_user_id()` 固定回傳 1（見 backend/app/deps.py）。
# 前端跟著用同一個值。真正要上線時這裡要換成登入後拿到的 id。
DEV_USER_ID = 1


class ApiError(Exception):
    """後端回了一個錯誤回應。message 已經是可以直接顯示給使用者的文字。"""

    def __init__(self, message: str, code: str = "unknown", status: int = 0):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


class ApiUnreachable(ApiError):
    """連不上後端——容器沒開、網路斷了、或逾時。

    跟 ApiError 分開是因為對使用者的意義不同：ApiError 是「後端說不行」，
    這個是「後端根本沒回應」。前者照後端的訊息顯示，後者要給操作提示。
    """


@st.cache_resource
def _session() -> requests.Session:
    """共用一條 HTTP 連線。

    用 cache_resource 而不是 cache_data：Session 物件不可序列化，
    而且我們要的就是「重複使用同一個物件」而非「快取回傳值」。

    為什麼需要它：Streamlit 每次互動都會把整支腳本重跑一遍。沒有快取的話，
    每次互動都會新建一個 Session、重新進行 TCP 與 TLS 交握。
    """
    return requests.Session()


# 讀取類的呼叫刻意**不加 @st.cache_data**。
#
# 加了會快一點，但也會引入一個失效問題：新增一筆紀錄之後，
# 總覽頁的合計要等快取過期才會更新，使用者會以為沒存進去。
# 這三頁的讀取都很輕（一次資料庫查詢），省下來的時間不值得那個風險。
#
# 要加的話，每個寫入操作之後都必須呼叫 st.cache_data.clear()——
# 漏一處就會出現「畫面與資料庫不一致」，而那不會報錯。


def _raise_for_error(resp: requests.Response) -> None:
    """把後端的錯誤回應轉成例外。所有端點共用。"""
    if resp.ok:
        return

    # 後端的錯誤一律是 {"error": {"code", "message"}}。但真的壞掉時
    # （反向代理回了一頁 HTML）也可能不是 JSON，所以要防。
    try:
        payload = resp.json()["error"]
        message, code = payload["message"], payload["code"]
    except (ValueError, KeyError, TypeError):
        message, code = f"後端回應異常（HTTP {resp.status_code}）", "malformed_response"

    raise ApiError(message, code=code, status=resp.status_code)


def _request(method: str, path: str, *, timeout: float, **kwargs) -> requests.Response:
    try:
        resp = _session().request(method, f"{API_BASE_URL}{path}", timeout=timeout, **kwargs)
    except requests.Timeout:
        raise ApiUnreachable(f"後端超過 {timeout:.0f} 秒沒有回應。", code="timeout")
    except requests.RequestException:
        # 刻意不把原始例外訊息給使用者——那裡面有主機名與連接埠。
        # 完整原因由 Streamlit 自己的 log 留下。
        raise ApiUnreachable("連不上後端服務，請確認它是否正在執行。", code="unreachable")

    _raise_for_error(resp)
    return resp


# ── 問答 ──────────────────────────────────────────────────────


def ask(question: str) -> dict:
    """問 agent 一個問題。

    回傳 {"answer": str, "degraded": bool, "tool_calls": [{"name", "arguments"}]}

    注意**沒有** user_id 參數：那由後端從登入狀態決定。
    讓前端指定的話，一句「查 user 2 的紀錄」就會洩漏別人的資料。
    """
    return _request("POST", "/chat", json={"question": question}, timeout=CHAT_TIMEOUT).json()


# ── 飲食紀錄 ──────────────────────────────────────────────────


def create_log(raw_text: str) -> dict:
    """新增一筆飲食紀錄。營養素由後端呼叫 LLM 解析，不是前端算的。"""
    return _request(
        "POST",
        "/food_logs",
        json={"user_id": DEV_USER_ID, "raw_text": raw_text},
        timeout=WRITE_TIMEOUT,
    ).json()


def list_logs() -> list[dict]:
    return _request(
        "GET", "/food_logs", params={"user_id": DEV_USER_ID}, timeout=READ_TIMEOUT
    ).json()


def delete_log(log_id: int) -> None:
    _request("DELETE", f"/food_logs/{log_id}", timeout=READ_TIMEOUT)


def day_summary(days_back: int = 0) -> dict:
    """某一天的摘要：逐餐、合計、與目標的差距。

    合計與剩餘額度**由後端算**（見 backend/app/nutrition.py）。
    前端自己篩日期加總的話，同一個數字會有兩份實作，
    而只有後端那份有測試守著——兩份遲早會分岔。
    """
    return _request(
        "GET", "/food_logs/summary", params={"days_back": days_back}, timeout=READ_TIMEOUT
    ).json()


# ── 健康檢查 ──────────────────────────────────────────────────


def health() -> dict:
    return _request("GET", "/health", timeout=READ_TIMEOUT).json()
