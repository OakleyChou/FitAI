"""統一的錯誤回應格式與全域例外處理。

## 為什麼不在每個端點寫 try/except

三個問題：重複、會漏（新加的端點忘記寫就裸奔）、格式不一致
（前端要處理 N 種錯誤形狀）。集中處理之後，端點只負責業務邏輯，
「例外怎麼變成 HTTP 回應」是這裡的事。

## 回應形狀固定

    {"error": {"code": "agent_unavailable", "message": "…"}}

code 給程式判斷（前端可針對特定錯誤做不同處理），message 給人看。
**形狀一致比欄位設計得漂亮重要**——前端只要寫一條錯誤處理路徑。

## 兩種觀眾

給使用者的訊息不能洩漏內部細節（檔案路徑、連線字串、SQL）。
完整錯誤用 logging 留下來，否則問題會無聲消失——
使用者只會覺得「這功能好像怪怪的」然後不用了。
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    """本專案的領域例外基底。

    業務邏輯層拋這個，轉成 HTTP 的事情集中在 handler 做——
    這樣 routers 裡不會散落 status code，換 API 形式時也只要改一處。
    """

    code = "internal_error"
    status_code = 500
    message = "系統發生未預期的問題，請稍後再試。"

    def __init__(self, message: str | None = None, *, detail: str | None = None):
        # message 給使用者看，detail 留給 log——兩種觀眾分開。
        self.message = message or self.message
        self.detail = detail
        super().__init__(self.message)


class ServiceUnavailable(AppError):
    """依賴的服務暫時不可用（資料庫、外部 API）。

    用 503 而不是 500：500 是「我壞了」，503 是「我依賴的東西壞了，
    等一下可能會好」。這個差別會影響呼叫端要不要重試。
    """

    code = "service_unavailable"
    status_code = 503
    message = "服務暫時無法使用，請稍後再試。"


def _error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def register_error_handlers(app: FastAPI) -> None:
    """把所有例外處理註冊到 app 上。在 main.py 呼叫一次即可。"""

    @app.exception_handler(AppError)
    def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "AppError %s at %s: %s", exc.code, request.url.path, exc.detail or exc.message
        )
        return JSONResponse(
            status_code=exc.status_code, content=_error_body(exc.code, exc.message)
        )

    @app.exception_handler(StarletteHTTPException)
    def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # 既有的 raise HTTPException(404, "Food log not exist") 也走這裡，
        # 讓它們的回應形狀跟其他錯誤一致，前端不用分兩種處理。
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(f"http_{exc.status_code}", str(exc.detail)),
        )

    @app.exception_handler(RequestValidationError)
    def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Pydantic 擋下來的請求。422 是 FastAPI 的慣例，
        # 這裡只是把預設格式改成跟其他錯誤一致。
        return JSONResponse(
            status_code=422,
            content=_error_body("validation_error", "請求格式不正確，請檢查輸入。"),
        )

    @app.exception_handler(Exception)
    def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # 最後一道防線。沒有這層，未接到的例外會把 stack trace 吐給使用者，
        # 那可能含檔案路徑、連線字串、SQL 語句。
        #
        # exc_info=True 讓完整 traceback 進 log——給使用者看籠統訊息，
        # 不代表可以把細節丟掉。
        logger.exception("Unhandled error at %s", request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_error_body("internal_error", "系統發生未預期的問題，請稍後再試。"),
        )
