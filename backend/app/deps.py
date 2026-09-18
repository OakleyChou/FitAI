"""FastAPI 的依賴（dependency）。

## 為什麼 user_id 要用 dependency 而不是請求參數

讓前端傳 user_id 的話，任何人都能查別人的資料——這正是 agent 層
SERVER_PROVIDED_ARGS 擋掉的事，不要在 API 層破功。

**DI 是「放置未來會變的東西」的地方。** 現在寫死回傳 1，之後接上認證時
只改這個函式，所有端點完全不用動。
"""


def get_current_user_id() -> int:
    """目前登入的使用者 ID。

    尚未實作認證，固定回傳開發用的 user 1。

    之後接上認證時，這裡會變成解析 Authorization header 的 JWT、
    驗證簽章、取出 sub 欄位——但簽章 `() -> int` 不變，
    所以端點程式碼一行都不用改。
    """
    return 1
