"""`/chat` 端點的整合測試：測「接縫」而不是單一函式。

## 為什麼要有這一層

這個專案踩過的 bug 全部發生在接縫上：

    user_id 無條件注入       迴圈 ↔ 工具        兩邊各自都正確
    degraded 永遠是 False    工具 ↔ 迴圈 ↔ API  每層單獨看都對
    DB 沒有 timeout          應用 ↔ 資料庫      要真的斷線才現形

三個都不是「某個函式寫錯」，而是「兩個都對的東西接不起來」——
單元測試涵蓋不到，因為它們各自都通過。

## 為什麼用假的 agent

整合測試要測的是**契約**（狀態碼、回應形狀、降級判斷），不是 LLM 的答案品質。
打真的 LLM 會讓測試變慢、花錢，而且**不穩定**——同樣的輸入會有不同輸出，
紅燈的原因往往是模型今天心情不同而不是程式壞了。

答案品質歸 eval 管（`evals/run_eval.py`），那是刻意跑得少、有人工標註的另一套機制。
"""

import pytest

from app.agent.loop import AgentResult, ToolCall
from app.routers import chat as chat_module

# client fixture 來自 tests/conftest.py


@pytest.fixture
def fake_agent(monkeypatch):
    """把 run_agent 換掉，讓測試可以指定 agent 的回傳值。

    回傳一個 setter：測試呼叫它來決定這次 /chat 會拿到什麼 AgentResult。

    patch 的是 chat 模組裡的名字而不是 loop 模組裡的——
    `from app.agent.loop import run_agent` 會在 chat 模組建立自己的名字，
    改 loop.run_agent 對它沒有影響。**patch 使用它的地方，不是定義它的地方。**
    """

    def _set(result: AgentResult | Exception):
        def fake(question: str, user_id: int) -> AgentResult:
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(chat_module, "run_agent", fake)

    return _set


def _tool_call(name: str, unavailable: bool = False) -> ToolCall:
    return ToolCall(name=name, arguments={}, result="...", unavailable=unavailable)


# ── 正常路徑 ──────────────────────────────────────────────────


def test_正常回答回傳工具路徑(client, fake_agent):
    fake_agent(
        AgentResult(
            answer="你今天吃了 1620 大卡。",
            tool_calls=[_tool_call("query_food_logs")],
            iterations=2,
        )
    )

    r = client.post("/chat", json={"question": "我今天吃了什麼？"})

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "你今天吃了 1620 大卡。"
    assert body["degraded"] is False
    assert [c["name"] for c in body["tool_calls"]] == ["query_food_logs"]


def test_回應不含工具的完整結果(client, fake_agent):
    """工具結果可能上千字，而且是內部細節，不該外流給前端。"""
    fake_agent(
        AgentResult(
            answer="好的。",
            tool_calls=[
                ToolCall(
                    name="query_food_logs",
                    arguments={"days_back": 0},
                    result="這是不該出現在回應裡的內部資料",
                )
            ],
        )
    )

    r = client.post("/chat", json={"question": "我今天吃了什麼？"})

    assert "不該出現在回應裡" not in r.text
    # 但工具名稱與參數要有——前端要顯示「查詢中」或除錯時用得到
    assert r.json()["tool_calls"][0]["arguments"] == {"days_back": 0}


# ── 降級 ──────────────────────────────────────────────────────


def test_工具不可用時標記降級但仍回200(client, fake_agent):
    """agent 降級成功時那個答案對使用者有價值。

    回 503 的話前端會當成失敗，反而把有用的回答丟掉——
    所以回 200 並用 degraded 告訴前端「部分功能不可用」。
    """
    fake_agent(
        AgentResult(
            answer="目前無法查詢你的飲食紀錄，但可以告訴我食物名稱。",
            tool_calls=[_tool_call("query_food_logs", unavailable=True)],
        )
    )

    r = client.post("/chat", json={"question": "我今天吃了什麼？"})

    assert r.status_code == 200
    assert r.json()["degraded"] is True


def test_迴圈層失敗也算降級(client, fake_agent):
    """降級的來源有兩個：個別工具失敗，以及迴圈本身失敗。

    只看其中一個的話會漏——這正是先前 degraded 永遠是 False 的原因。
    """
    fake_agent(AgentResult(answer="我盡量回答。", error="TimeoutError: loop budget exceeded"))

    r = client.post("/chat", json={"question": "隨便問"})

    assert r.status_code == 200
    assert r.json()["degraded"] is True


def test_完全沒答案時回503(client, fake_agent):
    """有錯誤且答案是空的，表示真的失敗了，沒有東西可以給使用者。"""
    fake_agent(AgentResult(answer="   ", error="RuntimeError: everything is broken"))

    r = client.post("/chat", json={"question": "隨便問"})

    assert r.status_code == 503
    assert r.json()["error"]["code"] == "service_unavailable"


# ── 錯誤處理 ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="缺欄位"),
        pytest.param({"question": ""}, id="空字串"),
        pytest.param({"question": "豆" * 600}, id="超過長度上限"),
        pytest.param({"question": 123}, id="型別錯誤"),
    ],
)
def test_不合法的請求回422且格式一致(client, payload):
    r = client.post("/chat", json=payload)

    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_未預期的例外不洩漏內部細節(client, fake_agent):
    """全域 handler 是最後一道防線。

    沒有它，未捕捉的例外會把 stack trace 吐給使用者——
    那可能含檔案路徑、連線字串、SQL 語句。
    """
    fake_agent(RuntimeError("connection to server at localhost:5432 failed"))

    r = client.post("/chat", json={"question": "隨便問"})

    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "localhost:5432" not in r.text
    assert "Traceback" not in r.text


def test_所有錯誤回應的形狀一致(client, fake_agent):
    """前端只要寫一條錯誤處理路徑。

    形狀一致比欄位設計得漂亮重要——不一致的話前端要處理 N 種形狀。
    """
    fake_agent(AgentResult(answer="", error="boom"))
    responses = [
        client.post("/chat", json={}),  # 422
        client.post("/chat", json={"question": "x"}),  # 503
        client.get("/no-such-route"),  # 404（Starlette 自己拋的）
    ]

    for r in responses:
        body = r.json()
        assert "error" in body, f"{r.status_code} 的回應沒有 error 欄位"
        assert set(body["error"]) == {"code", "message"}


# ── user_id 邊界 ──────────────────────────────────────────────


def test_客戶端無法指定user_id(client, monkeypatch):
    """模型能指定 user_id 的話會洩漏別人的資料，API 層也不能破功。

    ChatRequest 沒有這個欄位，多傳會被 Pydantic 忽略；
    真正生效的是 deps.get_current_user_id。
    """
    seen: dict[str, int] = {}

    def fake(question: str, user_id: int) -> AgentResult:
        seen["user_id"] = user_id
        return AgentResult(answer="ok")

    monkeypatch.setattr(chat_module, "run_agent", fake)

    r = client.post("/chat", json={"question": "我今天吃了什麼？", "user_id": 999})

    assert r.status_code == 200
    assert seen["user_id"] == 1, "客戶端傳的 user_id 必須被忽略"
