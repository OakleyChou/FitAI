"""Agent 錯誤處理的邊界測試。

測的是 `_execute` 的**錯誤分類**——決定一個錯誤該怎麼辦，看的是
「模型有沒有能力修它」：

- 模型能修（參數錯、值域錯）→ 訊息原樣回給它重試
- 模型修不了（DB 掛了）→ 只回籠統說明、標記 unavailable、不洩漏細節

這一層測得起來是因為 `_execute` 不碰 LLM，只是「查表 → 呼叫 → 分類錯誤」。
迴圈層級的行為（超時、放棄工具）要 mock LLM 才測得到，
排在 9/14 的 agent eval 一起做。
"""

import pytest

from app.agent import loop as loop_module
from app.agent.loop import _execute


@pytest.fixture
def fake_registry(monkeypatch):
    """把工具表換成可控的假工具，測試不依賴資料庫也不花 API 費用。"""

    def ok_tool(query: str) -> str:
        return f"查到了：{query}"

    def needs_user(user_id: int) -> str:
        return f"user={user_id}"

    def bad_value(nutrient: str) -> str:
        raise ValueError(f"不支援的營養素 {nutrient!r}，可用值：['protein_g']")

    def db_down(query: str) -> str:
        raise ConnectionError("connection to server at localhost:5432 failed")

    registry = {
        "ok_tool": ok_tool,
        "needs_user": needs_user,
        "bad_value": bad_value,
        "db_down": db_down,
    }
    monkeypatch.setattr(loop_module, "TOOL_REGISTRY", registry)
    return registry


def test_未知工具名會列出可用清單(fake_registry):
    """模型編了不存在的工具時，錯誤訊息要讓它能自我修正。"""
    outcome = _execute("search_everything", {}, user_id=1)

    assert "沒有名為" in outcome.content
    assert "ok_tool" in outcome.content, "要列出可用工具，模型才知道該改成什麼"
    assert outcome.error is not None
    assert not outcome.unavailable, "工具名打錯是模型能修的，不該標記為不可用"


def test_參數錯誤的訊息原樣回給模型(fake_registry):
    """TypeError 屬於模型能修的類別，訊息要具體。"""
    outcome = _execute("ok_tool", {"wrong_param": "x"}, user_id=1)

    assert "錯誤" in outcome.content
    assert "重試" in outcome.content
    assert not outcome.unavailable


def test_值域錯誤保留工具自己的提示(fake_registry):
    """ValueError 的訊息是工具作者寫給模型看的，不要蓋掉。"""
    outcome = _execute("bad_value", {"nutrient": "protein"}, user_id=1)

    assert "protein_g" in outcome.content, "白名單要出現在訊息裡，模型才改得對"
    assert not outcome.unavailable


def test_系統錯誤不洩漏細節且標記不可用(fake_registry):
    """DB 連不上這種錯誤，模型改參數也修不好。

    兩個要求：訊息不能含連線字串等內部資訊；要標記 unavailable
    讓迴圈知道別再試這個工具。
    """
    outcome = _execute("db_down", {"query": "豆漿"}, user_id=1)

    assert outcome.unavailable, "系統層失敗要標記，否則模型會一直重試"
    assert "localhost:5432" not in outcome.content, "內部細節不能給模型看到"
    assert "暫時不可用" in outcome.content
    # 完整訊息仍要留在 error 欄供除錯——問題不能無聲消失
    assert "localhost:5432" in outcome.error


def test_user_id_只注入給需要的工具(fake_registry):
    """9/11 踩過的坑：無條件注入會讓不收這個參數的工具每次都 TypeError。"""
    assert _execute("needs_user", {}, user_id=42).content == "user=42"
    # ok_tool 不收 user_id，注入的話會炸；能正常回傳就表示沒亂塞
    assert _execute("ok_tool", {"query": "豆漿"}, user_id=42).error is None


def test_模型不能自己指定_user_id(fake_registry):
    """模型能指定 user_id 的話，一句「查 user 2 的紀錄」就會洩漏別人的資料。"""
    outcome = _execute("needs_user", {"user_id": 999}, user_id=42)

    assert outcome.content == "user=42", "模型傳的 user_id 必須被忽略"
