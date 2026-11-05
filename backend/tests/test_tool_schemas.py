"""工具的 JSON Schema 與實際函式之間的不變式。

這些檢查守的是**手寫 schema 的代價**。`app/agent/schemas.py` 刻意手寫而不是
從型別註記自動產生（四五個工具的規模下手寫比較好讀，而且每個 description
都是寫給模型看的指令，值得逐字斟酌）。代價是兩邊會漂移：

    改了函式簽章但忘了改 schema  → 模型永遠不知道有這個新參數
    schema 寫了不存在的參數      → 模型傳了它，函式拋 TypeError
    新增工具但忘了註冊           → 模型看不到，或看到了卻派送不到
    user_id 洩漏進 schema        → 一句「查 user 2 的紀錄」就能看別人的資料

四個都不會在單元測試裡現形，因為兩邊各自都是合法的 Python。
"""

import inspect

import pytest

from app.agent.schemas import SERVER_PROVIDED_ARGS, TOOL_REGISTRY, TOOL_SCHEMAS

SCHEMA_BY_NAME = {t["function"]["name"]: t for t in TOOL_SCHEMAS}


def _declared_params(name: str) -> set[str]:
    return set(SCHEMA_BY_NAME[name]["function"]["parameters"]["properties"])


def test_registry_與schema的工具集合一致():
    """少一邊的話，模型要嘛看不到工具、要嘛看到了卻派送不到。"""
    assert set(TOOL_REGISTRY) == set(SCHEMA_BY_NAME)


@pytest.mark.parametrize("name", sorted(TOOL_REGISTRY))
def test_schema宣告的參數都存在於函式簽章(name):
    """schema 多寫一個參數，模型就會傳它，然後函式拋 TypeError。

    那個錯誤會被歸類成「模型能修」並原樣回給模型重試，
    但它改不了——參數在 schema 裡就是合法的，它會一直重試到用完次數。
    """
    real = set(inspect.signature(TOOL_REGISTRY[name]).parameters)
    assert _declared_params(name) <= real, (
        f"{name} 的 schema 宣告了函式沒有的參數：{_declared_params(name) - real}"
    )


@pytest.mark.parametrize("name", sorted(TOOL_REGISTRY))
def test_函式的參數都有宣告在schema裡(name):
    """反方向：函式有但 schema 沒寫的參數，模型永遠不知道可以用。

    實際發生過：`rank_by_nutrient` 本來只能由高到低，加了 `order` 之後
    如果忘記寫進 schema，模型問「低醣的東西」還是會拿到碳水最高的前三名。

    server_provided 的參數不算（user_id 由伺服器端注入）。
    """
    real = set(inspect.signature(TOOL_REGISTRY[name]).parameters) - SERVER_PROVIDED_ARGS
    missing = real - _declared_params(name)
    assert not missing, f"{name} 的這些參數沒有宣告在 schema 裡：{missing}"


@pytest.mark.parametrize("name", sorted(TOOL_REGISTRY))
def test_server提供的參數不可出現在schema裡(name):
    """user_id 絕對不能讓模型指定。

    一句「查 user 2 的紀錄」就會洩漏別人的資料。這是安全邊界，
    不是整潔問題——所以用測試釘住，不靠 code review 記得。
    """
    leaked = _declared_params(name) & SERVER_PROVIDED_ARGS
    assert not leaked, f"{name} 的 schema 洩漏了伺服器端參數：{leaked}"


@pytest.mark.parametrize("name", sorted(TOOL_REGISTRY))
def test_每個工具都有描述(name):
    """description 是模型選工具的唯一依據。空的等於這個工具不存在。"""
    desc = SCHEMA_BY_NAME[name]["function"].get("description", "")
    assert len(desc) > 30, f"{name} 的 description 太短或缺少"


@pytest.mark.parametrize("name", sorted(TOOL_REGISTRY))
def test_必填參數都有宣告在properties裡(name):
    """required 列了 properties 沒有的欄位時，JSON Schema 本身就不合法。"""
    fn = SCHEMA_BY_NAME[name]["function"]["parameters"]
    assert set(fn.get("required", [])) <= set(fn["properties"])


def test_兩個排序工具的營養素白名單刻意不同():
    """rank_by_nutrient 查公開資料庫（有纖維糖鈉），
    rank_my_meals 查使用者紀錄（只有熱量與三大營養素）。

    混用的話會出現「問我吃過的纖維最高」→ 查不到欄位 → 模型改去查公開資料庫，
    然後把資料庫的食物說成使用者吃過的。那正是 rank_my_meals 要防的事。
    """
    public = set(SCHEMA_BY_NAME["rank_by_nutrient"]["function"]["parameters"]["properties"]["nutrient"]["enum"])
    personal = set(SCHEMA_BY_NAME["rank_my_meals"]["function"]["parameters"]["properties"]["nutrient"]["enum"])

    assert "fiber_g" in public
    assert "fiber_g" not in personal
    assert "calories" in personal
