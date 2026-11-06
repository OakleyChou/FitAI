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


def _enum(tool: str, param: str) -> set[str]:
    return set(SCHEMA_BY_NAME[tool]["function"]["parameters"]["properties"][param]["enum"])


@pytest.mark.parametrize("personal_tool", ["rank_my_meals", "rank_my_foods"])
def test_個人紀錄的營養素白名單比公開資料庫窄(personal_tool):
    """rank_by_nutrient 查公開資料庫（有纖維糖鈉），
    另外兩個查使用者紀錄（只有熱量與三大營養素）。

    混用的話會出現「問我吃過的纖維最高」→ 查不到欄位 → 模型改去查公開資料庫，
    然後把資料庫的食物說成使用者吃過的。那正是個人排序工具要防的事。
    """
    public = _enum("rank_by_nutrient", "nutrient")
    personal = _enum(personal_tool, "nutrient")

    assert "fiber_g" in public
    assert "fiber_g" not in personal
    assert "calories" in personal


def test_只有食物層級能算每一百公克():
    """`food_logs` 沒有餐點重量，`food_log_items` 有 portion_g。

    這個差別是 food_log_items 存在的主要理由。如果哪天 rank_my_meals 也
    宣告了 per_100g，那表示有人以為餐有重量——那會算出一個沒有來源的數字。
    """
    assert "per_100g" in _enum("rank_my_foods", "metric")
    assert "per_100g" not in _enum("rank_my_meals", "metric")


@pytest.mark.parametrize(
    "tool, must_mention",
    [
        # 三個工具的 description 必須互相指路。少了指路，模型選錯工具之後
        # 只會回「查不到」，而其實換一個工具就答得出來（pitfalls #45）。
        ("rank_my_foods", ["rank_my_meals", "rank_by_nutrient"]),
        ("rank_my_meals", ["rank_my_foods"]),
    ],
)
def test_排序工具的描述要互相指路(tool, must_mention):
    desc = SCHEMA_BY_NAME[tool]["function"]["description"]

    for other in must_mention:
        assert other in desc, f"{tool} 的描述沒有提到 {other}"


def test_描述不可再聲稱逐項明細沒有保留():
    """2026-11-06 明細落地之後，這句話變成假的。

    description 是送給模型的 prompt，模型會把它當事實轉述給使用者。
    限制的敘述跟程式一樣會過期，差別是它過期時不會報錯。
    """
    for tool in ("rank_my_meals", "rank_my_foods"):
        assert "明細沒有保留" not in SCHEMA_BY_NAME[tool]["function"]["description"]
