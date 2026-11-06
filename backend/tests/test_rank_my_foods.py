"""`rank_my_foods` 的排序邏輯。純函式，不碰資料庫。

拆開的理由同 `_rank_meals`：密度換算、去重、涵蓋率計算都是純邏輯。

這個檔案有一半的測試在驗**輸出的文字**，不是數字。那不是為了湊覆蓋率：
工具的回傳值會原封不動進到模型的 context，所以那段文字就是 prompt。
數字算對但涵蓋率沒講，使用者會得到一個「涵蓋全部紀錄」的錯誤印象——
而那種錯誤不會報錯，答案也通順（pitfalls 第一節）。
"""

from datetime import datetime, timezone

import pytest

from app.agent.tools import _food_value, _rank_foods, rank_my_foods
from app.models import FoodLog, FoodLogItem
from app.schemas import InputType


def _item(
    name: str,
    portion: float,
    calories: float,
    protein: float,
    grounded: bool = False,
    source_ref: str | None = None,
) -> FoodLogItem:
    return FoodLogItem(
        name=name,
        portion_g=portion,
        calories=calories,
        protein_g=protein,
        carbs_g=0.0,
        fat_g=0.0,
        grounded=grounded,
        source_ref=source_ref,
        db_name=name if grounded else None,
    )


def _log(*items: FoodLogItem, day: int = 5, text: str = "午餐") -> FoodLog:
    log = FoodLog(
        id=1,
        user_id=1,
        # 資料庫存的是 naive UTC（見 pitfalls #33）
        logged_at=datetime(2026, 11, day, 4, 0, tzinfo=timezone.utc).replace(tzinfo=None),
        calories=sum(i.calories for i in items),
        protein_g=sum(i.protein_g for i in items),
        carbs_g=0.0,
        fat_g=0.0,
        raw_input_text=text,
        input_type=InputType.text,
    )
    log.items = list(items)
    return log


def _bare_log(calories: float = 700.0, day: int = 5) -> FoodLog:
    """2026-11-06 之前的紀錄：有合計，沒有明細。

    涵蓋率的測試需要這種紀錄。真實資料庫裡現在就是新舊混雜的狀態，
    假資料如果每一筆都有明細，涵蓋率那條路徑就永遠測不到。
    """
    return FoodLog(
        id=2,
        user_id=1,
        logged_at=datetime(2026, 11, day, 4, 0, tzinfo=timezone.utc).replace(tzinfo=None),
        calories=calories,
        protein_g=40.0,
        carbs_g=0.0,
        fat_g=0.0,
        raw_input_text="舊紀錄",
        input_type=InputType.text,
    )


# ── 單項的數值 ────────────────────────────────────────────────


def test_per_100g是每一百公克的含量():
    """這是 food_log_items 存在的主要理由。

    `food_logs` 沒有餐點重量，所以餐的層級只能算每 100 大卡；
    明細有 portion_g，所以「蛋白質密度最高的食物」才答得出來。
    """
    item = _item("烤鯖魚", portion=100, calories=270, protein=24)

    assert _food_value(item, "protein_g", "per_100g") == pytest.approx(24.0)


def test_total是吃進去的絕對量():
    item = _item("烤鯖魚", portion=100, calories=270, protein=24)

    assert _food_value(item, "protein_g", "total") == 24


def test_per_100kcal是每一百大卡的含量():
    item = _item("烤鯖魚", portion=100, calories=270, protein=27)

    assert _food_value(item, "protein_g", "per_100kcal") == pytest.approx(10.0)


@pytest.mark.parametrize(
    "portion, calories, metric",
    [
        (0, 270, "per_100g"),  # 份量不明
        (100, 0, "per_100kcal"),  # 估不出熱量
    ],
)
def test_基準為零時回None(portion, calories, metric):
    """回 None 而不是 0。

    0 在 asc 排序會被當成「密度最低」排到最前面，變成一個看起來合理的
    錯誤答案。這跟 `_meal_value` 是同一個決定。
    """
    item = _item("不明", portion=portion, calories=calories, protein=10)

    assert _food_value(item, "protein_g", metric) is None


# ── 參數驗證 ──────────────────────────────────────────────────


def test_熱量配每百大卡要被拒絕():
    """每 100 大卡裡的熱量恆等於 100。

    不擋的話每一項的數值都是 100.0，排序退化成任意順序，
    而模型會照常說「密度最高的是 X」——一個不報錯、數字也「正確」的假答案。
    """
    with pytest.raises(ValueError, match="恆等於 100"):
        rank_my_foods(user_id=1, nutrient="calories", metric="per_100kcal")


def test_錯誤訊息要指出可以改用什麼():
    """`TypeError`/`ValueError` 的訊息會回給模型讓它自我修正，

    所以訊息裡必須有可用的替代值。只說「不支援」的話，模型會原封不動重試一次
    （pitfalls #49 實測過：同一個錯誤參數連試兩輪）。
    """
    with pytest.raises(ValueError) as e:
        rank_my_foods(user_id=1, nutrient="calories", metric="per_100kcal")

    assert "per_100g" in str(e.value)
    assert "total" in str(e.value)


@pytest.mark.parametrize(
    "kwargs, hint",
    [
        ({"nutrient": "fiber_g"}, "可用值"),
        ({"metric": "per_serving"}, "可用值"),
        ({"order": "random"}, "可用值"),
        ({"days_back": -1}, "不能是負數"),
    ],
)
def test_不合法的參數會被擋下來(kwargs, hint):
    with pytest.raises(ValueError, match=hint):
        rank_my_foods(user_id=1, **kwargs)


# ── 排序 ──────────────────────────────────────────────────────


def test_依每百公克的密度排序():
    logs = [
        _log(
            _item("烤鯖魚", 100, 270, 24, grounded=True, source_ref="S001"),
            _item("白飯", 100, 180, 3, grounded=True, source_ref="S002"),
        )
    ]

    out = _rank_foods(logs, "protein_g", "per_100g", "desc", 3, 6)

    assert out.index("烤鯖魚") < out.index("白飯")
    assert "24.0" in out


def test_同一樣食物出現多次只列最高的那次():
    """不去重的話，limit=3 會被同一樣食物的三次出現佔滿。

    「我吃過哪樣食物蛋白質最高」就退化成「白飯、白飯、白飯」——
    每個數字都對，而答案沒有回答問題。
    """
    logs = [
        _log(_item("白飯", 100, 180, 3), day=5),
        _log(_item("白飯", 200, 360, 6), day=4),
        _log(_item("白飯", 150, 270, 4), day=3),
    ]

    out = _rank_foods(logs, "protein_g", "total", "desc", 3, 6)

    assert out.count("白飯") == 1
    assert "吃過 3 次" in out
    # 去重讓 total 變成「單次最多」而不是「期間累計」，那要講出來
    assert "去重規則" in out
    assert "單次" in out


def test_只出現一次時不加去重說明():
    """沒有發生的事不要解釋。

    每次都附上「只列最高的那一次」會讓模型在不需要的時候也轉述它，
    而使用者讀到的是一個不存在的限制。
    """
    out = _rank_foods([_log(_item("烤鯖魚", 100, 270, 24))], "protein_g", "per_100g", "desc", 3, 6)

    assert "去重規則" not in out


def test_asc時去重要留最低的那次():
    """去重的方向要跟排序方向一致。

    desc 留最高、asc 留最低。反了的話「最低的食物」會列出那樣食物
    最高的一次，而排名位置卻是照最低值算的——兩個數字互相矛盾。
    """
    logs = [
        _log(_item("白飯", 100, 180, 3), day=5),
        _log(_item("白飯", 100, 180, 9), day=4),
    ]

    out = _rank_foods(logs, "protein_g", "per_100g", "asc", 3, 6)

    assert "3.0" in out
    assert "9.0" not in out


# ── 涵蓋率：這是這個工具最容易說錯話的地方 ────────────────────


def test_輸出一定附上涵蓋率():
    """明細 2026-11-06 才落地，更早的紀錄只有合計。

    不講涵蓋率的話，「我吃過蛋白質最高的食物」會被理解成涵蓋全部紀錄。
    這類遺漏不會報錯、答案也通順——只有把涵蓋率送進 context 才擋得住。
    """
    logs = [_log(_item("烤鯖魚", 100, 270, 24)), _bare_log(), _bare_log()]

    out = _rank_foods(logs, "protein_g", "per_100g", "desc", 3, 6)

    assert "涵蓋率" in out
    assert "共 3 筆紀錄" in out
    assert "其中 1 筆有逐項明細" in out


def test_範圍欄位本身要帶涵蓋率():
    """涵蓋率不能只放在獨立的一行。

    實測（2026-11-06 連跑三次）：獨立的【涵蓋率】那行有兩次被模型丟掉，
    其中一次還把範圍說成「最近七天的飲食紀錄」——把部分涵蓋講成全部。
    但模型**每次都會**轉述範圍，所以限制要寫進範圍的敘述裡，
    它照抄範圍時就一起抄出去了。

    通則：要模型一定說出某件事，就放進它本來就要說的那個欄位，
    而不是另外加一句話叫它記得說。
    """
    logs = [_log(_item("烤鯖魚", 100, 270, 24)), _bare_log(), _bare_log()]

    out = _rank_foods(logs, "protein_g", "per_100g", "desc", 3, 6)
    scope_line = out.splitlines()[0]

    assert "範圍：" in scope_line
    # 範圍那一行自己就要講出「只有一部分」
    assert "有逐項明細的 1 筆紀錄" in scope_line
    assert "共 3 筆" in scope_line


def test_全部都有明細時範圍不加警語():
    """沒有缺口就不要製造一個。

    每次都說「只涵蓋有明細的紀錄」會讓使用者以為永遠有東西漏掉，
    久了就跟所有被忽略的紅燈一樣沒人看。
    """
    out = _rank_foods([_log(_item("烤鯖魚", 100, 270, 24))], "protein_g", "per_100g", "desc", 3, 6)
    scope_line = out.splitlines()[0]

    assert "全部 1 筆紀錄" in scope_line
    assert "其餘只有整餐合計" not in scope_line


def test_完全沒有明細時要指路到另一個工具():
    """舊紀錄全部沒有明細時，不能只說「查不到」。

    那會讓模型回「沒有資料」，而其實換 rank_my_meals 就答得出餐的層級。
    降級要保留能回答的部分（pitfalls 第五節）。
    """
    out = _rank_foods([_bare_log(), _bare_log()], "protein_g", "per_100g", "desc", 3, 6)

    assert "沒有逐項明細" in out
    assert "rank_my_meals" in out


def test_完全沒有紀錄跟沒有明細要分得出來():
    """「你還沒開始記錄」跟「你的紀錄太舊」該給的回應完全不同。"""
    out = _rank_foods([], "protein_g", "per_100g", "desc", 3, 6)

    assert "沒有任何飲食紀錄" in out
    assert "涵蓋率" not in out


def test_有明細但都算不出密度():
    """份量為 0 的項目算不出 per_100g。

    這跟「沒有明細」不同——有明細但不可用，訊息要分開，
    否則看輸出的人會以為是落地機制壞了。
    """
    out = _rank_foods([_log(_item("不明食物", 0, 100, 5))], "protein_g", "per_100g", "desc", 3, 6)

    assert "沒有任何項目能計算" in out
    assert "涵蓋率" in out


# ── 佐證狀態要跟著數字一起出去 ────────────────────────────────


def test_有佐證的項目要標出食藥署編號():
    """✅/⚠️ 的區別在前端做了，模型這一側也要看得到。

    看不到的話，模型會用同樣的語氣講估算值和查證值。
    """
    logs = [_log(_item("烤鯖魚", 100, 270, 24, grounded=True, source_ref="A0100101"))]

    out = _rank_foods(logs, "protein_g", "per_100g", "desc", 3, 6)

    assert "A0100101" in out
    assert "資料限制" not in out  # 全部有佐證，不需要那段警告


def test_無佐證的項目要附上誤差警告():
    logs = [_log(_item("雞胸肉便當", 450, 660, 40))]

    out = _rank_foods(logs, "protein_g", "per_100g", "desc", 3, 6)

    assert "LLM 估算" in out
    assert "誤差可能很大" in out


def test_沒有洩漏user_id():
    """工具的輸出會進到模型的 context，再進到答案裡。

    user_id 是伺服器端決定的（SERVER_PROVIDED_ARGS），不該出現在文字裡。
    """
    out = _rank_foods([_log(_item("烤鯖魚", 100, 270, 24))], "protein_g", "per_100g", "desc", 3, 6)

    assert "user_id" not in out
