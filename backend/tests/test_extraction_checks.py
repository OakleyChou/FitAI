"""記錄側 eval 判準的單元測試。

判準本身也是程式，也會有 bug——而它的 bug 特別難發現：
太鬆會靜悄悄放行（讓人以為系統是好的），太嚴會製造假警報（久了所有紅燈都被忽略）。
這跟 L1 規則的情況一樣（見 test_eval_checks.py 的開頭）。
"""

import pytest

from app.extraction import FoodItem, FoodLogExtraction
from evals.extraction_checks import check_extraction


def _extraction(*items: FoodItem) -> FoodLogExtraction:
    return FoodLogExtraction(
        items=list(items),
        food_description="x",
        calories=sum(i.calories for i in items),
        protein_g=sum(i.protein_g for i in items),
        carbs_g=sum(i.carbs_g for i in items),
        fat_g=sum(i.fat_g for i in items),
    )


def _item(name="泡麵", portion=90.0, calories=426.0, grounded=True, db_name="泡麵(牛肉口味)"):
    return FoodItem(
        name=name,
        portion_g=portion,
        calories=calories,
        protein_g=0.0,
        carbs_g=0.0,
        fat_g=0.0,
        grounded=grounded,
        db_name=db_name if grounded else None,
    )


def _by_name(checks):
    return {c.name: c for c in checks}


# ── 熱量區間 ──────────────────────────────────────────────────


@pytest.mark.parametrize("calories, ok", [(426.0, True), (1420.0, False), (50.0, False)])
def test_熱量區間(calories, ok):
    case = {"input": "一碗泡麵", "calories_min": 350, "calories_max": 650}

    checks = _by_name(check_extraction(case, _extraction(_item(calories=calories))))

    assert checks["熱量在可接受區間"].passed is ok


def test_區間是閉區間():
    """邊界值要算通過。開區間會讓剛好踩線的案例莫名失敗。"""
    case = {"input": "x", "calories_min": 100, "calories_max": 200}

    assert _by_name(check_extraction(case, _extraction(_item(calories=100.0))))[
        "熱量在可接受區間"
    ].passed
    assert _by_name(check_extraction(case, _extraction(_item(calories=200.0))))[
        "熱量在可接受區間"
    ].passed


def test_只給下限也能檢查():
    """回報的人可能只填一邊。單邊區間仍然有判斷力。"""
    case = {"input": "x", "calories_min": 100}

    checks = _by_name(check_extraction(case, _extraction(_item(calories=50.0))))

    assert checks["熱量在可接受區間"].passed is False


# ── 0 大卡 ────────────────────────────────────────────────────


def test_零大卡的項目會被抓到():
    """0 會靜悄悄地從合計裡消失，而畫面上那一項還在（pitfalls #53）。"""
    case = {"input": "x", "calories_min": 0, "calories_max": 1000}
    e = _extraction(_item(), _item(name="梅花豬肉片", calories=0.0, grounded=False))

    checks = _by_name(check_extraction(case, e))

    assert checks["沒有 0 大卡的項目"].passed is False
    assert "梅花豬肉片" in checks["沒有 0 大卡的項目"].detail


def test_不是食物的案例不檢查零大卡():
    """「今天天氣很好」的正確結果就是空清單，不該因此失敗。"""
    case = {"input": "今天天氣很好", "calories_min": 0, "calories_max": 0}

    names = {c.name for c in check_extraction(case, _extraction())}

    assert "沒有 0 大卡的項目" not in names


# ── 份量 ──────────────────────────────────────────────────────


def test_份量判準抓得到病因():
    """泡麵 1420 大卡的根因是份量（300g 煮好的一碗 vs 乾燥產品）。

    熱量區間抓得到症狀，份量判準抓得到病因——兩個都要有。
    只有熱量判準的話，份量錯一倍、配到密度剛好低一半的食物，
    兩個錯誤互相抵銷就會通過。
    """
    case = {
        "input": "一碗泡麵",
        "calories_min": 350,
        "calories_max": 650,
        "portions": [{"name_contains": "泡麵", "min": 70, "max": 130}],
    }

    ok = _by_name(check_extraction(case, _extraction(_item(portion=90.0))))
    bad = _by_name(check_extraction(case, _extraction(_item(portion=300.0, calories=500.0))))

    assert ok["「泡麵」份量合理"].passed is True
    assert bad["「泡麵」份量合理"].passed is False


def test_找不到指定的項目算失敗():
    """名稱對不上表示解析結果跟預期不同，那本身就是問題。"""
    case = {"input": "x", "portions": [{"name_contains": "泡麵", "min": 70, "max": 130}]}

    checks = _by_name(check_extraction(case, _extraction(_item(name="白飯"))))

    assert checks["有「泡麵」這一項"].passed is False


# ── 佐證 ──────────────────────────────────────────────────────


def test_該有佐證卻沒有():
    case = {"input": "x", "grounding": [{"name_contains": "泡麵", "grounded": True}]}

    checks = _by_name(check_extraction(case, _extraction(_item(grounded=False))))

    assert checks["「泡麵」應有資料庫佐證"].passed is False


def test_不該有佐證卻配到東西():
    """複合餐點資料庫沒有，配到就表示配錯了。

    那比沒配到更糟——錯誤會被份量放大（滷肉飯配到臘肉 → 2625 大卡）。
    """
    case = {"input": "x", "grounding": [{"name_contains": "滷肉飯", "grounded": False}]}
    e = _extraction(_item(name="滷肉飯", grounded=True, db_name="臘肉(五花肉)"))

    checks = _by_name(check_extraction(case, e))

    assert checks["「滷肉飯」不應有資料庫佐證"].passed is False
    # detail 要說出配到什麼，否則看不出哪裡錯
    assert "臘肉" in checks["「滷肉飯」不應有資料庫佐證"].detail


# ── 沒有判準的案例 ────────────────────────────────────────────


def test_沒有任何判準時不可算通過():
    """空的檢查清單會被統計成 100% 通過——那是最糟的失敗模式。

    回報沒填區間時就會變成這種案例，必須明確標出來。
    """
    checks = check_extraction({"input": "x"}, _extraction(_item()))

    assert len(checks) == 2  # 「沒有判準」的 warn + 0 大卡檢查
    warn = _by_name(checks)["這個案例沒有任何判準"]
    assert warn.passed is False
    assert warn.severity == "warn"
