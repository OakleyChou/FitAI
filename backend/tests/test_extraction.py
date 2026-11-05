"""兩階段擷取的組合邏輯。純函式，不呼叫 LLM、不碰資料庫。

這一層測的是**最後一步的算術**：份量換算、缺值處理、加總、佐證旗標。
前三步（LLM → 檢索 → LLM）有外部依賴，但錯最容易發生在這裡——
而「450 毫升乘以每 100 公克 27 大卡該得到 121.5」這件事
不需要 LLM 來驗證，也不該每次驗證都付錢。
"""

import pytest

from app.extraction import (
    ItemMatch,
    ParsedItem,
    build_extraction,
    name_matches,
    scale_to_portion,
)

# 食藥署資料庫裡真實存在的一筆，就是這次 bug 的關鍵證據。
MILK_TEA = {"name": "鮮奶茶(無糖)", "calories": 27.0, "protein_g": 2.0, "carbs_g": 3.4, "fat_g": 1.0}
CHICKEN = {"name": "雞絞肉", "calories": 131.0, "protein_g": 27.44, "carbs_g": 0.0, "fat_g": 1.56}


# ── 名稱配對的防線 ────────────────────────────────────────────


@pytest.mark.parametrize(
    "user_name, db_name",
    [
        pytest.param("無糖鮮奶茶", "鮮奶茶(無糖)", id="詞序與修飾語不同"),
        pytest.param("可樂", "可樂", id="完全相同"),
        pytest.param("豆漿", "豆漿(無糖)", id="資料庫多了括號修飾語"),
    ],
)
def test_名稱相符時接受配對(user_name, db_name):
    assert name_matches(user_name, db_name) is True


@pytest.mark.parametrize(
    "user_name, db_name, 錯法",
    [
        pytest.param("一碗滷肉飯", "臘肉(五花肉)", "整碗飯套上五花肉的密度，錯 4 倍", id="滷肉飯"),
        pytest.param("漢堡", "冷凍牛肉漢堡肉", "整個漢堡配到肉排", id="漢堡"),
        pytest.param("鮪魚蛋吐司", "鮪魚肚", "複合餐點配到單一食材", id="鮪魚蛋吐司"),
        pytest.param("雞胸肉便當", "雞胸骨湯", "便當配到湯", id="便當"),
    ],
)
def test_名稱差太多時拒絕配對(user_name, db_name, 錯法):
    """這四個都是實際跑出來的錯誤配對。

    LLM 會把複合餐點配到其中一種高密度食材，然後套用整份餐點的重量——
    一個系統性的乘法錯誤，而且不會報錯。
    """
    assert name_matches(user_name, db_name) is False, 錯法


def test_拒絕配對時退回估算並記錄原因():
    """被排除的候選要留在 note 裡。

    不留的話，「為什麼這項沒有佐證」要重跑一次才知道，
    而重跑的結果可能不一樣（LLM 的非決定性）。
    """
    parsed = [ParsedItem(name="一碗滷肉飯", portion_g=500.0, portion_note="一碗約 500 公克")]
    matches = [ItemMatch(index=0, source_ref="BAD", calories=600.0)]
    db_rows = {"BAD": {"name": "臘肉(五花肉)", "calories": 525.0}}

    result = build_extraction("一碗滷肉飯", parsed, matches, db_rows)

    item = result.items[0]
    assert item.grounded is False
    # 關鍵：沒有變成 525 × 5 = 2625
    assert item.calories == 600.0
    assert "臘肉" in item.note


# ── 份量換算 ──────────────────────────────────────────────────


def test_份量換算():
    """這就是舊版錯掉的那個計算。

    一杯 450 毫升的無糖鮮奶茶：27 大卡/100g × 4.5 = 121.5 大卡。
    舊版讓 LLM 自己估，它回 50 大卡——低估 2.4 倍，而且不會報錯。
    """
    n = scale_to_portion(MILK_TEA, 450.0)

    assert n["calories"] == pytest.approx(121.5)
    assert n["protein_g"] == pytest.approx(9.0)


def test_缺值當零而不是爆掉():
    """食藥署資料有些欄位沒測，meta 保留 key 設 None（見 pitfalls #32）。

    當 0 會低估，但比讓整項計算丟 TypeError 好——
    而且 grounded 旗標仍然誠實標示數值來源。
    """
    n = scale_to_portion({"calories": 100.0, "protein_g": None}, 200.0)

    assert n["calories"] == pytest.approx(200.0)
    assert n["protein_g"] == 0.0
    assert n["fat_g"] == 0.0  # key 根本不存在


# ── 組合 ──────────────────────────────────────────────────────


def test_有佐證的項目用資料庫數值():
    parsed = [ParsedItem(name="無糖鮮奶茶", portion_g=450.0, portion_note="一杯約 450 毫升")]
    matches = [ItemMatch(index=0, source_ref="X0000001", reason="同一種飲品")]

    result = build_extraction("一杯無糖鮮奶茶", parsed, matches, {"X0000001": MILK_TEA})

    item = result.items[0]
    assert item.grounded is True
    assert item.source_ref == "X0000001"
    # db_name 要顯示出來：使用者說「無糖鮮奶茶」、資料庫叫「鮮奶茶(無糖)」，
    # 不給他看正式名稱就無法判斷配對對不對。
    assert item.db_name == "鮮奶茶(無糖)"
    assert item.calories == pytest.approx(121.5)
    assert result.all_grounded is True


def test_沒佐證的項目用LLM估算值並標記():
    """複合餐點資料庫查不到。數值仍由 LLM 估，但必須標示出來。"""
    parsed = [ParsedItem(name="鮪魚蛋吐司", portion_g=150.0)]
    matches = [
        ItemMatch(
            index=0,
            source_ref=None,
            reason="資料庫只有鮪魚肚等食材，沒有這個複合餐點",
            calories=380.0,
            protein_g=20.0,
            carbs_g=35.0,
            fat_g=16.0,
        )
    ]

    result = build_extraction("鮪魚蛋吐司", parsed, matches, {})

    item = result.items[0]
    assert item.grounded is False
    assert item.source_ref is None
    assert item.calories == 380.0
    assert result.all_grounded is False
    assert result.grounded_count == 0


def test_混合情形的合計():
    parsed = [
        ParsedItem(name="無糖鮮奶茶", portion_g=450.0),
        ParsedItem(name="鮪魚蛋吐司", portion_g=150.0),
    ]
    matches = [
        ItemMatch(index=0, source_ref="X0000001"),
        ItemMatch(index=1, source_ref=None, calories=380.0, protein_g=20.0),
    ]

    result = build_extraction("早餐", parsed, matches, {"X0000001": MILK_TEA})

    assert result.calories == pytest.approx(121.5 + 380.0)
    assert result.grounded_count == 1
    assert result.all_grounded is False


def test_配對結果的順序不影響對應():
    """模型回傳的順序不保證，必須用 index 對應。

    這跟 embedding 那個坑是同一類：依賴回傳順序的程式碼不會報錯，
    只會把 A 的數值掛到 B 身上（見 pitfalls 的 resp.data 排序）。
    """
    parsed = [
        ParsedItem(name="無糖鮮奶茶", portion_g=100.0),
        ParsedItem(name="雞絞肉", portion_g=100.0),
    ]
    # 刻意把順序顛倒
    matches = [
        ItemMatch(index=1, source_ref="CHICKEN"),
        ItemMatch(index=0, source_ref="MILKTEA"),
    ]

    result = build_extraction(
        "亂序", parsed, matches, {"MILKTEA": MILK_TEA, "CHICKEN": CHICKEN}
    )

    assert result.items[0].db_name == "鮮奶茶(無糖)"
    assert result.items[1].db_name == "雞絞肉"


def test_配對指向不存在的編號時退回估算():
    """模型可能編出一個不存在的 source_ref。

    那時不能當成有佐證——會拿不到數值然後整項變 0 卻標示 grounded。
    """
    parsed = [ParsedItem(name="某種食物", portion_g=100.0)]
    matches = [ItemMatch(index=0, source_ref="NOPE9999", calories=200.0)]

    result = build_extraction("x", parsed, matches, {})

    assert result.items[0].grounded is False
    assert result.items[0].calories == 200.0


def test_沒有配對結果時不會爆掉():
    """模型可能漏掉某一項。那一項退回 0 並標示未佐證，不是整個請求失敗。"""
    parsed = [ParsedItem(name="某種食物", portion_g=100.0)]

    result = build_extraction("x", parsed, [], {})

    assert result.items[0].grounded is False
    assert result.items[0].calories == 0.0


def test_估不出熱量時會標示請手動修正():
    """0 大卡是最糟的輸出：那一餐會靜悄悄地不計入每日合計。

    使用者看不出有東西漏掉，所以要在 note 裡講明。
    """
    parsed = [ParsedItem(name="某種食物", portion_g=100.0)]

    result = build_extraction("x", parsed, [], {})

    assert "手動修正" in result.items[0].note


def test_不是食物的輸入回空結果():
    """「今天天氣很好」不該拋例外——那是使用者打錯，不是系統錯誤。"""
    result = build_extraction("今天天氣很好", [], [], {})

    assert result.items == []
    assert result.calories == 0
    # 空清單不算「全部有佐證」——沒有項目談不上佐證
    assert result.all_grounded is False


def test_負數被夾到零():
    """ge=0 的欄位若收到負數，Pydantic 會拋 ValidationError 讓整個請求失敗。

    模型偶爾會回負值，而那不該讓使用者的記錄存不進去。
    """
    parsed = [ParsedItem(name="x", portion_g=100.0)]
    matches = [ItemMatch(index=0, source_ref=None, calories=-50.0)]

    result = build_extraction("x", parsed, matches, {})

    assert result.items[0].calories == 0.0
