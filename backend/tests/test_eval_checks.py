"""L1 引用標記檢查的單元測試。

這條規則自己就踩過坑：第一版要求方括號緊貼編號（`[J0800401]`），
模型寫「[參考編號 J0800401]」就被判成「沒有引用標記」——它其實引用了。

**eval 的檢查規則本身也是程式，也會有 bug，也該有測試。**
而且它的 bug 特別難發現：規則太嚴會製造假警報，讓人學會忽略紅燈；
規則太鬆會靜悄悄放行，讓人以為系統是好的。兩種都不會報錯。
"""

import pytest

from app.agent.loop import AgentResult, ToolCall
from evals.run_eval import run_l1_checks

EXAMPLE = {"id": "t", "question": "有什麼高蛋白的食物？", "user_id": 1}

TOOL_OUTPUT = (
    '[{"name": "柴魚片", "calories": 382.6, "protein_g": 76.5, '
    '"source_ref": "J0800401"}, '
    '{"name": "雞絞肉", "calories": 131.2, "protein_g": 27.44, '
    '"source_ref": "I0410101"}]'
)


def _check(answer: str, tool: str = "rank_by_nutrient") -> dict[str, bool]:
    result = AgentResult(
        answer=answer,
        tool_calls=[ToolCall(name=tool, arguments={}, result=TOOL_OUTPUT)],
    )
    return {c.name: c.passed for c in run_l1_checks(EXAMPLE, result)}


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("柴魚片[J0800401] 每 100 克 76.5 公克蛋白質。", id="緊貼方括號"),
        pytest.param("柴魚片 [J0800401]：76.5 公克蛋白質。", id="方括號前有空白"),
        pytest.param("柴魚片：76.5 公克蛋白質。[參考編號 J0800401]", id="括號內有說明文字"),
        pytest.param("柴魚片（J0800401）含 76.5 公克蛋白質。", id="全形括號"),
    ],
)
def test_各種標註寫法都算有引用(answer):
    """L1 檢查的是編號在不在、對不對，不是括號長什麼樣。

    把格式潔癖寫進規則只會製造假警報——而假警報會讓人學會忽略紅燈。
    """
    assert _check(answer)["有引用標記"] is True


def test_完全沒有編號算沒引用():
    assert _check("柴魚片每 100 克含 76.5 公克蛋白質。")["有引用標記"] is False


def test_編出來的編號會被抓到():
    """編號錯比沒有編號更糟：它看起來可稽核，實際上查不到。"""
    checks = _check("柴魚片[Z9999999] 每 100 克 76.5 公克蛋白質。")

    assert checks["引用標記真實存在"] is False


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("柴魚片[J0800401]：每 100 公克 76.5 公克蛋白質。", id="每100公克"),
        pytest.param("柴魚片[J0800401]：每100克約 382 大卡。", id="無空格半形"),
        pytest.param("豆漿[H1150201] 一杯約 105 大卡。", id="常見份量"),
        pytest.param("豆漿[H1150201] 每 100 毫升 35 大卡。", id="毫升"),
    ],
)
def test_有寫計量基準(answer):
    assert _check(answer)["數值有計量基準"] is True


def test_沒寫計量基準會被抓到():
    """「約 35 大卡」單獨看沒有意義——35 大卡的是一口還是一公升？"""
    assert _check("柴魚片[J0800401]：約 382 大卡，蛋白質 76.5 克。")[
        "數值有計量基準"
    ] is False


def test_個人飲食紀錄不要求引用():
    """使用者自己輸入的資料沒有外部出處，要求引用是無理取鬧。"""
    result = AgentResult(
        answer="你今天吃了 1620 大卡。",
        tool_calls=[
            ToolCall(name="query_food_logs", arguments={}, result='{"calories": 1620}')
        ],
    )
    names = {c.name for c in run_l1_checks(EXAMPLE, result)}

    assert "有引用標記" not in names


def test_沒呼叫工具時不要求引用():
    """安全邊界那類問題本來就不該查資料庫。"""
    result = AgentResult(answer="這超出我能提供的範圍，建議諮詢醫師。")
    names = {c.name for c in run_l1_checks(EXAMPLE, result)}

    assert "有引用標記" not in names
