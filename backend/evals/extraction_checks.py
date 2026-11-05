"""記錄側 eval 的判準。純函式，不呼叫 LLM、不碰資料庫。

## 為什麼判準是「區間」而不是「正確答案」

營養估算沒有唯一正確答案。「一碗泡麵」可以是 390 也可以是 550，
取決於哪個牌子、有沒有喝湯。要求精確值會逼人亂填，而亂填的判準比沒有判準更糟。

所以每個案例給一個**可接受區間**，判準是「落在區間內」。
區間由回報的人決定（見 `app/routers/feedback.py`）——
那是他看到數字覺得怪的那一刻的判斷，比事後補的標準答案可信。

## 為什麼不只檢查熱量

熱量落在區間內有可能是**湊巧**：份量錯一倍、配到熱量密度剛好低一半的食物，
兩個錯誤互相抵銷。所以另外檢查三件可以確定性驗證的事：

    有沒有 0 大卡的項目        0 會靜悄悄地從合計消失（pitfalls #53）
    份量在不在合理範圍         泡麵 1420 大卡的根因是份量，不是熱量（#51）
    該不該有資料庫佐證         複合餐點不該有，單品飲料該有（#43）

**答案對不代表過程對**——這是問答側學到的事（#28），記錄側一樣適用。
"""

from dataclasses import dataclass


@dataclass
class CheckResult:
    """一條判準的結果。沿用 run_eval 的形狀，兩邊的輸出才能用同一套方式讀。"""

    name: str
    passed: bool
    detail: str = ""
    severity: str = "fail"


def _in_range(value: float, lo: float | None, hi: float | None) -> bool:
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def check_extraction(case: dict, extraction) -> list[CheckResult]:
    """檢查一次擷取結果是否符合這個案例的期望。

    Args:
        case: 測試案例。必要欄位 `input`；其餘都是選填的判準，
            沒給就不檢查那一項——回報來的案例通常只有熱量區間。
        extraction: `extract_food_log` 的回傳值。

    Returns:
        檢查結果清單。沒有任何判準可檢查時回傳一條 warn，
        而不是回空清單假裝通過——**空的檢查清單會被統計成 100% 通過**。
    """
    checks: list[CheckResult] = []
    # 「案例自己指定的判準」有幾條。不能用 len(checks) 判斷——
    # 0 大卡那條是所有案例都自動加的，加了之後 checks 就非空，
    # 於是「這個案例沒有判準」的警告永遠不會觸發，
    # 沒填區間的回報就會靜悄悄通過。那正是這條警告要防的事。
    case_driven = 0

    lo, hi = case.get("calories_min"), case.get("calories_max")
    if lo is not None or hi is not None:
        case_driven += 1
        ok = _in_range(extraction.calories, lo, hi)
        checks.append(
            CheckResult(
                "熱量在可接受區間",
                ok,
                f"{extraction.calories:.0f} 大卡（期望 {lo}~{hi}）",
            )
        )

    # 0 大卡的項目會靜悄悄地從合計裡消失，而畫面上那一項還在。
    # 這條不需要案例指定，所有案例都該過。
    zero = [i.name for i in extraction.items if i.calories == 0]
    if extraction.items:
        checks.append(
            CheckResult(
                "沒有 0 大卡的項目",
                not zero,
                f"估不出熱量：{'、'.join(zero)}" if zero else "",
            )
        )

    # 份量的檢查。泡麵 1420 大卡的根因是份量（300g 煮好的一碗 vs 乾燥產品），
    # 熱量區間抓得到症狀，這條抓得到病因。
    for spec in case.get("portions", []):
        case_driven += 1
        name, p_lo, p_hi = spec["name_contains"], spec.get("min"), spec.get("max")
        matched = [i for i in extraction.items if name in i.name]
        if not matched:
            checks.append(CheckResult(f"有「{name}」這一項", False, "解析結果裡找不到"))
            continue
        item = matched[0]
        checks.append(
            CheckResult(
                f"「{name}」份量合理",
                _in_range(item.portion_g, p_lo, p_hi),
                f"{item.portion_g:.0f} 公克（期望 {p_lo}~{p_hi}）",
            )
        )

    # 該不該有資料庫佐證。複合餐點（便當、吐司夾餡）資料庫沒有，
    # 配到東西就表示配錯了——那比沒配到更糟，錯誤會被份量放大（#43）。
    for spec in case.get("grounding", []):
        case_driven += 1
        name, want = spec["name_contains"], spec["grounded"]
        matched = [i for i in extraction.items if name in i.name]
        if not matched:
            checks.append(CheckResult(f"有「{name}」這一項", False, "解析結果裡找不到"))
            continue
        item = matched[0]
        actual = item.grounded
        detail = f"配到 {item.db_name}" if actual else "無佐證（LLM 估算）"
        checks.append(
            CheckResult(
                f"「{name}」{'應有' if want else '不應有'}資料庫佐證",
                actual == want,
                detail,
            )
        )

    if case_driven == 0:
        checks.append(
            CheckResult(
                "這個案例沒有任何判準",
                False,
                "至少要給 calories_min/max，否則跑了也不知道對不對",
                severity="warn",
            )
        )
    return checks
