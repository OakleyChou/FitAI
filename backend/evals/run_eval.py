"""跑 eval 資料集：重置資料 → 執行 QA → 套 L1 自動檢查 → 產出待人工標註的 CSV。

跑法（在 backend/ 底下）：
    poetry run python -m evals.run_eval

分兩層（Hamel Husain 的 Level 1 / Level 2）：

L1 自動斷言 —— 確定性、幾行程式、幾乎零成本。能用 L1 解決的絕不用 LLM judge，
   又慢又貴還不穩定。這裡的檢查全是正則或字串比對。

L2 人工標註 —— 需要判斷力的部分（例如「10.7g 算不算高蛋白」），程式做不到。
   輸出成 CSV 讓人逐項標記。這批標註之後會成為 LLM-as-Judge 的校準基準：
   沒有人工標註當黃金標準，就無從驗證 judge 本身可不可信。

注意 dataset.jsonl 裡沒有「標準答案」欄位，只有 expect 和 rubric。
LLM 輸出有無數種合理說法，存固定字串去比對只會得到永遠失敗的測試。
"""

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.rag.qa import QAResult, answer_question
from scripts.seed_dev_data import seed

EVALS_DIR = Path(__file__).parent
DATASET_PATH = EVALS_DIR / "dataset.jsonl"
RESULTS_DIR = EVALS_DIR / "results"

MAX_ANSWER_CHARS = 200

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
CITATION_RE = re.compile(r"\[\d+\]")

# 只檢查「緊接著營養單位」的數字。
# 這條規則要防的是編造營養數據，不是份量假設——
# 「一杯 300ml」是 prompt 明文要求模型自行估算的，不該被當成編造。
NUTRIENT_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:大卡|kcal|公克|克|毫克|[gG]\b|mg)")

# 允許模型做份量換算時用的倍率（半份、兩份、三份…）。
# 沒有這些，「35 大卡 × 3 杯 = 105」會被誤判成編造。
SCALE_FACTORS = [0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 10]

# 四捨五入容差：模型常把 34.98 講成 35。
DERIVE_TOLERANCE = 0.55

# 只有這些類型的問題會用到營養知識庫，才需要引用標記。
# 個人問題（「我今天吃了什麼」）的答案全部來自 user_food_log，
# 硬要求它標引用是規則設計錯誤，不是模型的錯。
CITATION_REQUIRED_TYPES = {"knowledge", "mixed"}


@dataclass
class CheckResult:
    """一條 L1 檢查的結果。

    severity 區分兩種規則：
    - "fail"：確定性判斷，或已用人工標註驗證過精準度的規則。違反就是錯。
    - "warn"：還在試驗階段的啟發式規則。值得看一眼，但不列入失敗統計。

    分級的用意是避免「狼來了」——一條會製造假警報的規則若和確定性規則
    混在同一個 FAIL 池裡，久了所有警告都會被忽略。
    規則累積足夠的驗證資料後，再從 warn 升級成 fail。
    """

    name: str
    passed: bool
    detail: str = ""
    severity: str = "fail"


def _has_simplified_chinese(text: str) -> bool:
    """偵測簡體字。

    查每個中文字的 Unicode 名稱有沒有 SIMPLIFIED 標記——
    不需要維護對照表，也不會漏掉罕見字。
    """
    for ch in text:
        if "一" <= ch <= "鿿":
            try:
                if "SIMPLIFIED" in unicodedata.name(ch):
                    return True
            except ValueError:
                continue
    return False


def _is_derivable(target: float, pool: list[float]) -> bool:
    """target 能否由 pool 裡的數字經「一步」四則運算得出（含捨入容差）。

    為什麼需要這個：模型的合理推導會產生 context 裡沒有的數字。
    實際遇過的兩個——
        105 = 35 大卡 × 3 杯      （份量換算）
        32  = 96 目標 − 64 攝取   （差距計算）
    只比對字面數字的話，這兩個都會被誤判成「編造」。

    只做一步運算是刻意的：兩步以上的組合數會爆炸，而且幾乎任何數字
    都能被湊出來——檢查就失去意義了。寧可漏抓，不要放行過寬。
    """
    if any(abs(target - n) <= DERIVE_TOLERANCE for n in pool):
        return True

    operands = pool + SCALE_FACTORS
    for a in operands:
        for b in operands:
            candidates = [a + b, a - b, a * b]
            if b:
                candidates.append(a / b)
            if any(abs(target - c) <= DERIVE_TOLERANCE for c in candidates):
                return True
    return False


def run_l1_checks(example: dict, result: QAResult) -> list[CheckResult]:
    """確定性檢查，每一條都來自實際觀察到的失敗。

    每個 bug 都變成一條測試——這是 eval 資料集長大的方式，
    不是一開始就設計好一套完美指標。
    """
    answer = result.answer
    checks: list[CheckResult] = []

    # 實際發生過：「我將乐意協助你」
    checks.append(CheckResult("繁體中文", not _has_simplified_chinese(answer)))

    checks.append(
        CheckResult("長度≤200", len(answer) <= MAX_ANSWER_CHARS, f"{len(answer)} 字")
    )

    checks.append(CheckResult("不說「使用者」", "使用者" not in answer))

    if example["type"] in CITATION_REQUIRED_TYPES and result.retrieved:
        checks.append(CheckResult("有引用標記", bool(CITATION_RE.search(answer))))

    # Groundedness 的廉價近似：答案裡的營養數值要能在 context 中找到，
    # 或能由 context 的數字一步算出。
    #
    # 列為 warn 而非 fail：第一版 5 題就製造了 2 個假警報（105、32），
    # 加了算術推導之後應該會好轉，但它本質仍是啟發式，還沒有足夠的
    # 人工標註證明它可信。累積更多樣本、確認精準度後再考慮升級成 fail。
    context = result.user_context + result.knowledge_context
    context_numbers = [float(n) for n in NUMBER_RE.findall(context)]
    claimed = {float(n) for n in NUTRIENT_VALUE_RE.findall(answer)}
    ungrounded = sorted(n for n in claimed if not _is_derivable(n, context_numbers))
    checks.append(
        CheckResult(
            "數值有出處",
            not ungrounded,
            f"找不到出處：{', '.join(f'{n:g}' for n in ungrounded)}" if ungrounded else "",
            severity="warn",
        )
    )

    return checks


def load_dataset() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    # 先把測試資料重置成已知狀態。
    # 沒有這一步，eval 就不可重現——seed 的日期是相對於執行當下算的，
    # 隔天再跑「今天」就沒有紀錄了，個人問題全部答不出來，
    # 而你會誤以為是 prompt 改壞了。
    seeded = seed()
    print(f"已重置測試資料：{seeded} 筆 food_logs\n")

    examples = load_dataset()
    rows: list[dict] = []
    total_strict = passed_strict = warn_count = 0

    for example in examples:
        result = answer_question(example["question"], user_id=example["user_id"])
        checks = run_l1_checks(example, result)

        # 只有 severity="fail" 的規則計入通過率；warn 另外統計。
        # 混在一起算的話，一條噪音規則會把整體分數拖低，
        # 看起來像模型退步，實際上只是規則太嚴。
        strict = [c for c in checks if c.severity == "fail"]
        failed = [c.name for c in strict if not c.passed]
        warned = [c.name for c in checks if c.severity == "warn" and not c.passed]
        total_strict += len(strict)
        passed_strict += sum(c.passed for c in strict)
        warn_count += len(warned)

        print("=" * 78)
        print(f"[{example['id']}]  type={example['type']}")
        print(f"Q: {example['question']}")
        print("-" * 78)
        print(result.answer)
        print("-" * 78)
        for c in checks:
            if c.passed:
                mark = "PASS"
            else:
                mark = "FAIL" if c.severity == "fail" else "WARN"
            suffix = f"  ({c.detail})" if c.detail else ""
            print(f"  [{mark}] {c.name}{suffix}")
        print()

        # 每個 rubric 項目一列，讓標註者逐項判斷而不是給整體印象分。
        # 拆細的好處是失敗時能直接指出「哪一條沒過」，而不是「感覺不太好」。
        for item in example["rubric"]:
            rows.append(
                {
                    "example_id": example["id"],
                    "type": example["type"],
                    "question": example["question"],
                    "answer": result.answer,
                    "l1_fail": ", ".join(failed),
                    "l1_warn": ", ".join(warned),
                    "rubric": item,
                    "label": "",  # 由你填 pass / fail
                    "note": "",  # 填 fail 的理由，之後會變成 judge 的 rubric
                }
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"run_{stamp}.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        # utf-8-sig：加 BOM，Excel 開中文 CSV 才不會變亂碼
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 78)
    print(f"L1 嚴格規則：{passed_strict}/{total_strict} 通過")
    print(f"L1 警告（不計入通過率）：{warn_count} 項")
    print(f"L2 待標註：{len(rows)} 列 → {out_path}")
    print("請在 label 欄填 pass / fail，fail 的在 note 欄寫一句理由。")


if __name__ == "__main__":
    main()
