"""比較不同模型當 judge 的表現，判斷便宜模型夠不夠用。

跑法（在 backend/ 底下）：
    poetry run python -m evals.compare_judges

實驗設計：**同一批人工標註、同一個 judge prompt，只換模型。**
只變動一個因子，差異才能歸因到模型本身。

為什麼值得做：gpt-4o-mini 的輸入單價約是 gpt-4o 的 1/16。
如果 kappa 掉得不多，eval 成本就能降一個數量級；
如果掉很多，你也得到「這個任務便宜模型不夠用」的證據——
兩種結果都有價值，不做才是損失。

注意 gpt-4o 的判定已經在快取裡，所以這次實驗實際只會付 mini 那一輪的錢。
"""

import csv
from pathlib import Path

from evals.cache import JudgeCache
from evals.calibrate import _evidence
from evals.calibrate import cohens_kappa
from evals.judge import judge_rubric

RESULTS_DIR = Path(__file__).parent / "results"
MODELS = ["gpt-4o", "gpt-4o-mini"]

# 兩份標註集合併使用。31 列仍然很少，但比單獨 15 或 16 列稍微可靠一些。
ANNOTATED = [
    RESULTS_DIR / "run_20260831_1745.csv",
    RESULTS_DIR / "run_20260902_1546.csv",
]


def load_annotated() -> list[dict]:
    rows: list[dict] = []
    for path in ANNOTATED:
        with path.open(encoding="utf-8-sig") as f:
            rows.extend(list(csv.DictReader(f)))
    return [r for r in rows if r["label"].strip().lower() in {"pass", "fail"}]


def main() -> None:
    rows = load_annotated()
    cache = JudgeCache()

    # 證據每列算一次，兩個模型共用——確保兩邊看到的輸入完全相同。
    # 來源見 calibrate._evidence：agent 的 CSV 用歸檔的 tool_results，
    # 固定流程時代的 CSV 才退回重跑檢索。
    evidence = {id(row): _evidence(row) for row in rows}

    print(f"標註列數：{len(rows)}\n")
    print(f"{'模型':<16}{'一致率':>8}{'TPR':>8}{'TNR':>8}{'kappa':>9}")
    print("-" * 49)

    summaries = {}
    for model in MODELS:
        tp = fn = fp = tn = 0
        disagreements = []
        for row in rows:
            verdict = judge_rubric(
                question=row["question"],
                answer=row["answer"],
                criterion=row["rubric"],
                evidence=evidence[id(row)],
                cache=cache,
                model=model,
            ).verdict
            human = row["label"].strip().lower()

            if human == "fail" and verdict == "fail":
                tp += 1
            elif human == "fail":
                fn += 1
                disagreements.append((row, verdict))
            elif verdict == "fail":
                fp += 1
                disagreements.append((row, verdict))
            else:
                tn += 1

        n = tp + fn + fp + tn
        agreement = (tp + tn) / n
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) else float("nan")
        kappa = cohens_kappa(tp, fn, fp, tn)
        summaries[model] = (kappa, disagreements)
        print(f"{model:<16}{agreement:>7.0%}{tpr:>8.0%}{tnr:>8.0%}{kappa:>9.2f}")

    print(f"\n{cache.summary()}")

    for model, (_, disagreements) in summaries.items():
        if disagreements:
            print(f"\n{model} 與人工不一致的項目：")
            for row, verdict in disagreements:
                print(f"  [{row['example_id']}] {row['rubric'][:36]}")
                print(f"    人工={row['label']}  judge={verdict}")


if __name__ == "__main__":
    main()
