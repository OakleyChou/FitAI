"""校準 LLM-as-Judge：拿 judge 的判定跟人工標註比對，量它有多可信。

跑法（在 backend/ 底下）：
    poetry run python -m evals.calibrate evals/results/run_20260831_1745.csv

這支程式評估的不是 QA 系統，是**judge 本身**。
人工標註是黃金標準；judge 是想要規模化取代人工的代理。
沒有這一步就直接用 judge 跑大量資料，等於用一把沒校準過的尺量東西。

為什麼不只看整體一致率：
    假設 90% 的輸出本來就是好的，一個「永遠回答 pass」的 judge
    一致率就有 90%，但它完全沒用。這是類別不平衡下 accuracy 會騙人的老問題，
    所以要分開看 TPR（會不會漏放）和 TNR（會不會亂咬），
    再用 Cohen's kappa 扣掉隨機猜中的部分。
"""

import csv
import sys
from pathlib import Path

from app.rag.qa import build_contexts
from evals.judge import judge_rubric

VALID_LABELS = {"pass", "fail"}


def cohens_kappa(tp: int, fn: int, fp: int, tn: int) -> float:
    """兩位評審（人工 vs judge）在二元判定上的一致度，已扣除隨機一致。

    kappa = (實際一致率 − 期望隨機一致率) / (1 − 期望隨機一致率)
    1 = 完全一致，0 = 跟瞎猜一樣，負值 = 比瞎猜還差。
    """
    n = tp + fn + fp + tn
    if n == 0:
        return 0.0
    observed = (tp + tn) / n
    # 兩邊各自判 fail / pass 的邊際機率相乘，就是隨機情況下的期望一致率
    p_fail = ((tp + fn) / n) * ((tp + fp) / n)
    p_pass = ((fp + tn) / n) * ((fn + tn) / n)
    expected = p_fail + p_pass
    if expected >= 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：poetry run python -m evals.calibrate <標註過的 csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    unlabeled = [r for r in rows if r["label"].strip().lower() not in VALID_LABELS]
    if unlabeled:
        print(f"還有 {len(unlabeled)} 列沒標註，無法校準。")
        sys.exit(1)

    # context 每題只算一次，同一題的多條 rubric 共用。
    # 這裡重算而不是從 CSV 讀，是因為 context 可重現而答案不可重現——
    # 答案一定要用歸檔的那個，否則 judge 跟人工看的不是同一份東西。
    context_cache: dict[str, tuple[str, str]] = {}

    # 以「fail」為陽性：evals 的目的是抓問題，抓到問題才是有價值的偵測。
    tp = fn = fp = tn = 0
    disagreements: list[dict] = []

    for row in rows:
        example_id = row["example_id"]
        if example_id not in context_cache:
            user_ctx, know_ctx, _ = build_contexts(row["question"], user_id=1)
            context_cache[example_id] = (user_ctx, know_ctx)
        user_ctx, know_ctx = context_cache[example_id]

        result = judge_rubric(
            question=row["question"],
            answer=row["answer"],
            user_context=user_ctx,
            knowledge_context=know_ctx,
            criterion=row["rubric"],
        )

        human = row["label"].strip().lower()
        machine = result.verdict

        if human == "fail" and machine == "fail":
            tp += 1
        elif human == "fail" and machine == "pass":
            fn += 1
            disagreements.append({**row, "judge": machine, "critique": result.critique})
        elif human == "pass" and machine == "fail":
            fp += 1
            disagreements.append({**row, "judge": machine, "critique": result.critique})
        else:
            tn += 1

        mark = "OK " if human == machine else "!! "
        print(f"{mark}[{example_id}] {row['rubric'][:28]:<30} 人工={human:<4} judge={machine}")

    total = tp + fn + fp + tn
    agreement = (tp + tn) / total if total else 0
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")

    print("\n" + "=" * 66)
    print(f"樣本數：{total}\n")
    print("                judge fail   judge pass")
    print(f"人工 fail          {tp:^10}   {fn:^10}")
    print(f"人工 pass          {fp:^10}   {tn:^10}\n")
    print(f"整體一致率 : {agreement:.0%}   ← 類別不平衡時會虛高，不能單看")
    print(f"TPR (抓到率): {tpr:.0%}   ← 人工判 fail 的，judge 抓到幾成")
    print(f"TNR (不亂咬): {tnr:.0%}   ← 人工判 pass 的，judge 沒誤報幾成")
    print(f"Cohen's kappa: {cohens_kappa(tp, fn, fp, tn):.2f}   ← 扣掉隨機一致後的真實一致度")

    if disagreements:
        print("\n不一致的項目（judge prompt 或 rubric 要修的地方）：")
        for d in disagreements:
            print(f"\n  [{d['example_id']}] {d['rubric']}")
            print(f"    人工={d['label']}  judge={d['judge']}")
            print(f"    人工理由：{d['note'] or '（未填）'}")
            print(f"    judge 理由：{d['critique'][:140]}")


if __name__ == "__main__":
    main()
