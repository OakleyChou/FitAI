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

from evals.cache import JudgeCache
from evals.judge import judge_rubric

VALID_LABELS = {"pass", "fail"}


def _evidence(row: dict) -> str:
    """從歸檔的一列取出「AI 當時看到的東西」。

    兩種 CSV 形態都要支援：

        agent（2026-09-14 之後）  tool_results 欄，含呼叫參數與工具回傳
        固定流程（之前的兩次跑）    沒有這一欄，只能退回重跑檢索

    重跑檢索是**下策**，因為它重現的是「當時大概會拿到什麼」而不是
    「當時真的拿到什麼」。固定流程的檢索沒有隨機性，所以退路勉強可用；
    agent 不行——它自己選工具，重跑等於換一份證據。

    找不到證據時直接失敗，不要靜悄悄地送出空的 <evidence>：
    那會讓 judge 把每一條「數值有依據」都判成 fail，而數字看起來仍然正常。
    """
    archived = row.get("tool_results", "").strip()
    if archived:
        return archived

    # 固定流程時代的 CSV。延後 import：agent 的 CSV 不需要碰資料庫。
    from app.rag.qa import build_contexts

    user_ctx, know_ctx, _ = build_contexts(row["question"], user_id=1)
    return (
        f"<user_food_log>\n{user_ctx}\n</user_food_log>\n\n"
        f"<nutrition_reference>\n{know_ctx}\n</nutrition_reference>"
    )


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

    # judge 判定的快取。校準常常要反覆跑（改一條判準就重跑整批），
    # 輸入沒變的列直接讀快取，不重複付費。
    cache = JudgeCache()

    # 以「fail」為陽性：evals 的目的是抓問題，抓到問題才是有價值的偵測。
    tp = fn = fp = tn = 0
    disagreements: list[dict] = []

    for row in rows:
        example_id = row["example_id"]

        result = judge_rubric(
            question=row["question"],
            answer=row["answer"],
            criterion=row["rubric"],
            evidence=_evidence(row),
            cache=cache,
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
    print(f"樣本數：{total}    {cache.summary()}\n")
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
