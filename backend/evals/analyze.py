"""比對「L1 自動檢查」與「人工標註」，找出檢查規則要修的地方。

跑法（在 backend/ 底下）：
    poetry run python -m evals.analyze evals/results/run_20260831_1745.csv

這支程式不評估模型，它評估的是**你的自動檢查**。
人工標註是黃金標準，L1 是想要逼近它的廉價代理——
兩者不一致的地方，就是代理不準的地方。

比對在「題目層級」而非「列層級」進行。
理由：L1 檢查（繁體中文、有引用標記…）和 rubric 項目（沒把低蛋白稱為高蛋白…）
是兩套不同的東西，一對一比對沒有意義。有意義的問題是
「這題 L1 有沒有示警」對上「這題人工有沒有判失敗」。
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

VALID_LABELS = {"pass", "fail"}


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：poetry run python -m evals.analyze <標註過的 csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    rows = load_rows(path)

    # 未標註的列直接擋下來。少標幾列會讓所有統計失真，
    # 而且失真的方式很隱蔽（那題會被當成「全部通過」）。
    unlabeled = [r for r in rows if r["label"].strip().lower() not in VALID_LABELS]
    if unlabeled:
        print(f"還有 {len(unlabeled)} 列沒標註（label 要填 pass 或 fail）：")
        for r in unlabeled[:10]:
            print(f"  - {r['example_id']} / {r['rubric']}")
        sys.exit(1)

    # ── 彙整到題目層級 ──────────────────────────────────────────
    # 一題只要有任何一條 rubric 沒過，這題就算失敗。
    examples: dict[str, dict] = {}
    for r in rows:
        # l1_failed 是舊格式（沒有 warn 分級）的欄位名，保留相容性，
        # 否則之前標註過的 CSV 就讀不了——那些標註是不可重生的資產。
        fail_field = r.get("l1_fail", r.get("l1_failed", ""))
        ex = examples.setdefault(
            r["example_id"],
            {
                "l1_failed": [c for c in fail_field.split(", ") if c],
                "l1_warned": [c for c in r.get("l1_warn", "").split(", ") if c],
                "human_fail": False,
                "reasons": [],
            },
        )
        if r["label"].strip().lower() == "fail":
            ex["human_fail"] = True
            ex["reasons"].append(f"{r['rubric']}｜{r['note'] or '（未填理由）'}")

    # ── 交叉表 ────────────────────────────────────────────────
    agree_pass = agree_fail = blind_spot = false_alarm = 0
    for ex in examples.values():
        l1_fail = bool(ex["l1_failed"])
        if l1_fail and ex["human_fail"]:
            agree_fail += 1
        elif not l1_fail and not ex["human_fail"]:
            agree_pass += 1
        elif not l1_fail and ex["human_fail"]:
            blind_spot += 1
        else:
            false_alarm += 1

    total = len(examples)
    agreement = (agree_pass + agree_fail) / total if total else 0

    print(f"檔案：{path.name}    題數：{total}\n")
    print("            人工 pass   人工 fail")
    print(f"L1 PASS        {agree_pass:^9}   {blind_spot:^9}  ← 盲區")
    print(f"L1 FAIL        {false_alarm:^9}   {agree_fail:^9}")
    print("                ↑ 假警報\n")
    print(f"一致率：{agreement:.0%}（{agree_pass + agree_fail}/{total}）")
    print("註：5 題的一致率極不穩定，這個數字只能當方向參考，不能當結論。\n")

    # ── 每條 L1 規則的表現 ──────────────────────────────────────
    # 一條規則若總是在人工判 pass 的題目上示警，它就是在製造噪音，
    # 該放寬或刪掉；反之若從未示警，可能是寫錯了或情境還沒遇到。
    fired: dict[str, dict[str, int]] = defaultdict(lambda: {"on_pass": 0, "on_fail": 0})
    for ex in examples.values():
        for check in ex["l1_failed"]:
            fired[check]["on_fail" if ex["human_fail"] else "on_pass"] += 1

    # warn 級規則另外統計。它們不計入上面的交叉表，
    # 但同樣要追蹤精準度——這是判斷「能不能升級成 fail」的依據。
    for ex in examples.values():
        for check in ex["l1_warned"]:
            fired[f"{check}(warn)"]["on_fail" if ex["human_fail"] else "on_pass"] += 1

    if fired:
        print("每條 L1 規則示警的情況：")
        print(f"  {'規則':<14}{'人工也判 fail':<16}{'人工判 pass（假警報）'}")
        for check, counts in sorted(fired.items()):
            print(f"  {check:<16}{counts['on_fail']:^14}{counts['on_pass']:^18}")
        print("  註：(warn) 是試驗中的規則，累積足夠精準度後才升級成 fail。\n")

    # ── 逐題列出不一致的地方 ────────────────────────────────────
    print("需要處理的題目：")
    for name, ex in examples.items():
        l1_fail = bool(ex["l1_failed"])
        if l1_fail and not ex["human_fail"]:
            print(f"\n  [假警報] {name}")
            print(f"    L1 示警：{', '.join(ex['l1_failed'])}")
            print("    → 人工判定沒問題，這條規則太嚴，考慮放寬或加例外")
        elif not l1_fail and ex["human_fail"]:
            print(f"\n  [盲區] {name}")
            for reason in ex["reasons"]:
                print(f"    人工判失敗：{reason}")
            print("    → L1 完全沒抓到，需要新規則，或這類問題只能交給 L2")

    if blind_spot == 0 and false_alarm == 0:
        print("  （無不一致，L1 目前與人工判斷吻合）")


if __name__ == "__main__":
    main()
