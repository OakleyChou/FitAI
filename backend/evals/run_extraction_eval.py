"""記錄側的 eval：驗 `extract_food_log` 估得準不準。

跑法（在 backend/ 底下）：

    poetry run python -m evals.run_extraction_eval              # 檔案裡的案例
    poetry run python -m evals.run_extraction_eval --reports    # 加上資料庫的待處理回報
    poetry run python -m evals.run_extraction_eval --runs 3     # 每個案例跑三次看穩定度
    poetry run python -m evals.run_extraction_eval --baseline results/x.json

## 為什麼需要這一層

問答側有 8 題 eval，記錄側**從 7 月寫到 11 月零自動化檢查**（pitfalls #46）。
所以每個 bug 都是使用者手動踩到才發現的：鮮奶茶 50 大卡、泡麵 1420 大卡、
五片豬肉片 698 大卡。一個一個修是打地鼠，而且每次只驗證剛改的那一個案例——
泡麵那次就差點出事，改另一件事時順手加的 wrapper 讓多項目輸入回 500。

## 兩個資料來源

    evals/extraction_dataset.jsonl    進版控的穩定案例（回歸守衛）
    feedback_reports 表               使用者剛回報、還沒轉成案例的

後者用 `--reports` 帶進來。這樣你在前端報完就能立刻跑，不必等人手動搬。
確認要長期守住的，再用 `--promote` 寫進檔案（那會標記 converted_to_eval）。

## 為什麼不自動修

eval 的作用是**告訴你壞了**，不是修它。照回報自動調 prompt 的話，
那份回報就變成訓練資料——pitfalls #25：「每次盯著資料修東西，
那份資料就變成訓練資料」。到時候測試集 100% 通過，
而你完全不知道系統在沒見過的輸入上表現如何。

## 這個 eval 會花錢

每個案例每次執行 2～3 次 LLM 呼叫（解析、配對，有缺口時再補估）。
9 個案例跑一輪約 20 次呼叫。`--runs 3` 就是三倍。
刻意不做成 CI 每次跑——改了擷取流程才跑。
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from app.database import SessionLocal
from app.llm import extract_food_log
from app.models import FeedbackReport
from evals.extraction_checks import check_extraction

DATASET_PATH = Path(__file__).parent / "extraction_dataset.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def load_cases() -> list[dict]:
    if not DATASET_PATH.exists():
        return []
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_report_cases(user_id: int = 1) -> list[dict]:
    """把還沒轉成案例的回報變成測試案例。

    回報只有熱量區間（使用者填的那兩個數字），沒有份量與佐證的判準——
    那些要人判斷之後才加得上。所以回報來的案例判準比較少，這是預期的。
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(FeedbackReport)
            .filter(
                FeedbackReport.user_id == user_id,
                FeedbackReport.converted_to_eval.is_(False),
                FeedbackReport.source == "food_log",
            )
            .order_by(FeedbackReport.id)
            .all()
        )
        return [
            {
                "id": f"report_{r.id}",
                "input": r.raw_input,
                "origin": f"回報 #{r.id}（{r.created_at:%m/%d}）",
                "why": r.note or "",
                "calories_min": r.expected_min,
                "calories_max": r.expected_max,
                "report_id": r.id,
            }
            for r in rows
            # 沒有區間的回報跑了也不知道對不對，直接跳過並在輸出說明。
            if r.expected_min is not None or r.expected_max is not None
        ]
    finally:
        db.close()


def promote(report_ids: list[int]) -> None:
    """把回報標記成已轉入案例。

    刻意是**獨立的動作**而不是跑完自動標記：跑一次只是看結果，
    決定要長期守住哪些是人的判斷。自動標記的話，跑一次就再也撈不到那筆回報。
    """
    db = SessionLocal()
    try:
        for rid in report_ids:
            row = db.get(FeedbackReport, rid)
            if row is not None:
                row.converted_to_eval = True
        db.commit()
    finally:
        db.close()


def run_case(case: dict, runs: int) -> dict:
    """跑一個案例 N 次。

    N 次是因為擷取是機率性的：同一個輸入兩次執行可能給不同份量
    （實測泡麵拿過 90、120、200、300 公克）。跑一次看對錯，
    跑 N 次看**通過率**——對機率性系統，後者才是有意義的量測。
    """
    attempts = []
    for _ in range(runs):
        try:
            extraction = extract_food_log(case["input"])
        except Exception as e:
            attempts.append(
                {
                    "crashed": f"{type(e).__name__}: {e}",
                    "calories": None,
                    "checks": [],
                    "passed": False,
                }
            )
            continue
        checks = check_extraction(case, extraction)
        strict = [c for c in checks if c.severity == "fail"]
        attempts.append(
            {
                "crashed": None,
                "calories": extraction.calories,
                "grounded": f"{extraction.grounded_count}/{len(extraction.items)}",
                "items": [
                    {
                        "name": i.name,
                        "portion_g": i.portion_g,
                        "calories": i.calories,
                        "grounded": i.grounded,
                        "db_name": i.db_name,
                    }
                    for i in extraction.items
                ],
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail, "severity": c.severity}
                    for c in checks
                ],
                "passed": bool(strict) and all(c.passed for c in strict),
            }
        )

    passed = sum(1 for a in attempts if a["passed"])
    cals = [a["calories"] for a in attempts if a["calories"] is not None]
    return {
        "id": case["id"],
        "input": case["input"],
        "origin": case.get("origin", ""),
        "runs": runs,
        "passed_runs": passed,
        "pass_rate": passed / runs if runs else 0.0,
        "calories_mean": mean(cals) if cals else None,
        "calories_spread": (max(cals) - min(cals)) if len(cals) > 1 else 0.0,
        "attempts": attempts,
    }


def _print_case(res: dict) -> None:
    rate = res["pass_rate"]
    mark = "PASS" if rate == 1.0 else ("FLAKY" if rate > 0 else "FAIL")
    print(f"\n[{mark}] {res['id']}　{res['origin']}")
    print(f"  輸入：{res['input']}")
    if res["runs"] > 1:
        print(
            f"  通過 {res['passed_runs']}/{res['runs']} 次"
            f"　熱量平均 {res['calories_mean']:.0f}"
            f"　同輸入最大落差 {res['calories_spread']:.0f} 大卡"
        )

    last = res["attempts"][-1]
    if last["crashed"]:
        print(f"  ❌ 拋例外：{last['crashed']}")
        return
    print(f"  熱量 {last['calories']:.0f}　佐證 {last['grounded']}")
    for item in last["items"]:
        src = f"✓ {item['db_name']}" if item["grounded"] else "~ 估算"
        print(f"      {item['name']} {item['portion_g']:.0f}g → {item['calories']:.0f}　{src}")
    for c in last["checks"]:
        if not c["passed"]:
            tag = "FAIL" if c["severity"] == "fail" else "WARN"
            print(f"      [{tag}] {c['name']}：{c['detail']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="記錄側 eval")
    ap.add_argument("--reports", action="store_true", help="一併跑資料庫裡待處理的回報")
    ap.add_argument("--runs", type=int, default=1, help="每個案例跑幾次（看穩定度）")
    ap.add_argument("--promote", action="store_true", help="跑完把回報標記成已轉入案例")
    ap.add_argument("--baseline", type=str, default=None, help="跟這份結果比較")
    args = ap.parse_args()

    cases = load_cases()
    report_cases = load_report_cases() if args.reports else []
    if args.reports:
        print(f"從資料庫帶入 {len(report_cases)} 筆待處理回報（有填區間的）")
    cases += report_cases

    if not cases:
        print("沒有任何案例。先寫 evals/extraction_dataset.jsonl 或在前端回報幾筆。")
        return

    print(f"案例數：{len(cases)}　每個跑 {args.runs} 次")
    print("這個 eval 會呼叫 LLM，每個案例每次約 2~3 次呼叫。\n" + "=" * 70)

    results = [run_case(c, args.runs) for c in cases]
    for res in results:
        _print_case(res)

    print("\n" + "=" * 70)
    full = sum(1 for r in results if r["pass_rate"] == 1.0)
    flaky = sum(1 for r in results if 0 < r["pass_rate"] < 1.0)
    dead = sum(1 for r in results if r["pass_rate"] == 0.0)
    print(f"全過 {full}　不穩定 {flaky}　全不過 {dead}　／共 {len(results)} 個案例")
    if args.runs > 1:
        worst = max(results, key=lambda r: r["calories_spread"])
        print(
            f"同輸入落差最大：{worst['id']}　{worst['calories_spread']:.0f} 大卡"
            "（擷取是機率性的，落差大表示那個案例的份量判斷不穩）"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"extraction_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(
        json.dumps({"runs": args.runs, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n結果：{out}")

    if args.baseline:
        _compare(Path(args.baseline), results)

    if args.promote and report_cases:
        ids = [c["report_id"] for c in report_cases]
        promote(ids)
        print(f"已標記 {len(ids)} 筆回報為已轉入案例（下次 --reports 不會再帶入）")
    elif report_cases:
        print("回報尚未標記為已轉入。確認要長期守住的話加 --promote。")


def _compare(baseline_path: Path, results: list[dict]) -> None:
    """跟上一次的結果比。

    這是整個 eval 最常用的功能：**改完之後有沒有弄壞別的。**
    單看通過率只知道現在的狀態，比較才看得出方向。
    """
    if not baseline_path.exists():
        print(f"\n找不到基準檔：{baseline_path}")
        return
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    old = {r["id"]: r for r in baseline["results"]}

    # 單次執行的比較會報假回歸。
    #
    # 2026-11-05 實測：改完 prompt 之後比較結果說 composite_bento 從 100% 掉到 0%，
    # 看起來是我改壞了。跑三次才發現同一個輸入的熱量是 350／500／825——
    # 落差 475 大卡，**它本來就不穩**。基準那次剛好抽到高的，修完那次剛好抽到低的。
    #
    # 對機率性系統，單次比較的「變化」有一半是抽樣噪音。
    min_runs = min(baseline.get("runs", 1), max((r["runs"] for r in results), default=1))
    if min_runs < 2:
        print(
            "\n⚠️ 兩邊都只跑一次，以下的「變化」可能是雜訊而不是回歸。"
            "要判斷改動的影響請用 --runs 3 以上重跑兩邊。"
        )

    better, worse = [], []
    for r in results:
        prev = old.get(r["id"])
        if prev is None:
            continue
        if r["pass_rate"] > prev["pass_rate"]:
            better.append((r["id"], prev["pass_rate"], r["pass_rate"]))
        elif r["pass_rate"] < prev["pass_rate"]:
            worse.append((r["id"], prev["pass_rate"], r["pass_rate"]))

    print(f"\n跟 {baseline_path.name} 比較：")
    for cid, a, b in better:
        print(f"  ↑ {cid}　{a:.0%} → {b:.0%}")
    for cid, a, b in worse:
        print(f"  ↓ {cid}　{a:.0%} → {b:.0%}　← 這個被改壞了")
    if not better and not worse:
        print("  沒有變化")


if __name__ == "__main__":
    main()
