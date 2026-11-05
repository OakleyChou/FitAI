"""跑 eval 資料集：重置資料 → 執行 agent → 套 L1 自動檢查 → 產出待人工標註的 CSV。

跑法（在 backend/ 底下）：
    poetry run python -m evals.run_eval

## 評估對象從一個變成三個

固定流程的 eval 只看答案。換成 agent 之後多了兩層：

    答案   回答對不對          ← 需要人工或 judge 判斷
    路徑   有沒有選對工具      ← 程式就能檢查，成本為零
    效率   幾輪、幾次呼叫      ← 同上

**路徑必須單獨評估**，因為「選錯工具但答案通順」只看答案完全看不出來。
實例：問「我吃過什麼蛋白質密度最高的食物」，agent 呼叫 rank_by_nutrient
（查的是公開資料庫，範圍錯了）回答柴魚片——兩次呼叫都成功、答案也通順，
只有看路徑才發現它答錯了問題。

## 路徑判準寫寬鬆，不要寫死

不要求「路徑必須完全等於 [A, B]」——合理的路徑常常不只一種，而且模型的
非決定性會讓同一題這次一步、下次兩步。改用四種約束（見 _check_path）：
必要工具、禁用工具、順序、次數上限。

## L1 / L2 分層

L1 自動斷言確定性、幾乎零成本，能用它解決的絕不用 LLM judge。
L2 人工標註處理需要判斷力的部分，那批標註之後會成為 judge 的校準基準。
"""

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.agent.loop import AgentResult, run_agent
from scripts.seed_dev_data import seed

EVALS_DIR = Path(__file__).parent
DATASET_PATH = EVALS_DIR / "dataset.jsonl"
RESULTS_DIR = EVALS_DIR / "results"

# 跟 loop.py 的 prompt 保持一致。從 200 放寬到 300 是因為 agent 的工具會
# 回傳結構化資料，列幾項食物加上必要說明就會接近這個長度；同時在 prompt
# 裡加了「最多列 3 項」，限制項數比限制字數更容易被遵守（數項目比數字容易）。
# 2026-09-22 從 300 調到 350：prompt 新增了「引用標記」與「計量基準」兩項要求，
# 每列食物因此多約 40 字，三項就是 120 字。**加了要求就要給預算**——
# 不調的話等於因為 agent 照做而扣它分，那條規則量到的就不再是簡潔度了。
MAX_ANSWER_CHARS = 350

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# 只檢查緊接著營養單位的數字。這條規則要防的是編造營養數據，
# 不是份量假設——「一杯 300ml」是 prompt 明文要求模型自行估算的。
NUTRIENT_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:大卡|kcal|公克|克|毫克|[gG]\b|mg)")

# 食藥署整合編號：一個大寫字母加 5~7 位數字（A05001、I0410101）。
#
# 刻意不要求方括號緊貼編號。第一版寫成 r"\[([A-Z]\d{5,7})\]"，
# 結果模型寫「[參考編號 J0800401]」就被判定成「沒有引用標記」——
# 它其實引用了，只是多加了兩個字。L1 要檢查的是**編號在不在、對不對**，
# 不是括號長什麼樣；把格式潔癖寫進規則只會製造假警報。
CITATION_RE = re.compile(r"\b([A-Z]\d{5,7})\b")

# 會回傳營養資料庫內容的工具。只有這些的結果需要附來源編號——
# 使用者自己的飲食紀錄是他自己輸入的，沒有外部出處可引。
KNOWLEDGE_TOOLS = {"search_nutrition", "rank_by_nutrient"}

# 計量基準：「35 大卡」單獨出現是沒有意義的，要講清楚是多少份量的 35 大卡。
# 資料庫一律以每 100 公克計，但答案換算成常見份量（一杯、一份、一顆）也可以，
# 只要有寫。涵蓋全形與半形、公克與毫升、以及常見的份量單位。
MEASURE_BASIS_RE = re.compile(
    r"每\s*\d+\s*(?:公克|克|毫升|ml|ML|g|G)"
    r"|每(?:100)?\s*(?:公克|克|毫升)"
    r"|一(?:杯|份|顆|片|碗|匙|根|條|塊)"
    r"|\d+\s*(?:ml|ML|毫升|cc|CC)"
)

# 允許模型做份量換算時用的倍率。沒有這些，「35 大卡 × 3 杯 = 105」
# 會被誤判成編造。
SCALE_FACTORS = [0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 10]

DERIVE_TOLERANCE = 0.55


@dataclass
class CheckResult:
    """一條 L1 檢查的結果。

    severity 區分兩種規則：
    - "fail"：確定性判斷，或已用人工標註驗證過精準度的規則。
    - "warn"：還在試驗階段的啟發式規則。值得看一眼，但不列入失敗統計。

    分級是為了避免「狼來了」——會製造假警報的規則若和確定性規則混在同一個
    FAIL 池裡，久了所有警告都會被忽略。
    """

    name: str
    passed: bool
    detail: str = ""
    severity: str = "fail"


def _has_simplified_chinese(text: str) -> bool:
    """查每個中文字的 Unicode 名稱有沒有 SIMPLIFIED 標記。

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

    模型的合理推導會產生工具結果裡沒有的數字，例如
        105 = 35 大卡 × 3 杯      （份量換算）
        32  = 96 目標 − 64 攝取   （差距計算）
    只比對字面數字的話，這兩個都會被誤判成編造。

    只做一步是刻意的：兩步以上的組合數會爆炸，而且幾乎任何數字都能被湊出來，
    檢查就失去意義。寧可漏抓，不要放行過寬。
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


def _check_path(example: dict, result: AgentResult) -> list[CheckResult]:
    """路徑判準：檢查 agent 呼叫了什麼，而不只是它回答了什麼。

    四種約束，全部是字串比對或計數，不需要 LLM judge：

    required_tools  這題必須用到的工具（「高蛋白低卡」必須走數值排序）
    forbidden_tools 這題不該用的工具（純知識問題不該查個人紀錄）
    order           前者必須出現在後者之前（多跳題的依賴關係）
    max_calls       呼叫次數上限（防止繞路與浪費）

    沒有寫 path 的 example 就跳過——不是每一題都值得定義路徑。
    """
    spec = example.get("path")
    if not spec:
        return []

    names = [c.name for c in result.tool_calls]
    checks: list[CheckResult] = []

    for tool in spec.get("required_tools", []):
        checks.append(
            CheckResult(
                f"必須呼叫 {tool}",
                tool in names,
                f"實際路徑：{' → '.join(names) or '（未呼叫工具）'}",
            )
        )

    for tool in spec.get("forbidden_tools", []):
        checks.append(
            CheckResult(
                f"不應呼叫 {tool}",
                tool not in names,
                f"實際路徑：{' → '.join(names) or '（未呼叫工具）'}",
            )
        )

    order = spec.get("order", [])
    if len(order) >= 2 and all(t in names for t in order):
        positions = [names.index(t) for t in order]
        checks.append(
            CheckResult(
                f"順序 {' → '.join(order)}",
                positions == sorted(positions),
                f"實際位置：{positions}",
            )
        )

    if "max_calls" in spec:
        limit = spec["max_calls"]
        checks.append(
            CheckResult(f"呼叫次數 ≤ {limit}", len(names) <= limit, f"{len(names)} 次")
        )

    return checks


def run_l1_checks(example: dict, result: AgentResult) -> list[CheckResult]:
    """確定性檢查，每一條都來自實際觀察到的失敗。

    每個 bug 都變成一條測試——這是 eval 資料集長大的方式，
    不是一開始就設計好一套完美指標。
    """
    answer = result.answer
    checks: list[CheckResult] = []

    # 實際發生過：「我將乐意協助你」
    checks.append(CheckResult("繁體中文", not _has_simplified_chinese(answer)))

    checks.append(
        CheckResult(f"長度≤{MAX_ANSWER_CHARS}", len(answer) <= MAX_ANSWER_CHARS, f"{len(answer)} 字")
    )

    checks.append(CheckResult("不說「使用者」", "使用者" not in answer))

    # 迴圈有沒有正常收斂。hit_limit 表示模型用完次數還沒給出答案，
    # 使用者拿到的是罐頭訊息而不是回答。
    checks.append(CheckResult("迴圈正常結束", not result.hit_limit))

    # 系統層錯誤。有值表示有東西壞了，即使使用者看到的訊息還算得體。
    checks.append(
        CheckResult("無系統錯誤", result.error is None, result.error or "")
    )

    # 每一個工具呼叫都失敗時，那個「答案」不可能是真的答案。
    #
    # 2026-11-05 實測：模型把參數名打成 "n nutrient"，兩次呼叫都 TypeError，
    # 然後回「看起來目前工具無法正確處理該請求」。而路徑判準全部通過——
    # required_tools 有呼叫到、forbidden_tools 沒踩到、max_calls 也沒超。
    #
    # **路徑對不代表答案對。** 這是「答案對但路徑錯」的鏡像，而先前只防了前者。
    # 用「有沒有任何一次工具呼叫成功」當判準，是確定性的，不需要讀答案內容。
    if result.tool_calls:
        checks.append(
            CheckResult(
                "至少一個工具成功",
                any(c.error is None for c in result.tool_calls),
                "全部工具呼叫都失敗了" if all(c.error for c in result.tool_calls) else "",
            )
        )

    checks.extend(_check_path(example, result))

    # Groundedness 的廉價近似：答案裡的營養數值要能在工具結果中找到，
    # 或能由工具結果的數字一步算出。
    #
    # 列為 warn 而非 fail：它本質是啟發式，還沒有足夠的人工標註證明它可信。
    # 累積更多樣本、確認精準度後再考慮升級成 fail。
    #
    # 這裡用 ToolCall.result（完整）而不是 result_preview（截短）——
    # 比對模型實際看到的全部內容，截短過的會製造大量假警報。
    tool_output = "\n".join(c.result for c in result.tool_calls)

    # 引用標記。dataset 有三條 rubric 在考這件事，2026-09-22 之前
    # SYSTEM_PROMPT 卻從來沒要求過——三條永遠 fail，而 L1 一條都沒抓到。
    # 規格對齊之後，這是格式問題，正則又快又準又免費，不該佔用 judge。
    used = {c.name for c in result.tool_calls if not c.unavailable}
    cited = set(CITATION_RE.findall(answer))
    available = set(CITATION_RE.findall(tool_output))

    if used & KNOWLEDGE_TOOLS and available:
        checks.append(CheckResult("有引用標記", bool(cited)))

        # 「豆漿約 35 大卡」單獨看是沒有意義的——35 大卡的是一口還是一公升？
        # 資料庫的數字一律是每 100 公克，答案不寫基準等於把單位吞掉了。
        # 只在引用營養資料庫時檢查：個人紀錄講的是「那一餐」，本來就沒有基準可言。
        checks.append(
            CheckResult("數值有計量基準", bool(MEASURE_BASIS_RE.search(answer)))
        )

    # 編出來的編號比沒有編號更糟：它看起來可稽核，實際上查不到。
    fabricated = sorted(cited - available)
    checks.append(
        CheckResult(
            "引用標記真實存在",
            not fabricated,
            f"工具結果中沒有：{', '.join(fabricated)}" if fabricated else "",
        )
    )

    pool = [float(n) for n in NUMBER_RE.findall(tool_output)]
    claimed = {float(n) for n in NUTRIENT_VALUE_RE.findall(answer)}
    ungrounded = sorted(n for n in claimed if not _is_derivable(n, pool))
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
    # 先把測試資料重置成已知狀態。沒有這一步 eval 就不可重現——
    # seed 的日期是相對於執行當下算的，隔天再跑「今天」就沒有紀錄了，
    # 而你會誤以為是 prompt 改壞了。
    seeded = seed()
    print(f"已重置測試資料：{seeded} 筆 food_logs\n")

    examples = load_dataset()
    rows: list[dict] = []
    total_strict = passed_strict = warn_count = 0

    for example in examples:
        result = run_agent(example["question"], user_id=example["user_id"])
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

        path = " → ".join(c.name for c in result.tool_calls) or "（未呼叫工具）"

        print("=" * 78)
        print(f"[{example['id']}]  type={example['type']}")
        print(f"Q: {example['question']}")
        print(f"路徑（{result.iterations} 輪）: {path}")
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
        #
        # tool_results 存完整結果是給 judge 用的：judge 要判斷「數值對不對」，
        # 就必須看到 agent 當時看到的東西。路徑不可重現（模型的選擇會變），
        # 所以必須歸檔，不能事後重算。
        tool_results = "\n\n".join(
            f"[{c.name}({json.dumps(c.arguments, ensure_ascii=False)})]\n{c.result}"
            for c in result.tool_calls
        )
        for item in example["rubric"]:
            rows.append(
                {
                    "example_id": example["id"],
                    "type": example["type"],
                    "question": example["question"],
                    "answer": result.answer,
                    "tool_path": path,
                    "tool_results": tool_results,
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
