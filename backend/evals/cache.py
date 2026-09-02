"""Judge 判定的本地快取。

Judge 的輸出是這幾樣東西的函數：
    (模型, judge system prompt, 問題, 答案, 兩份 context, 判準)
全部沒變就沒有理由重算——但校準會反覆執行（改一條判準就要重跑整批），
不快取的話同樣的呼叫會付好幾次錢。

快取 key 把上面每一項都算進雜湊，所以 **prompt 一改 key 就變、快取自動失效**，
不需要手動清。這一點很重要：手動管理失效遲早會忘記，
然後你會拿舊 prompt 的判定去解讀新 prompt 的表現。

快取檔進 git。它是可重生的（刪掉只是要重花一次錢），但留著能讓
其他人 clone 之後重現校準數字而不必付費。
"""

import hashlib
import json
from pathlib import Path

CACHE_PATH = Path(__file__).parent / "judge_cache.json"


def make_key(**parts: str) -> str:
    """把所有影響判定的輸入雜湊成一個 key。

    用排序過的 JSON 而不是直接串接字串，避免「欄位邊界」造成的碰撞——
    ("ab", "c") 和 ("a", "bc") 直接串接會得到同一個字串。
    """
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeCache:
    """讀寫 judge 判定的簡易快取。

    整份載入記憶體、每次寫入就存檔。資料量在幾千筆以內都沒問題，
    真的長到需要 SQLite 再說——現在用 JSON 的好處是可以直接打開來看。
    """

    def __init__(self, path: Path = CACHE_PATH) -> None:
        self.path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict | None:
        value = self._data.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def set(self, key: str, value: dict) -> None:
        self._data[key] = value
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total if total else 0
        return f"快取命中 {self.hits}/{total}（{rate:.0%}），實際呼叫 {self.misses} 次"
