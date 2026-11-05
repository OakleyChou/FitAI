import logging

import instructor
from langfuse.openai import openai
from pydantic import BaseModel
from langfuse.decorators import langfuse_context, observe

from app.config import settings
from app.database import SessionLocal
from app.extraction import (
    FoodItem,  # noqa: F401  其他模組從這裡 import，保留相容
    FoodLogExtraction,
    ItemMatch,
    ParsedItem,
    build_extraction,
)
from app.models import PromptVersion
from app.rag.retrieval import search

logger = logging.getLogger(__name__)

# 用 langfuse.openai 包過的 openai client，取代原生 openai.OpenAI
# LLM 呼叫會自動被記錄成 Generation 類型的 observation，包含 token 用量/成本
# 也可以在 langfuse 上更新 prompt，不用進資料庫更新
client = instructor.from_openai(openai.OpenAI(api_key=settings.openai_api_key))

FOOD_EXTRACTION_PROMPT_NAME = "food_extraction_system_prompt"

# 兩階段擷取各有自己的 prompt。分開存而不是塞成一份，是為了能各自換版——
# 份量換算規則調整不該讓配對判準的版本號也跳動。
FOOD_PARSE_PROMPT_NAME = "food_parse_system_prompt"
FOOD_MATCH_PROMPT_NAME = "food_match_system_prompt"


def get_active_prompt(name: str) -> str:
    """查詢資料庫裡這個 name 底下版本號最大的 prompt 內容。

    目前用 version 決定要用哪一版（永遠取最新版）；is_active 欄位保留在資料表裡，
    之後如果需要「不換版本號也能切換/回滾」的彈性，可以再改回用它篩選。
    """
    # 因為不在 FastAPI route裡，需要手動另外開session
    db = SessionLocal()
    try:
        prompt = (
            db.query(PromptVersion)
            .filter(PromptVersion.name == name)
            .order_by(PromptVersion.version.desc())
            .first()
        )
        if prompt is None:
            raise ValueError(f"找不到 name={name!r} 的 prompt version")
        return prompt.content
    finally:
        db.close()


def add_prompt_version(name: str, content: str) -> PromptVersion:
    """新增一個 prompt 版本—
    以這個 name 的目前最大版本號+1」當作新版本號
    """
    db = SessionLocal()
    try:
        latest = (
            db.query(PromptVersion)
            .filter(PromptVersion.name == name)
            .order_by(PromptVersion.version.desc())
            .first()
        )
        next_version = (latest.version + 1) if latest is not None else 1

        new_prompt = PromptVersion(
            name=name,
            version=next_version,
            content=content,
            is_active=True,
        )
        db.add(new_prompt)
        db.commit()
        db.refresh(new_prompt)
        return new_prompt
    finally:
        db.close()


CANDIDATES_PER_ITEM = 5


class _ParsedMeal(BaseModel):
    """第一階段的 response_model。包一層是因為 instructor 對 list[T] 的
    支援不如單一 model 穩定，而且多一個欄位可以放模型的整體判斷。"""

    items: list[ParsedItem]


class _MatchResult(BaseModel):
    matches: list[ItemMatch]


def _parse_items(raw_text: str) -> list[ParsedItem]:
    """第一階段：LLM 只辨識項目與份量，不估營養素。"""
    result = client.chat.completions.create(
        model=settings.openai_model_cheap,
        response_model=_ParsedMeal,
        messages=[
            {"role": "system", "content": get_active_prompt(FOOD_PARSE_PROMPT_NAME)},
            {"role": "user", "content": raw_text},
        ],
    )
    return result.items


def _retrieve_candidates(items: list[ParsedItem]) -> dict[str, dict]:
    """為每一項檢索候選，回傳 source_ref → meta。

    檢索失敗時回空 dict 而不是拋例外：那樣會退化成「全部由 LLM 估算」，
    也就是改版前的行為——比整個記錄功能壞掉好。
    grounded 旗標仍然誠實標示那些數值沒有資料庫佐證。
    """
    rows: dict[str, dict] = {}
    for item in items:
        try:
            chunks = search(item.name, top_k=CANDIDATES_PER_ITEM)
        except Exception:
            logger.exception("檢索候選失敗，這一項將退回 LLM 估算：%s", item.name)
            continue
        for chunk in chunks:
            if chunk.source_ref:
                rows[chunk.source_ref] = chunk.meta
    return rows


def _format_candidates(items: list[ParsedItem], db_rows: dict[str, dict]) -> str:
    """把候選整理成給模型看的文字。

    每一項重新檢索一次而不是共用上面的結果，是為了讓候選跟項目對應清楚——
    db_rows 是扁平的 source_ref 字典，看不出哪個候選屬於哪一項。
    """
    lines = []
    for i, item in enumerate(items):
        lines.append(f"[項目 {i}] {item.name}（份量 {item.portion_g:.0f} 公克/毫升）")
        try:
            chunks = search(item.name, top_k=CANDIDATES_PER_ITEM)
        except Exception:
            lines.append("  （檢索失敗，無候選）")
            continue
        if not chunks:
            lines.append("  （無候選）")
        for chunk in chunks:
            m = chunk.meta
            if not chunk.source_ref:
                continue
            db_rows[chunk.source_ref] = m
            # **刻意不給營養素數字。**
            #
            # 判斷「這是不是同一種食物」只需要名稱、分類與配料描述。
            # 給了數字反而害事：模型會把候選的「每 100 公克」值直接抄進
            # 它該填的「整項估算」欄位——實際踩到的例子是「雞胸肉便當 500 公克」
            # 被填上 45 大卡（那是雞胸骨湯每 100 公克的熱量）。
            # 被名稱檢查擋掉之後，那 45 大卡就成了整份便當的數值。
            #
            # 通則：**只給模型做這個決定所需要的資訊。** 多給的欄位它會想辦法用上。
            lines.append(
                f"  - {m.get('name')}｜source_ref={chunk.source_ref}｜"
                f"分類：{m.get('category') or '未分類'}"
            )
            # 配料描述是判斷「是不是同一種食物」的關鍵，不能只給名稱。
            #
            # 實際踩到的例子：使用者說「一個漢堡」，候選有「漢堡包」。
            # 名稱幾乎一樣，模型判定相符，然後用整個漢堡的重量（200g）
            # 去乘麵包的熱量密度，得到 644 大卡——**比讓它自己估更錯**。
            # 而「漢堡包」的描述寫著「麵粉,糖,奶油,酵母,鹽,奶粉,酥油,蛋,白芝麻」，
            # 沒有肉。給模型看描述，它就有機會看出這只是麵包。
            desc = _ingredient_hint(chunk.content)
            if desc:
                lines.append(f"      說明：{desc}")
    return "\n".join(lines)


def _ingredient_hint(content: str, limit: int = 120) -> str:
    """從 chunk 的敘述裡取出「這是什麼」的部分，丟掉營養素數字。

    營養素已經在上面以數字形式給過了，重複餵只是稀釋 prompt。
    真正有判斷價值的是樣品狀態與前處理描述（生/乾、有沒有肉、怎麼處理的）。
    """
    head = content.split("，每 100 公克")[0]
    # 名稱與分類已經單獨列出，這裡只要句號之後的描述。
    parts = head.split("。", 1)
    desc = parts[1] if len(parts) > 1 else ""
    return desc.strip()[:limit]


def _match_items(raw_text: str, items: list[ParsedItem], candidates: str) -> list[ItemMatch]:
    """第二階段：判斷每一項在資料庫裡有沒有對應。"""
    result = client.chat.completions.create(
        model=settings.openai_model_cheap,
        response_model=_MatchResult,
        messages=[
            {"role": "system", "content": get_active_prompt(FOOD_MATCH_PROMPT_NAME)},
            {
                "role": "user",
                "content": f"使用者的描述：{raw_text}\n\n候選清單：\n{candidates}",
            },
        ],
    )
    return result.matches


@observe()
def extract_food_log(raw_text: str) -> FoodLogExtraction:
    """把自然語言飲食描述轉成結構化的 FoodLogExtraction。

    ## 為什麼分兩階段

    舊版讓 LLM 同時負責「這是什麼」與「有多少熱量」，而後者它做不好：
    問「一杯無糖鮮奶茶」它回 50 大卡，而資料庫裡「鮮奶茶(無糖)」是
    27 大卡/100g，一杯 450 毫升應該是 121 大卡——**低估 2.4 倍，而且不會報錯**。
    使用者只會以為自己喝的東西很健康，每日合計就這樣系統性偏低。

    現在 LLM 只做語言（辨識項目、估份量），數字盡量來自食藥署資料庫。
    這跟問答側「所有具體營養數值都必須來自工具回傳的結果」是同一個標準——
    兩條路徑不該有不同的可靠性要求。

    ## 代價

    一次記錄從 1 次 LLM 呼叫變成 2 次 + N 次 embedding，大約慢 2～3 秒。
    換到的是可回溯的數值：每一項都有 source_ref，或明確標示為估算。

    ## 資料庫查不到的那些

    食藥署資料庫收一般食材與少數市售飲品，**不收複合餐點**。
    「鮪魚蛋吐司」「雞胸肉便當」查不到，這些項目的 `grounded` 會是 False，
    數值仍由 LLM 估算。那是誠實的降級，不是失敗——
    重點是使用者看得出哪些數字有依據、哪些沒有。
    """
    parsed = _parse_items(raw_text)
    if not parsed:
        # 不是食物的輸入。回空結果而不是拋例外——那是使用者打錯字，不是系統錯誤。
        return build_extraction(raw_text, [], [], {})

    db_rows: dict[str, dict] = {}
    candidates = _format_candidates(parsed, db_rows)
    matches = _match_items(raw_text, parsed, candidates)

    return build_extraction(raw_text, parsed, matches, db_rows)


if __name__ == "__main__":
    for text in ["一杯無糖鮮奶茶", "我晚上吃了一個漢堡加一杯可樂", "鮪魚蛋吐司"]:
        result = extract_food_log(text)
        print(f"\n{text} → {result.calories:.0f} 大卡（{result.grounded_count}/{len(result.items)} 項有佐證）")
        for item in result.items:
            mark = f"✓ {item.db_name} [{item.source_ref}]" if item.grounded else "~ LLM 估算"
            print(f"   {item.name} {item.portion_g:.0f}g → {item.calories:.0f} 大卡  {mark}")
    langfuse_context.flush()  # 強制把排隊中的 trace 資料立刻送出，不等背景批次處理
