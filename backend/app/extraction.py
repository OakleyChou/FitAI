"""把解析結果與資料庫比對結果組合成最終營養素。純函式，不呼叫 LLM、不碰資料庫。

## 為什麼這一層要獨立出來

`extract_food_log` 的流程是「LLM → 檢索 → LLM → 算數字」。前三步都有外部依賴，
最後一步是純計算——而**錯最容易發生在最後一步**（份量換算、單位、加總）。

把它拆出來之後，那些計算可以不花錢、不開資料庫就測到：
份量 450 毫升乘以每 100 公克 27 大卡該得到 121.5，這件事不需要 LLM 來驗證。

## 為什麼要分「有佐證」與「無佐證」

食藥署資料庫收一般食材與少數市售飲品，**不收複合餐點**。
查「鮪魚蛋吐司」會撈回「鮪魚肚」（106 大卡/100g）——名稱像、數值看起來合理、
但它量的是另一種食物，而且沒有人會發現。

所以每一項都要記錄它的數字是從哪裡來的：

    grounded=True   數值來自資料庫，有 source_ref 可回溯
    grounded=False  資料庫沒有對應項目，數值是 LLM 估的

這個旗標要一路傳到畫面上。使用者看到「這一項是估算的」才知道哪裡需要核對——
全部混在一起顯示的話，可靠的數字和猜測看起來一模一樣。
"""

import re

from pydantic import BaseModel, Field

# 資料庫的營養素一律以每 100 公克為基準（見 app/rag/ingest.py）。
GRAMS_PER_BASIS = 100.0

NUTRIENT_FIELDS = ("calories", "protein_g", "carbs_g", "fat_g")


class ParsedItem(BaseModel):
    """第一階段的輸出：LLM 只辨識項目與份量，不估營養素。"""

    name: str = Field(description="食物本身的名稱，不要帶份量數字")
    portion_g: float = Field(gt=0, description="份量，公克（液體用毫升）")
    portion_note: str = Field(default="", description="份量怎麼換算的，給人看")


class ItemMatch(BaseModel):
    """第二階段的輸出：這一項在資料庫裡有沒有對應。"""

    index: int = Field(description="對應 ParsedItem 在清單中的位置，從 0 開始")
    source_ref: str | None = Field(
        default=None, description="相符的資料庫項目整合編號；不相符則為 null"
    )
    reason: str = Field(default="", description="為什麼相符或不相符，一句話")
    # 以下只在 source_ref 為 null 時填，是**整項**的量而非每 100 公克。
    calories: float | None = None
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None


class FoodItem(BaseModel):
    """一個食物項目的最終數值。

    `grounded` 區分「數值來自食藥署資料庫」與「LLM 估的」。
    這個欄位要顯示給使用者——不標的話，可靠的數字和猜測看起來一樣。
    """

    name: str
    portion_g: float
    calories: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)
    grounded: bool = False
    source_ref: str | None = None
    # 資料庫裡的正式名稱。跟使用者的說法常常不同（「無糖鮮奶茶」vs「鮮奶茶(無糖)」），
    # 顯示出來使用者才能判斷配對對不對。
    db_name: str | None = None
    note: str = ""


class FoodLogExtraction(BaseModel):
    """最終結果。合計由程式加總，不讓 LLM 自己加。"""

    items: list[FoodItem]
    food_description: str
    calories: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)

    @property
    def grounded_count(self) -> int:
        return sum(1 for item in self.items if item.grounded)

    @property
    def all_grounded(self) -> bool:
        """全部項目都有資料庫佐證。空清單視為 False——沒有項目談不上佐證。"""
        return bool(self.items) and self.grounded_count == len(self.items)


# 名稱相似度的門檻。低於這個值就拒絕採用資料庫數值，退回 LLM 估算。
#
# 這是**啟發式**，用實際踩到的五個例子校準出來的（分數是雙向覆蓋率的較小值）：
#
#     無糖鮮奶茶  ↔ 鮮奶茶(無糖)        0.60  ✓ 該接受（只是詞序與修飾語不同）
#     可樂        ↔ 可樂                1.00  ✓ 該接受
#     鮪魚蛋吐司  ↔ 鮪魚肚              0.40  ✗ 該拒絕（複合餐點配到單一食材）
#     漢堡        ↔ 冷凍牛肉漢堡肉      0.33  ✗ 該拒絕（整個漢堡配到肉排）
#     一碗滷肉飯  ↔ 臘肉(五花肉)        0.33  ✗ 該拒絕（整碗飯配到五花肉）
#
# 0.5 把這五個分乾淨，而且留了邊界。樣本只有五個，之後遇到新的錯誤案例要重新校準。
NAME_MATCH_THRESHOLD = 0.5

# 括號內通常是修飾語（無糖、生、乾、7月取樣），不是食物本身的識別資訊。
_PARENTHETICAL = re.compile(r"[(（][^)）]*[)）]")
_PUNCTUATION = re.compile(r"[\s,，、。:：/()（）＋+－-]")


def _significant_chars(name: str) -> set[str]:
    return set(_PUNCTUATION.sub("", _PARENTHETICAL.sub("", name)))


def name_matches(user_name: str, db_name: str, threshold: float = NAME_MATCH_THRESHOLD) -> bool:
    """使用者說的食物名稱，跟資料庫項目的名稱是不是指同一種東西。

    ## 為什麼需要這道程式層的防線

    配對交給 LLM 判斷，但它會犯一種**系統性的乘法錯誤**：
    把複合餐點配到其中一種高密度食材，然後套用整份餐點的重量。

        「一碗滷肉飯」→ 配到「臘肉(五花肉)」(525 大卡/100g) × 500g = 2623 大卡
                       真實約 600 大卡，錯了 4 倍

    prompt 寫了「寧可判不相符」也擋不住——模型看到兩者都是滷的豬肉就配了。
    而這個錯誤**不會報錯**，使用者只會看到一個離譜但說不上哪裡怪的數字。

    ## 這道防線只會變保守，不會憑空產生佐證

    它唯一的作用是**拒絕**可疑的配對，讓那一項退回 LLM 估算並標示未佐證。
    誤拒的代價是少一個有來源的數值（回到改版前的行為），
    誤收的代價是一個錯 4 倍、看起來卻很權威的數字。兩者不對稱，所以寧可誤拒。

    ## 怎麼算

    取兩邊的「有意義字元集合」（去掉括號修飾語與標點），算交集對各自的覆蓋率，
    取**較小的那個**。單向覆蓋率不夠：「漢堡」的每個字都在「冷凍牛肉漢堡肉」裡，
    但後者多出「冷凍牛肉」「肉」，那表示它是另一種東西。
    """
    a, b = _significant_chars(user_name), _significant_chars(db_name)
    if not a or not b:
        return False
    overlap = len(a & b)
    return min(overlap / len(a), overlap / len(b)) >= threshold


def scale_to_portion(per_100g: dict, portion_g: float) -> dict[str, float]:
    """把每 100 公克的數值換算成實際份量的數值。

    缺值當 0 處理：食藥署資料有些欄位沒測（見 pitfalls #32，`meta` 保留 key 設 None）。
    當 0 會低估，但比讓整項計算爆掉好——而且 grounded 旗標仍然誠實標示來源。
    """
    factor = portion_g / GRAMS_PER_BASIS
    return {f: (per_100g.get(f) or 0.0) * factor for f in NUTRIENT_FIELDS}


def build_extraction(
    raw_text: str,
    parsed: list[ParsedItem],
    matches: list[ItemMatch],
    db_rows: dict[str, dict],
) -> FoodLogExtraction:
    """組合成最終結果。

    Args:
        raw_text: 使用者的原始描述，原樣保留。
        parsed: 第一階段解析出的項目。
        matches: 第二階段的配對結果，用 index 對應 parsed。
        db_rows: source_ref → 該筆的 meta（每 100 公克的數值）。

    Returns:
        FoodLogExtraction。項目順序跟 parsed 一致，不依賴 matches 的順序——
        模型回傳的順序不保證，用 index 對應才可靠（同 pitfalls 的 embedding 排序問題）。
    """
    by_index = {m.index: m for m in matches}
    items: list[FoodItem] = []

    for i, item in enumerate(parsed):
        match = by_index.get(i)
        row = db_rows.get(match.source_ref) if match and match.source_ref else None

        # 名稱差太多就拒絕採用資料庫數值。見 name_matches 的說明：
        # LLM 會把複合餐點配到單一高密度食材，再套整份重量，錯好幾倍。
        rejected_match: str | None = None
        if row is not None and not name_matches(item.name, row.get("name") or ""):
            rejected_match = row.get("name")
            row = None

        if row is not None:
            nutrients = scale_to_portion(row, item.portion_g)
            items.append(
                FoodItem(
                    name=item.name,
                    portion_g=item.portion_g,
                    **nutrients,
                    grounded=True,
                    source_ref=match.source_ref,
                    db_name=row.get("name"),
                    note=item.portion_note,
                )
            )
            continue

        # 沒有資料庫佐證：用 LLM 估的整項數值。
        # 它也沒給的話全部當 0——寧可顯示 0 並標示未佐證，
        # 也不要讓一個看起來合理的數字混進合計裡。
        calories = max(getattr(match, "calories", None) or 0.0, 0.0)

        note = item.portion_note
        if calories == 0:
            # 0 大卡是最糟的輸出：那一餐會靜悄悄地不計入每日合計，
            # 而使用者看不出有東西漏掉。標示出來讓他能手動修正。
            note = f"{note}（無法估算熱量，請手動修正）"
        if rejected_match:
            # 把拒絕的原因留下來。沒有這行的話，「為什麼這項沒有佐證」
            # 要重跑一次才知道，而重跑的結果可能不一樣。
            note = f"{note}（已排除不相符的候選「{rejected_match}」）"

        items.append(
            FoodItem(
                name=item.name,
                portion_g=item.portion_g,
                calories=calories,
                protein_g=max(getattr(match, "protein_g", None) or 0.0, 0.0),
                carbs_g=max(getattr(match, "carbs_g", None) or 0.0, 0.0),
                fat_g=max(getattr(match, "fat_g", None) or 0.0, 0.0),
                grounded=False,
                note=note,
            )
        )

    return FoodLogExtraction(
        items=items,
        food_description=raw_text,
        # 加總一律在 Python 端算，不讓 LLM 自己加。
        calories=sum(x.calories for x in items),
        protein_g=sum(x.protein_g for x in items),
        carbs_g=sum(x.carbs_g for x in items),
        fat_g=sum(x.fat_g for x in items),
    )
