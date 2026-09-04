"""把外部資料灌進 knowledge_chunks：讀檔 → 轉文字 → embed → 寫 DB。"""

from pathlib import Path

import pandas as pd

from app.database import SessionLocal
from app.models import KnowledgeChunk
from app.rag.chunking import COL_CATEGORY, COL_ID, COL_NAME, food_record_to_text
from app.rag.embedding import embed_texts

# 食藥署食品營養成分資料庫。用常數而不是每次傳字串，避免打錯字造成
# 「刪不到舊資料 + 灌進一批新的」的髒資料。
SOURCE_FOOD_COMPOSITION = "twfda_food_composition"

FOOD_DB_SHEET = "台灣食品成分表"
FOOD_DB_HEADER_ROW = 1  # 第 0 列是「*本資料庫所列數值單位均為每100 g...」的說明文字

# xlsx 欄位名 → meta 的 key。用英文 snake_case 而不是原始中文欄位名，
# 是為了跟 models.FoodLog / schemas.FoodLog 的命名對齊——之後把檢索結果
# 換算成使用者的飲食紀錄時可以直接對應，不用中間再翻譯一層。
META_KEYS = {
    "熱量(kcal)": "calories",
    "粗蛋白(g)": "protein_g",
    "粗脂肪(g)": "fat_g",
    "總碳水化合物(g)": "carbs_g",
    "膳食纖維(g)": "fiber_g",
    "糖質總量(g)": "sugar_g",
    "鈉(mg)": "sodium_mg",
}


def _build_meta(record: dict) -> dict:
    """從一列原始資料抽出結構化欄位，給 KnowledgeChunk.meta 用。

    跟 content 的兩個差別：
    - 數值不 round。meta 是給程式運算的，保留原始精度；content 才是給人讀的。
    - 缺值保留 key、值設成 None（content 是整項跳過）。這樣讀取端不會 KeyError，
      而且 None 明確表達「這項沒測」，跟「我忘了存這欄」區分得開。

    NaN 一定要轉掉：JSON 標準沒有 NaN 這個值，直接丟給 JSONB 會存不進去。
    順便用 float() 轉成 Python 原生型別——numpy 的 float64 不能被 json 序列化。
    """
    meta = {
        "food_id": record[COL_ID],
        # name 是接 agent 工具時才發現要的：工具回傳結構化的 FoodItem，
        # 需要單獨的名稱欄位。名稱雖然也在 content 裡，但從那段中文句子
        # 用字串切割還原名稱很脆弱，格式一改就壞。
        "name": record[COL_NAME],
        "category": record[COL_CATEGORY],
    }
    for col, key in META_KEYS.items():
        value = record[col]
        meta[key] = None if pd.isna(value) else float(value)
    return meta


def ingest_food_composition(xlsx_path: Path, limit: int | None = None) -> int:
    """把食藥署食品營養成分資料庫灌進知識庫。全表 2213 筆。

    這個函式必須是冪等的——重跑一次不該讓資料變兩份。
    做法是灌之前先把同一個 source 的舊資料整批刪掉（delete-then-insert）。
    比起逐筆比對更新，這在資料量小的時候簡單得多，也不會留下孤兒資料。

    Args:
        xlsx_path: 食藥署 xlsx 的路徑。
        limit: 只灌前 N 筆，開發時用來省 embedding 費用跟等待時間。

    Returns:
        實際寫進去的 chunk 數量。
    """
    df = pd.read_excel(xlsx_path, sheet_name=FOOD_DB_SHEET, header=FOOD_DB_HEADER_ROW)
    if limit is not None:
        df = df.head(limit)
    records = df.to_dict("records")

    # 文字跟 meta 分開組，但兩個 list 的順序必須跟 records 一致——後面要靠位置 zip 起來
    texts = [food_record_to_text(record) for record in records]
    metas = [_build_meta(record) for record in records]

    # 一次全部送進去，embed_texts 內部會自己依 embedding_batch_size 分批
    vectors = embed_texts(texts)

    db = SessionLocal()
    try:
        # 冪等的關鍵：先把這個 source 的舊資料整批刪掉，再寫新的。
        # 跟下面的 add_all 共用同一個 transaction，commit 前不會有「刪掉了但還沒寫入」的空窗。
        db.query(KnowledgeChunk).filter(
            KnowledgeChunk.source == SOURCE_FOOD_COMPOSITION
        ).delete()

        # strict=True：三個 list 長度不一致時直接報錯。
        # 沒有它的話 zip 會靜靜地截短到最短的那個，變成「少灌了幾筆但沒人發現」。
        chunks = [
            KnowledgeChunk(
                source=SOURCE_FOOD_COMPOSITION,
                source_ref=meta["food_id"],
                content=text,
                meta=meta,
                embedding=vector,
            )
            for text, meta, vector in zip(texts, metas, vectors, strict=True)
        ]
        db.add_all(chunks)
        db.commit()
        return len(chunks)
    finally:
        db.close()


if __name__ == "__main__":
    # 開發時直接跑：poetry run python -m app.rag.ingest
    count = ingest_food_composition(Path("data/食品營養成分資料庫2025版.xlsx"))  # , limit=20
    print(f"ingested {count} chunks")
    
    db = SessionLocal()
    try:
        total = db.query(KnowledgeChunk).filter(
            KnowledgeChunk.source == SOURCE_FOOD_COMPOSITION
        ).count()
        print(f"資料庫實際筆數: {total}")
    finally:
        db.close()
