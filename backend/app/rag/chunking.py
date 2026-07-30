"""把原始資料轉成「一段適合被 embed 的文字」。

chunk 的大小是 RAG 最核心的取捨之一：
- 太小 → 失去上下文，撈出來的片段 LLM 讀不懂（例如只有「含 12 公克」不知道在講什麼）
- 太大 → 一個 chunk 塞太多主題，向量變成各主題的平均，跟任何 query 都只有中等相似度
  （這叫 semantic dilution，是召回率變差的主因）

食物營養成分這種結構化資料不需要切分——一筆食物天生就是一個語意完整的單位，
所以 food_record_to_text() 做的是「轉寫」而不是「切分」。
真正需要 chunk_text() 的是之後要加的知識文章（B 類資料）。
"""

import pandas as pd

# 食藥署 2025 版 xlsx 的欄位名稱。sheet 是「台灣食品成分表」，
# 且第 1 列是說明文字、第 2 列才是表頭，讀的時候要 header=1。
COL_ID = "整合編號"
COL_CATEGORY = "食品分類"
COL_NAME = "樣品名稱"
COL_DESC = "內容物描述"
COL_ALIAS = "俗名"

# (欄位名, 顯示名, 單位) —— 會依序寫進描述句。
# 只挑跟飲食追蹤有關的；原始資料有 110 欄，胺基酸/脂肪酸那些對本專案沒用，
# 全寫進去只會稀釋向量、灌爆 token。
NUTRIENT_COLS = [
    ("熱量(kcal)", "熱量", "大卡"),
    ("粗蛋白(g)", "蛋白質", "公克"),
    ("粗脂肪(g)", "脂肪", "公克"),
    ("總碳水化合物(g)", "碳水化合物", "公克"),
    ("膳食纖維(g)", "膳食纖維", "公克"),
    ("糖質總量(g)", "糖", "公克"),
    ("鈉(mg)", "鈉", "毫克"),
]


def food_record_to_text(record: dict) -> str:
    """把食藥署營養成分資料庫的一筆紀錄轉成一句自然語言描述。

    為什麼不直接把欄位值丟去 embed？
    embedding model 是用自然語言訓練的，`"大麥仁,364.6,8.5,1.5"` 這種字串
    在向量空間裡的位置跟使用者會打的 `"我早餐吃了洋薏仁"` 差很遠。
    轉寫成完整句子能讓兩者在向量空間靠得更近，檢索命中率明顯較好。

    Args:
        record: 一列資料，key 是上面那些欄位名稱（pandas 的 to_dict("records")）。

    Returns:
        目標格式（有俗名時）：
            "大麥仁（穀物類），又稱小薏仁、洋薏仁、珍珠薏仁。樣品狀態:生,已去殼。
             每 100 公克可食部分含熱量 364.6 大卡、蛋白質 8.6 公克、脂肪 1.6 公克、
             碳水化合物 77.1 公克、膳食纖維 8.5 公克、鈉 12.6 毫克。"
        沒有俗名時就跳過「又稱」那一段。

    """
    # 食品名
    result = [f"{record[COL_NAME]}（{record[COL_CATEGORY]}）"]
    # 俗名
    if pd.notna(record[COL_ALIAS]):
        result.append(f"，又稱{record[COL_ALIAS].replace(',', '、')}")
    # 內容物描述
    result.append(f"。{record[COL_DESC]}")
    # 營養素
    nutrient = []
    for col, label, unit in NUTRIENT_COLS:
        if pd.notna(record[col]):
            nutrient.append(f"{label} {round(record[col], 1)} {unit}")
    result.append(f"，每 100 公克可食用部分含：{'、'.join(nutrient)}。")
    
    return "".join(result)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """把長文字切成有重疊的固定長度片段。今天先放著，灌知識文章（B）時才會用到。

    overlap 的用意：避免一個完整語意剛好被切點劈成兩半。
    有重疊的話至少有一個 chunk 會包含完整的那段話。

    Args:
        text: 原始長文字。
        chunk_size: 每段的目標字元數。
        overlap: 相鄰兩段重疊的字元數，必須 < chunk_size。

    Returns:
        切好的字串 list，順序與原文一致。

    TODO(你):
        1. 用 while 迴圈滑動視窗，每次前進 (chunk_size - overlap)
        2. 注意 overlap >= chunk_size 會造成無限迴圈——要擋
        3. 進階：改成優先在句號/換行處斷開，而不是硬切在字元數上
           （這就是 recursive character splitting 的核心概念）
    """
    raise NotImplementedError


if __name__ == "__main__":
    # 驗證 food_record_to_text：poetry run python -m app.rag.chunking
    # 挑這三筆是因為它們各自代表一種要處理的資料狀況。

    PATH = "data/食品營養成分資料庫2025版.xlsx"
    CASES = {
        "A0100101": "俗名有值 → 要出現「又稱小薏仁、洋薏仁、珍珠薏仁」",
        "A0200101": "糖質總量是 NaN → 該項要整個消失，不能是「糖 nan 公克」",
        "I0402101": "俗名是 NaN → 整段「又稱」跳過，句子不能斷掉或多出逗號",
    }

    df = pd.read_excel(PATH, sheet_name="台灣食品成分表", header=1)
    for food_id, expectation in CASES.items():
        record = df[df[COL_ID] == food_id].to_dict("records")[0]
    #     print(f"\n=== {food_id} ｜ {expectation}")
        print(food_record_to_text(record))
