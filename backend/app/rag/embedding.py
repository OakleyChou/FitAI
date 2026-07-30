"""呼叫 OpenAI Embeddings API，把文字轉成向量。"""

from langfuse.openai import openai

from app.config import settings

# 注意：這裡不能用 app.llm 那個 client。
# llm.py 的 client 被 instructor.from_openai() 包過，只暴露 chat.completions 這條路徑，
# 沒有 .embeddings。所以這裡另外開一個「只有 langfuse 包過、沒有 instructor」的 client——
# 一樣會自動被 Langfuse 記錄成 Generation，token 用量/成本照樣抓得到。
client = openai.OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把多段文字批次轉成向量。

    務必批次送。Embeddings API 的 input 參數吃 list，一次送 100 筆跟送 100 次
    單筆請求相比，延遲差一個數量級（省掉 99 次 round-trip），而且不容易撞 rate limit。

    Args:
        texts: 要轉換的文字，順序有意義。

    Returns:
        向量 list，第 i 個向量對應 texts[i]。長度必須等於 len(texts)。
    """
    # 空輸入直接回傳，不要送一個空請求出去。
    # 用 not texts 而不是 texts is None，這樣 [] 跟 None 都擋得到。
    if not texts:
        return []

    size = settings.embedding_batch_size
    result: list[list[float]] = []

    # 分批處理
    for i in range(0, len(texts), size):
        # 切片超出長度不會 IndexError，最後一批不滿也會自動只取到結尾
        batch = texts[i : i + size]
        resp = client.embeddings.create(
            input=batch,
            model=settings.openai_embedding_model,
        )
        # 先依 index 排序是保險：順序錯掉不會報錯，只會讓文字配到別人的向量。
        sorted_items = sorted(resp.data, key=lambda x: x.index)

        # resp.data 裝的是 Embedding 物件而不是向量本身，要再取 .embedding。
        for item in sorted_items:
            result.append(item.embedding)

    return result


def embed_query(text: str) -> list[float]:
    """把單一查詢字串轉成向量，給 retrieval 用。

    刻意跟 embed_texts 分開命名，是因為之後可能會分化：
    有些 embedding model（例如 BGE、E5 系列）要求 query 和 document 加不同的前綴詞，
    OpenAI 的模型目前不用，但介面先留好。
    """
    return embed_texts([text])[0]


if __name__ == "__main__":
    vs = embed_texts(["雞胸肉", "巧克力蛋糕"])
    print(len(vs), len(vs[0]), len(vs[1]))  # 應該是 2 1536 1536
    print(len(embed_query("午餐")))  # 應該是 1536
