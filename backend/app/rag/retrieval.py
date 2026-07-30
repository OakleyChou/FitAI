"""向量檢索：把 query 轉成向量，去 knowledge_chunks 找最相近的幾筆。"""

from dataclasses import dataclass

from app.database import SessionLocal
from app.models import KnowledgeChunk
from app.rag.embedding import embed_query


@dataclass
class RetrievedChunk:
    """檢索結果。刻意不直接回傳 SQLAlchemy 物件——

    一是 session 關掉之後那些物件會變得不能存取（DetachedInstanceError），
    二是 distance 是查詢時算出來的，本來就不屬於 model 的欄位。
    """

    content: str
    source: str
    source_ref: str | None
    meta: dict
    distance: float  # cosine distance，0 = 完全相同，2 = 完全相反


def search(query: str, top_k: int = 5, source: str | None = None) -> list[RetrievedChunk]:
    """找出跟 query 語意最相近的 top_k 個 chunk。

    Args:
        query: 使用者的自然語言查詢。
        top_k: 回傳幾筆。太少會漏掉正確答案，太多會稀釋 prompt、增加成本，
            實務上通常從 3~5 開始調。
        source: 只搜這個來源；None 表示全部都搜。

    Returns:
        依相似度由高到低排序（distance 由小到大）。

    為什麼用 cosine 而不是 L2：OpenAI 的 embedding 已經 L2-normalized，
    正規化之後 cosine 相似度跟歐氏距離的排序結果是等價的，
    但 cosine 的數值範圍固定在 [0, 2]，跨 query 比較時比較好解讀。
    """
    query_vector = embed_query(query)
    # cosine_distance 越小越像
    distance = KnowledgeChunk.embedding.cosine_distance(query_vector)
    db = SessionLocal()
    try:
        q = db.query(KnowledgeChunk, distance.label("distance"))  # 等同 SELECT ... AS distance
        # 若有指定來源
        if source is not None:
            q = q.filter(KnowledgeChunk.source == source)
        q = q.order_by(distance).limit(top_k)

        # q.all() 回傳 list of tuples，每個 tuple 是 (KnowledgeChunk 物件, 距離)。
        # 在 session 還開著的時候就複製成純 dataclass，讓結果跟 ORM 脫鉤。
        results = []
        for chunk, dist in q.all():
            results.append(
                RetrievedChunk(
                    content=chunk.content,
                    source=chunk.source,
                    source_ref=chunk.source_ref,
                    meta=chunk.meta,
                    distance=dist,
                )
            )
        return results
    finally:
        db.close()


if __name__ == "__main__":
    # 驗證用：先 ingest 再跑這個，看撈出來的東西合不合理
    # poetry run python -m app.rag.retrieval
    for q in ["雞胸肉", "我早上喝了一杯豆漿", "高蛋白質的食物"]:
        print(f"\n=== {q} ===")
        for chunk in search(q, top_k=3):
            print(f"  [{chunk.distance:.4f}] {chunk.content[:60]}")
