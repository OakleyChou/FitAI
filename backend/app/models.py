from datetime import datetime
from typing import Any, Optional
from pgvector.sqlalchemy import Vector
from sqlalchemy import Enum as SqlEnum
from sqlalchemy import ForeignKey, Text
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base
from app.schemas import InputType

# text-embedding-3-small 的輸出維度。換 embedding model 就要一起改，
# 而且要重新 embed 所有既有資料——不同模型的向量空間不相通。
EMBEDDING_DIM = 1536


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True)
    height_cm: Mapped[float] = mapped_column()
    current_weight_kg: Mapped[float] = mapped_column()
    target_weight_kg: Mapped[Optional[float]] = mapped_column(nullable=True)
    daily_calories_goal: Mapped[Optional[float]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class FoodLog(Base):
    __tablename__ = "food_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    logged_at: Mapped[datetime]

    calories: Mapped[float] = mapped_column()
    protein_g: Mapped[float] = mapped_column()
    carbs_g: Mapped[float] = mapped_column()
    fat_g: Mapped[float] = mapped_column()
    raw_input_text: Mapped[Optional[str]] = mapped_column(nullable=True)
    input_type: Mapped[InputType] = mapped_column(SqlEnum(InputType))
  

class WeightLog(Base):
    __tablename__ = "weight_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    weight_kg: Mapped[float] = mapped_column()
    logged_at: Mapped[datetime] = mapped_column()


class PromptVersion(Base):
    """Prompt 版本紀錄。
    name: 用來區分不同用途的 prompt（例如:"food_extraction_system_prompt"），
        同一個 name 底下可以有多筆不同version
    is_active: 標記目前生效的是哪一筆——換版本時新增一筆新資料、
        把舊的 is_active 改 False，而不是直接覆蓋舊內容。
    """

    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    version: Mapped[int]
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    is_active: Mapped[bool] = mapped_column(default=True)


class KnowledgeChunk(Base):
    """RAG 知識庫的最小檢索單位。

    一筆 = 一段會被 embed 起來、之後可能被撈出來塞進 prompt 的文字。

    source: 來源類別，例如 "twfda_food_composition"（食藥署營養成分）
        或 "nutrition_article"。重新 ingest 時用它來刪掉整批舊資料，
        檢索時也可以用它限定只搜某一種來源。
    source_ref: 這筆在原始資料裡的識別碼（食品代碼、檔名等），用來回溯來源。
    content: 實際送去 embed 的文字。retrieval 時塞進 prompt 的也是這個欄位，
        所以它必須是「人看得懂、LLM 也讀得懂」的完整句子，不是原始 CSV 欄位。
    meta: 結構化欄位，放不適合寫進 content 但之後可能要拿來過濾或顯示的東西
        （食物分類、chunk 序號、每 100g 的熱量數值……）。用 JSONB 是因為
        不同 source 需要的欄位不一樣，不想每加一種來源就改一次 schema。
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(index=True)
    source_ref: Mapped[Optional[str]] = mapped_column(nullable=True)
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # 向量索引先不建。理由：pgvector 的索引是 approximate search，會犧牲召回率換速度，
    # 資料量小（<10k 筆）時全表掃描既準又快，建了反而是負優化。
    # 之後資料量長大再打開下面這段（需要 from sqlalchemy import Index；
    # cosine 要配 vector_cosine_ops，距離運算子必須跟索引的 ops class 一致，
    # 否則查詢不會走索引）：
    #
    # __table_args__ = (
    #     Index(
    #         "ix_knowledge_chunks_embedding_hnsw",
    #         "embedding",
    #         postgresql_using="hnsw",
    #         postgresql_with={"m": 16, "ef_construction": 64},
    #         postgresql_ops={"embedding": "vector_cosine_ops"},
    #     ),
    # )


class FeedbackReport(Base):
    """使用者回報「這個數字怪怪的」。

    ## 這不是客服信箱，是 eval 案例的收集器

    這個專案的記錄側沒有 eval（見 docs/pitfalls.md #46），所以每一個 bug 都是
    使用者手動踩到才發現的：鮮奶茶 50 大卡、泡麵 1420 大卡、五片豬肉片 698 大卡。
    一個一個修是打地鼠，而且每次只驗證剛改的那一個，不知道有沒有弄壞別的。

    這張表的欄位是**照 eval case 的需要設計的**：

        raw_input                 輸入
        system_value              系統算出什麼
        expected_min/max          可接受區間  ← eval 的判準就是這個
        context                   當時的擷取細節（份量、配到哪一筆、有無佐證）

    累積足夠筆數之後可以直接轉成 evals/dataset 的測試案例，
    那時「換模型會不會更好」才有辦法用數據回答。

    ## 為什麼新增表不需要 migration

    `create_all` 只建立「不存在」的表，所以新表會被建起來；
    它不會修改既有表的結構——那才是這個專案的技術債（見 README 已知限制）。
    既有的資料庫重跑一次 `python -m scripts.init_db` 就會補上這張表。
    """

    __tablename__ = "feedback_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # "food_log"（記錄的數字怪）或 "chat"（問答的回答怪）
    source: Mapped[str] = mapped_column(index=True)
    # 使用者當初輸入的原話。eval 要用它當輸入，所以必須原樣保留。
    raw_input: Mapped[str] = mapped_column(Text)
    # 系統當時給的數字（熱量）。問答類的回報可能沒有數字，所以允許 None。
    system_value: Mapped[Optional[float]] = mapped_column(nullable=True)

    # 使用者認為合理的區間。**這就是 eval 的判準**——
    # 刻意要區間而不是單一數字：營養估算本來就沒有唯一正確答案，
    # 要求使用者給一個精確值會逼他亂填。
    expected_min: Mapped[Optional[float]] = mapped_column(nullable=True)
    expected_max: Mapped[Optional[float]] = mapped_column(nullable=True)

    # "too_high" / "too_low" / "wrong_match" / "other"
    kind: Mapped[str] = mapped_column(index=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 當時的擷取細節：逐項的份量、配到哪一筆 source_ref、有沒有佐證。
    # 存 JSON 而不是拆成欄位：這份結構會隨擷取流程演進，
    # 而回報是**歷史紀錄**，不該因為流程改版就要 migration。
    #
    # with_variant 讓它在 Postgres 上是 JSONB、在 SQLite 上是 JSON。
    # 沒有這個的話整合測試建表就會失敗（SQLiteTypeCompiler 沒有 visit_JSONB），
    # 而那不是真的限制——我們從來不在這個欄位裡做 JSONB 的運算子查詢，
    # 只是整包存整包取。knowledge_chunks.meta 不一樣，那裡真的需要 JSONB
    # （rank_by_nutrient 要對它做數值排序），所以那張表仍然只能在 Postgres 上建。
    context: Mapped[dict[str, Any]] = mapped_column(
        JSONB().with_variant(JSON, "sqlite"), default=dict
    )

    # 已經轉成 eval 案例的回報標記起來，避免重複處理。
    converted_to_eval: Mapped[bool] = mapped_column(default=False)
