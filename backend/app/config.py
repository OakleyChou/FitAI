from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings 只把 .env 讀進 Settings 物件，不會寫進 os.environ。
# Langfuse SDK 是直接讀 os.environ 找 key，所以這裡額外用 python-dotenv 載入一次，
# 讓 .env 裡的值也變成真正的環境變數。
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "FitAI Coach"
    app_version: str = "0.1.0"
    debug: bool = False

    # Database
    database_url: str = "postgresql+psycopg://user:password@localhost:5432/fitai"

    # LLM
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_model_cheap: str = "claude-haiku-4-5-20251001"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_model_cheap: str = "gpt-4o-mini"

    # Embedding —— 換模型的話 models.EMBEDDING_DIM 要跟著改，且既有向量全部要重算
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
