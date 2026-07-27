import instructor
from langfuse.openai import openai
from pydantic import BaseModel, Field
from langfuse.decorators import langfuse_context, observe

from app.config import settings
from app.database import SessionLocal
from app.models import PromptVersion

# 用 langfuse.openai 包過的 openai client，取代原生 openai.OpenAI
# LLM 呼叫會自動被記錄成 Generation 類型的 observation，包含 token 用量/成本
# 也可以在 langfuse 上更新 prompt，不用進資料庫更新
client = instructor.from_openai(openai.OpenAI(api_key=settings.openai_api_key))

FOOD_EXTRACTION_PROMPT_NAME = "food_extraction_system_prompt"


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


class FoodItem(BaseModel):
    """單一食物項目的營養素估算——這是實際傳給 LLM 的 response_model。"""

    name: str = Field(description="食物項目名稱")
    calories: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)


class FoodLogExtraction(BaseModel):
    """extract_food_log() 組裝出的最終結果，LLM 不會直接產生這個 model。

    items 來自 LLM；calories/protein_g/carbs_g/fat_g 是程式碼加總 items 算出來的。
    id / user_id / logged_at 由系統/使用者提供，不放在這裡。
    """

    items: list[FoodItem]
    food_description: str
    calories: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)

@observe()
def extract_food_log(raw_text: str) -> FoodLogExtraction:
    """把自然語言飲食描述轉成結構化的 FoodLogExtraction。

    LLM 只負責辨識每一項食物並各自估算營養素；加總由這個函式計算。
    """
    items = client.chat.completions.create(
        model=settings.openai_model_cheap,
        response_model=list[FoodItem],
        messages=[
            {
                "role": "system",
                "content": get_active_prompt(FOOD_EXTRACTION_PROMPT_NAME),
            },
            {"role": "user", "content": raw_text},
        ],
    )

    return FoodLogExtraction(
        items=items,
        food_description=raw_text,
        calories=sum(item.calories for item in items),
        protein_g=sum(item.protein_g for item in items),
        carbs_g=sum(item.carbs_g for item in items),
        fat_g=sum(item.fat_g for item in items),
    )


if __name__ == "__main__":
    result = extract_food_log("我晚上吃了一個漢堡加一杯可樂")
    print(result)
    langfuse_context.flush()  # 強制把排隊中的 trace 資料立刻送出，不等背景批次處理
