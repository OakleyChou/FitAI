import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import settings

client = instructor.from_openai(OpenAI(api_key=settings.openai_api_key))


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
                "content": (
                    "你是一個專業的營養分析助手。使用者輸入可能包含多個食物項目，"
                    "請逐一辨識所有提到的食物項目，針對每一項各自估算營養素（不用加總）。"
                    "若份量不明時，用一般常見的份量估算。"
                    "熱量以大卡（kcal）為單位，重量以公克為單位。"
                ),
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
    result = extract_food_log("我中午吃了一碗雞肉飯加一杯珍珠奶茶")
    print(result)
