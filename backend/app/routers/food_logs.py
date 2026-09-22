from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.llm import extract_food_log
from app.models import FoodLog as FoodLogModel
from app.models import User as UserModel
from app.schemas import FoodLog, FoodLogCreate, FoodLogUpdate, InputType

router = APIRouter(prefix="/food_logs", tags=["food_logs"])


@router.post("", response_model=FoodLog)
def create_food_log(payload: FoodLogCreate, db: Session = Depends(get_db)):
    """新增一筆飲食紀錄：把使用者的自然語言輸入丟給 LLM 解析，再存進資料庫。"""
    # 先確認使用者存在，再呼叫 LLM。順序反過來的話有兩個問題：
    #   1. user_id 不存在時，外鍵約束會在 commit 才爆，回給客戶端的是 500——
    #      但這是客戶端給錯資料，500 等於叫對方重試，而重試永遠不會成功。
    #   2. 那次 LLM 呼叫白花了錢，而且結果注定要丟掉。
    # 通則：便宜的驗證放在昂貴的呼叫前面。
    if db.get(UserModel, payload.user_id) is None:
        raise HTTPException(status_code=404, detail="User not exist")

    extraction = extract_food_log(payload.raw_text)

    new_log = FoodLogModel(
        user_id=payload.user_id,
        logged_at=datetime.now(timezone.utc),
        calories=extraction.calories,
        protein_g=extraction.protein_g,
        carbs_g=extraction.carbs_g,
        fat_g=extraction.fat_g,
        raw_input_text=payload.raw_text,
        input_type=InputType.text,
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)  # 把自動產生的id 同步回new_log

    return new_log


@router.get("", response_model=list[FoodLog])
def list_food_logs(user_id: int | None = None, db: Session = Depends(get_db)):
    """查詢飲食紀錄清單，選填 user_id 篩選只看某個使用者的紀錄。"""

    query = db.query(FoodLogModel)

    if user_id is not None:
        query = query.filter(FoodLogModel.user_id == user_id)
        
    return query.all()


@router.get("/{food_log_id}", response_model=FoodLog)
def get_food_log(food_log_id: int, db: Session = Depends(get_db)):
    """查詢單一筆飲食紀錄。"""
    query = db.query(FoodLogModel).filter(FoodLogModel.id == food_log_id)
    result = query.first()
    if result is not None:
        return result
    else:
        raise HTTPException(status_code=404, detail='Food log not exist')
    

@router.put("/{food_log_id}", response_model=FoodLog)
def update_food_log(food_log_id: int, payload: FoodLogUpdate, db: Session = Depends(get_db)):
    """更新一筆飲食紀錄的營養素欄位（手動修正用，不會重新呼叫 LLM）。"""
    query = db.query(FoodLogModel).filter(FoodLogModel.id == food_log_id)
    result = query.first()

    if result is not None:
        update_data = payload.model_dump(exclude_unset=True)  # 把 Pydantic 物件轉成dict
        for field, value in update_data.items():
            setattr(result, field, value)  # 物件, 欄位名稱, 值
        db.commit()
        db.refresh(result)
        return result
    else:
        raise HTTPException(status_code=404, detail='Food log not exist')


@router.delete("/{food_log_id}", status_code=204)
def delete_food_log(food_log_id: int, db: Session = Depends(get_db)):
    """刪除一筆飲食紀錄。"""
    query = db.query(FoodLogModel).filter(FoodLogModel.id == food_log_id)
    result = query.first()
    if result is not None:
        db.delete(result)
        db.commit()
        # status_code=204 -> 成功但不回傳內容，常用在刪除
    else:
        raise HTTPException(status_code=404, detail='Food log not exist')
    
