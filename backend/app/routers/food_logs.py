from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user_id
from app.llm import extract_food_log
from app.models import FoodLog as FoodLogModel
from app.models import FoodLogItem as FoodLogItemModel
from app.models import User as UserModel
from app.nutrition import local_today, summarize_day
from app.rag.qa import _local_day_range_utc
from app.schemas import (
    DaySummary,
    ExtractedItemInfo,
    FoodLog,
    FoodLogCreate,
    FoodLogCreated,
    FoodLogDetail,
    FoodLogUpdate,
    InputType,
)

router = APIRouter(prefix="/food_logs", tags=["food_logs"])


@router.post("", response_model=FoodLogCreated)
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
    # 逐項明細跟著落地。注意這裡用的是 relationship 而不是先 commit 再分開 insert：
    # 那樣會有一個中間狀態——紀錄存在、明細還沒存進去。如果第二段 insert 失敗，
    # 資料庫裡就留下一筆「合計有值但沒有明細」的紀錄，而那跟 11 月之前的舊紀錄
    # 長得一模一樣，事後完全分不出是歷史資料還是寫壞的。一次 commit 就沒有這個狀態。
    new_log.items = [
        FoodLogItemModel(
            name=item.name,
            portion_g=item.portion_g,
            calories=item.calories,
            protein_g=item.protein_g,
            carbs_g=item.carbs_g,
            fat_g=item.fat_g,
            grounded=item.grounded,
            source_ref=item.source_ref,
            db_name=item.db_name,
            note=item.note,
        )
        for item in extraction.items
    ]

    db.add(new_log)
    db.commit()
    db.refresh(new_log)  # 把自動產生的id 同步回new_log

    # 回應的明細直接用 extraction 的結果，不從 new_log.items 讀回來。
    # 兩者內容相同，但 extraction 是這次呼叫的真實輸出——
    # 從資料庫讀回來的話，萬一寫入漏了欄位，回應看起來仍然正常，
    # 而使用者核對的是回應。要讓寫入的錯在測試裡現形，就不能拿回應去掩蓋它。
    return FoodLogCreated(
        **{c.name: getattr(new_log, c.name) for c in new_log.__table__.columns},
        items=[ExtractedItemInfo(**item.model_dump()) for item in extraction.items],
        grounded_count=extraction.grounded_count,
    )


@router.get("", response_model=list[FoodLog])
def list_food_logs(user_id: int | None = None, db: Session = Depends(get_db)):
    """查詢飲食紀錄清單，選填 user_id 篩選只看某個使用者的紀錄。"""

    query = db.query(FoodLogModel)

    if user_id is not None:
        query = query.filter(FoodLogModel.user_id == user_id)
        
    return query.all()


@router.get("/summary", response_model=DaySummary)
def get_day_summary(
    days_back: int = 0,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """某一天的飲食摘要：逐餐明細、合計、與目標的差距。

    ## 這條路由必須註冊在 /{food_log_id} 之前

    FastAPI 依**宣告順序**比對路由。如果 `/{food_log_id}` 在前面，
    `/food_logs/summary` 會先命中它，然後 "summary" 轉 int 失敗回 422——
    而錯誤訊息會說「路徑參數不是合法整數」，完全看不出是路由順序的問題。

    ## 為什麼不讓前端自己算

    前端拿 GET /food_logs 的清單自己篩日期加總看起來更省事，但那會讓
    「今天吃了多少」有兩份實作，而只有後端這份有測試。詳見 app/nutrition.py。
    """
    start, end = _local_day_range_utc(days_back)
    logs = (
        db.query(FoodLogModel)
        .filter(
            FoodLogModel.user_id == user_id,
            FoodLogModel.logged_at >= start,
            FoodLogModel.logged_at < end,
        )
        .order_by(FoodLogModel.logged_at)
        .all()
    )
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    day = local_today() - timedelta(days=days_back)

    return summarize_day(logs, user, day)


@router.get("/{food_log_id}", response_model=FoodLogDetail)
def get_food_log(food_log_id: int, db: Session = Depends(get_db)):
    """查詢單一筆飲食紀錄，含逐項明細。

    2026-11-06 之前存進來的紀錄沒有明細，`items` 會是空清單。
    那不是錯誤，但也不表示「這餐只有一個項目」——呼叫端要分得出
    「沒有明細」和「明細是空的」。這裡用空清單表示前者是可接受的妥協，
    因為合計欄位永遠有值；真正需要分辨的是排序工具，那裡有涵蓋率說明。
    """
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
    
