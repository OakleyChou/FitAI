"""回報「這個數字怪怪的」。

## 這不是客服信箱

這個專案的記錄側沒有 eval（見 `docs/pitfalls.md` #46），所以每一個 bug 都是
使用者手動踩到才發現的：鮮奶茶 50 大卡、泡麵 1420 大卡、五片豬肉片 698 大卡。
一個一個修是打地鼠——而且每次只驗證剛改的那一個案例，
不知道有沒有弄壞別的（泡麵那次就差點發生，順手加的 wrapper 讓多項目輸入 500）。

所以這個端點的目的是**把「使用者覺得怪」轉成一筆可以放進 eval 的資料**：
輸入、系統算出什麼、使用者認為合理的區間、當時的擷取細節。
那正好是一筆 eval case 的全部欄位。

累積足夠筆數之後，`GET /feedback?pending_only=true` 可以撈出還沒轉成
eval 的回報，那時「換模型會不會更好」才有辦法用數據回答。
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user_id
from app.models import FeedbackReport
from app.models import User as UserModel
from app.schemas import Feedback, FeedbackCreate

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=Feedback)
def create_feedback(
    payload: FeedbackCreate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """記下一筆回報。

    `user_id` 從登入狀態取得，不讓客戶端指定——跟 /chat 同樣的理由。
    """
    if db.get(UserModel, user_id) is None:
        raise HTTPException(status_code=404, detail="User not exist")

    # 區間顛倒直接擋掉。Pydantic 只能驗單一欄位的範圍，
    # 「min 不能大於 max」是跨欄位的約束，要在這裡檢查。
    lo, hi = payload.expected_min, payload.expected_max
    if lo is not None and hi is not None and lo > hi:
        raise HTTPException(
            status_code=422,
            detail=f"區間顛倒：expected_min({lo}) 不能大於 expected_max({hi})",
        )

    report = FeedbackReport(
        user_id=user_id,
        source=payload.source.value,
        raw_input=payload.raw_input,
        kind=payload.kind.value,
        system_value=payload.system_value,
        expected_min=lo,
        expected_max=hi,
        note=payload.note,
        context=payload.context,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@router.get("", response_model=list[Feedback])
def list_feedback(
    pending_only: bool = False,
    limit: int = 50,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """列出回報，最新的在前。

    Args:
        pending_only: 只看還沒轉成 eval 案例的。累積一批之後要處理時用。
        limit: 最多幾筆。
    """
    query = db.query(FeedbackReport).filter(FeedbackReport.user_id == user_id)
    if pending_only:
        query = query.filter(FeedbackReport.converted_to_eval.is_(False))
    return query.order_by(FeedbackReport.id.desc()).limit(limit).all()


@router.delete("/{report_id}", status_code=204)
def delete_feedback(
    report_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """刪掉一筆回報（誤報時用）。

    一併比對 user_id：少了這個條件，任何人都能刪別人的回報。
    目前沒有認證所以看不出差別，但這條過濾是對的，不該等到加認證才補。
    """
    report = (
        db.query(FeedbackReport)
        .filter(FeedbackReport.id == report_id, FeedbackReport.user_id == user_id)
        .first()
    )
    if report is None:
        raise HTTPException(status_code=404, detail="Feedback not exist")
    db.delete(report)
    db.commit()
