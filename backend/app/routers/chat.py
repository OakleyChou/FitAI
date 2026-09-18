"""飲食問答 API。把 agent 接到 HTTP 上。"""

from fastapi import APIRouter, Depends

from app.agent.loop import run_agent
from app.deps import get_current_user_id
from app.errors import ServiceUnavailable
from app.schemas import ChatRequest, ChatResponse, ToolCallInfo

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    user_id: int = Depends(get_current_user_id),
) -> ChatResponse:
    """回答一個飲食相關問題。

    端點是 `def` 而不是 `async def`：底層的資料庫查詢與 OpenAI 呼叫都是
    阻塞式的。FastAPI 會把同步端點丟到執行緒池，不會卡住事件迴圈；
    寫成 async 反而會讓一個慢查詢卡住所有請求。

    ## AgentResult 怎麼對應到 HTTP

    run_agent 不會拋例外，它用 AgentResult.error 表示系統層問題。
    所以這裡要判斷「有錯誤」到底算不算失敗：

        有答案 + 有錯誤 → 200，degraded=True
            agent 已經降級成功，那個答案對使用者有價值。
            回 503 的話前端會當成失敗，反而把有用的回答丟掉。

        沒答案 + 有錯誤 → 503
            完全失敗，沒有東西可以給使用者。
    """
    result = run_agent(payload.question, user_id=user_id)

    # 降級的判斷要涵蓋兩種來源：
    #   result.error          迴圈本身失敗（LLM 掛了、超過時間預算）
    #   tool_call.unavailable 個別工具失敗（資料庫連不上）
    # 只看前者的話，「DB 掛掉但 agent 成功降級」會被回報成一切正常——
    # 前端就不會提示使用者有部分功能不可用。
    tool_failed = any(c.unavailable for c in result.tool_calls)
    degraded = result.error is not None or tool_failed

    if result.error and not result.answer.strip():
        # detail 進 log，message 給使用者——兩種觀眾分開
        raise ServiceUnavailable(detail=result.error)

    return ChatResponse(
        answer=result.answer,
        degraded=degraded,
        # 只回傳工具名稱與參數，不含完整結果。後者可能上千字，
        # 而且是內部細節，前端用不到。
        tool_calls=[
            ToolCallInfo(name=c.name, arguments=c.arguments) for c in result.tool_calls
        ],
    )
