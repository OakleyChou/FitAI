"""問答頁。這是整個專案最值得展示的一頁。

## 為什麼要把工具路徑畫出來

一般的 AI 聊天介面到處都是，從畫面上看不出跟「包一層 ChatGPT」的差別。
但 `/chat` 回傳了別人沒有的東西：

    {"answer": "...", "degraded": false,
     "tool_calls": [{"name": "query_food_logs", "arguments": {"days_back": 0}},
                    {"name": "rank_by_nutrient", "arguments": {...}}]}

把它展開顯示，看的人一眼就知道：有工具選擇、有參數決策、有多步推理。
**而且是當場跑出來的，不是宣稱的。**

## 為什麼每次 rerun 要重畫整段歷史

Streamlit 每次互動都把整支腳本重跑一遍，畫面不是增量更新的。
所以歷史訊息存在 session_state 裡，每次重跑時整批重畫——
看起來很浪費，但那就是 Streamlit 的模型：**畫面是狀態的函數**。

每則 assistant 訊息要連 degraded 與 tool_calls 一起存。只存文字的話，
重跑之後工具路徑就消失了——使用者按一下輸入框，上一則的證據就沒了。
"""

import json

import streamlit as st

import api

EXAMPLES = [
    "我今天熱量還剩多少？可以吃什麼點心？",
    "有什麼高蛋白又低卡的食物？",
    "我這週跟上週比，熱量有下降嗎？",
    "我吃的那個叫什麼粿的熱量多少？",
]


def _init_state() -> None:
    # 只在第一次跑時初始化。少了這個 if，每次互動都會把歷史清空。
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None


def _render_tool_calls(tool_calls: list[dict]) -> None:
    """把 agent 的工具呼叫路徑畫出來。"""
    if not tool_calls:
        st.caption("🔧 沒有呼叫工具（一般對話或超出範圍的問題）")
        return

    names = " → ".join(c["name"] for c in tool_calls)
    with st.expander(f"🔧 呼叫了 {len(tool_calls)} 個工具：{names}"):
        for i, call in enumerate(tool_calls, 1):
            args = json.dumps(call.get("arguments", {}), ensure_ascii=False)
            st.code(f"{i}. {call['name']}({args})", language="python")
        st.caption(
            "工具結果本身不回傳到前端——那可能上千字，而且是內部細節。"
            "除錯時先看路徑，那比看答案有用得多。"
        )


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # degraded 的意思是「部分工具失敗，但 agent 仍產出了有用的回答」。
            # 後端刻意回 200 而不是 503，因為那個答案對使用者有價值；
            # 前端有責任把這件事說出來，否則使用者會以為資料是完整的。
            if msg.get("degraded"):
                st.warning("部分功能暫時不可用，以下回答可能不完整。", icon="⚠️")
            st.markdown(msg["content"])
            _render_tool_calls(msg.get("tool_calls", []))
        else:
            st.markdown(msg["content"])


def _ask(question: str) -> None:
    """送出問題並把結果存進 session_state。"""
    st.session_state.messages.append({"role": "user", "content": question})

    try:
        # agent 要跑迴圈、呼叫多次 LLM，通常 10～30 秒。
        # 沒有任何回饋的 20 秒等於一個壞掉的頁面。
        with st.spinner("查詢工具並思考中…（agent 要跑多步，約 10～30 秒）"):
            data = api.ask(question)
    except api.ApiUnreachable as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"❌ {exc.message}\n\n請確認後端已啟動："
                "`docker compose up -d`，或在 backend/ 執行 "
                "`poetry run uvicorn app.main:app --reload`。",
            }
        )
        return
    except api.ApiError as exc:
        # 後端的 message 已經是給使用者看的文字（細節留在它的 log 裡），
        # 直接顯示即可，不要自己再包一層猜測。
        st.session_state.messages.append(
            {"role": "assistant", "content": f"❌ {exc.message}"}
        )
        return

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "degraded": data.get("degraded", False),
            "tool_calls": data.get("tool_calls", []),
        }
    )


def render() -> None:
    _init_state()

    st.title("飲食問答")
    # 刻意不寫工具的「個數」。工具清單維護在後端（app/agent/schemas.py），
    # 前端拿不到也不該拿到——寫一個數字在這裡，它就會在加工具的那天變成假的，
    # 而且不會報錯。這份專案已經為同一類錯誤付過三次代價（pitfalls 第十三節）。
    st.caption(
        "問題會送給一個只有唯讀工具的 agent：查你的飲食紀錄、排序你吃過的餐與食物、"
        "語意搜尋營養資料庫、跨天統計。它自己決定呼叫哪些、傳什麼參數——"
        "每個回答下面都可以展開看它實際走了什麼路徑。"
    )

    if not st.session_state.messages:
        st.markdown("**可以試試：**")
        # 範例按鈕排成兩欄。按下只記錄待處理的問題，不直接送——
        # 送出要發生在歷史重畫之後，否則使用者看不到自己的那則訊息先出現。
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, use_container_width=True, key=f"eg{i}"):
                st.session_state.pending = example
                st.rerun()

    for msg in st.session_state.messages:
        _render_message(msg)

    typed = st.chat_input("問我關於你的飲食或營養成分的問題…")
    question = typed or st.session_state.pending
    st.session_state.pending = None

    if question:
        _ask(question)
        # 重跑一次讓新訊息從歷史迴圈畫出來，而不是在這裡另外畫一份。
        # 少了這行，這一輪的訊息會用不同的程式碼路徑渲染，兩邊遲早會不一致。
        st.rerun()

    if st.session_state.messages and st.button("清空對話"):
        st.session_state.messages = []
        st.rerun()
