"""記錄飲食。自然語言輸入 → 後端呼叫 LLM 解析 → 存進資料庫。

這一頁展示的是**結構化擷取**：使用者打「一個漢堡加一杯可樂」，
後端的 `extract_food_log` 用 instructor 要求 LLM 回傳 `list[FoodItem]`，
然後**在 Python 端加總**（不讓 LLM 自己加——多項相加它會算錯，
而那種錯誤在文字裡看不出來）。

解析結果直接顯示出來，而且標明是估算值。藏起來的話使用者會把
LLM 的猜測當成測量結果。
"""

import streamlit as st

import api

EXAMPLES = [
    "早餐吃了鮪魚蛋吐司加一杯無糖豆漿",
    "一個漢堡加一杯可樂",
    "午餐是雞胸肉便當配燙青菜",
]


def _submit(raw_text: str) -> None:
    try:
        with st.spinner("解析中…（要呼叫一次 LLM）"):
            created = api.create_log(raw_text)
    except api.ApiUnreachable as exc:
        st.error(f"{exc.message}\n\n請確認後端已啟動（`docker compose up -d`）。")
        return
    except api.ApiError as exc:
        st.error(exc.message)
        return

    st.success("已記錄")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("熱量", f"{created['calories']:.0f} 大卡")
    c2.metric("蛋白質", f"{created['protein_g']:.1f} g")
    c3.metric("碳水", f"{created['carbs_g']:.1f} g")
    c4.metric("脂肪", f"{created['fat_g']:.1f} g")
    st.caption(
        "⚠️ 以上是 LLM 依描述**估算**的，不是測量值。"
        "份量講得越具體（「一碗白飯」而不是「飯」）估得越準。"
    )


def _render_recent() -> None:
    try:
        logs = api.list_logs()
    except api.ApiError:
        # 清單拿不到不該讓整頁掛掉——上面的新增功能仍然可用。
        st.caption("（無法載入既有紀錄）")
        return

    if not logs:
        return

    st.divider()
    recent = sorted(logs, key=lambda x: x["logged_at"], reverse=True)[:10]
    st.subheader(f"最近 {len(recent)} 筆")

    for log in recent:
        cols = st.columns([6, 2, 2])
        cols[0].write(f"**{log['raw_input_text'] or '（無描述）'}**")
        cols[1].write(f"{log['calories']:.0f} 大卡")
        # key 必須唯一且穩定。用 id 而不是迴圈索引——
        # 刪掉一筆之後索引會位移，Streamlit 會把按鈕狀態對到錯的那一列。
        if cols[2].button("刪除", key=f"del{log['id']}"):
            try:
                api.delete_log(log["id"])
            except api.ApiError as exc:
                st.error(exc.message)
            else:
                st.rerun()


def render() -> None:
    st.title("記錄飲食")
    st.caption("用自然語言描述就好，營養素由後端的 LLM 解析，不用自己查表填數字。")

    # 用 form 包起來：沒有 form 的話，每打一個字都會觸發一次 rerun。
    # 送出鈕按下時才跑一次，輸入體驗才正常。
    with st.form("log_form", clear_on_submit=True):
        raw_text = st.text_area(
            "今天吃了什麼？",
            placeholder="例如：早餐吃了鮪魚蛋吐司加一杯無糖豆漿",
            height=100,
        )
        submitted = st.form_submit_button("記錄", type="primary")

    if submitted:
        if not raw_text.strip():
            st.warning("請先描述吃了什麼。")
        else:
            _submit(raw_text.strip())

    with st.expander("不知道怎麼寫？"):
        for ex in EXAMPLES:
            st.code(ex, language=None)

    _render_recent()
