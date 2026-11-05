"""回報怪怪的數字。

## 這一頁不是客服信箱

這個專案的記錄側沒有 eval（見 `docs/pitfalls.md` #46），所以每一個 bug 都是
手動踩到才發現的：鮮奶茶 50 大卡、泡麵 1420 大卡、五片豬肉片 698 大卡。
一個一個修是打地鼠，而且每次只驗證剛改的那一個案例。

所以這一頁的目的是**把「覺得怪」轉成一筆可以放進 eval 的資料**。
送出去的欄位就是一筆 eval case 的全部內容：

    輸入            raw_input
    系統算出什麼    system_value
    可接受區間      expected_min / expected_max   ← eval 的判準
    擷取細節        context（份量、配到哪一筆、有無佐證）

## 為什麼要從既有紀錄預填

手打的話，`raw_input` 會跟當初輸入的不完全一樣，而 eval 的輸入必須是原話；
`context` 更不可能手打。點既有的紀錄預填，拿到的才是真的那一筆。
"""

import streamlit as st

import api

KINDS = {
    "too_high": "數字太高",
    "too_low": "數字太低",
    "wrong_match": "配到錯的食物／答錯問題",
    "other": "其他",
}


def _init_state() -> None:
    if "report_target" not in st.session_state:
        st.session_state.report_target = None


def _pick_from_logs() -> None:
    """列出最近的紀錄，點一下就預填。"""
    try:
        logs = api.list_logs()
    except api.ApiError as exc:
        st.error(exc.message)
        return

    if not logs:
        st.info("還沒有飲食紀錄。到「記錄飲食」頁新增一筆之後就能從這裡回報。")
        return

    recent = sorted(logs, key=lambda x: x["logged_at"], reverse=True)[:10]
    st.caption(f"最近 {len(recent)} 筆紀錄，點「這筆怪怪的」就會帶入下面的表單：")

    for log in recent:
        cols = st.columns([5, 2, 2])
        cols[0].write(f"**{log['raw_input_text'] or '（無描述）'}**")
        cols[1].write(f"{log['calories']:.0f} 大卡")
        # key 用 id 而不是迴圈索引——刪掉一筆之後索引會位移，
        # Streamlit 會把按鈕狀態對到錯的那一列。
        if cols[2].button("這筆怪怪的", key=f"pick{log['id']}"):
            st.session_state.report_target = {
                "source": "food_log",
                "raw_input": log["raw_input_text"] or "",
                "system_value": float(log["calories"]),
                # 逐項明細沒有落地（food_logs 只存合計），所以這裡只帶得到
                # 合計與 id。要完整的擷取細節得在記錄當下回報——
                # 這是目前的限制，不是設計選擇。
                "context": {
                    "food_log_id": log["id"],
                    "totals": {
                        "calories": log["calories"],
                        "protein_g": log["protein_g"],
                        "carbs_g": log["carbs_g"],
                        "fat_g": log["fat_g"],
                    },
                },
            }
            st.rerun()


def _report_form(target: dict) -> None:
    st.divider()
    st.subheader("回報內容")

    is_chat = target["source"] == "chat"
    st.write(f"**輸入**：{target['raw_input'] or '（請在下方填寫）'}")
    if target.get("system_value") is not None:
        st.write(f"**系統算出**：{target['system_value']:.0f} 大卡")

    with st.form("report_form"):
        raw_input = st.text_input(
            "當初輸入的原話",
            value=target["raw_input"],
            help="eval 要用它當輸入，所以必須跟你當初打的一字不差",
        )
        kind = st.selectbox(
            "哪裡怪",
            options=list(KINDS),
            format_func=lambda k: KINDS[k],
        )

        lo = hi = None
        if not is_chat:
            st.caption(
                "**你覺得合理的熱量區間** —— 這是最重要的欄位，eval 的判準就是它。"
                "刻意要區間而不是單一數字：營養估算沒有唯一正確答案。"
            )
            c1, c2 = st.columns(2)
            lo = c1.number_input("最低（大卡）", min_value=0.0, step=10.0, value=0.0)
            hi = c2.number_input("最高（大卡）", min_value=0.0, step=10.0, value=0.0)

        note = st.text_area(
            "為什麼覺得怪（選填）",
            placeholder="例如：配到山豬肉片，而且一片不可能 30 公克",
        )
        submitted = st.form_submit_button("送出回報", type="primary")

    if not submitted:
        return

    if not raw_input.strip():
        st.warning("請填寫當初輸入的原話。")
        return
    if lo and hi and lo > hi:
        st.warning("區間顛倒了：最低不能大於最高。")
        return

    try:
        api.report_feedback(
            source=target["source"],
            raw_input=raw_input.strip(),
            kind=kind,
            system_value=target.get("system_value"),
            # 0 當成「沒填」。number_input 沒有空值，所以用 0 當哨兵——
            # 熱量為 0 的可接受區間本來就沒有意義。
            expected_min=lo or None,
            expected_max=hi or None,
            note=note.strip() or None,
            context=target.get("context", {}),
        )
    except api.ApiError as exc:
        st.error(exc.message)
        return

    st.success("已記錄。累積一批之後可以直接轉成 eval 的測試案例。")
    st.session_state.report_target = None


def _render_existing() -> None:
    try:
        rows = api.list_feedback()
    except api.ApiError:
        return
    if not rows:
        return

    st.divider()
    pending = sum(1 for r in rows if not r["converted_to_eval"])
    st.subheader(f"已回報 {len(rows)} 筆（{pending} 筆還沒轉成 eval）")

    for r in rows:
        mark = "🟡" if not r["converted_to_eval"] else "✅"
        rng = (
            f"（你認為 {r['expected_min']:.0f}~{r['expected_max']:.0f}）"
            if r["expected_min"] is not None and r["expected_max"] is not None
            else ""
        )
        value = f"{r['system_value']:.0f} 大卡" if r["system_value"] is not None else "—"
        cols = st.columns([6, 2])
        cols[0].write(f"{mark} **{r['raw_input']}** → {value} {rng}　`{KINDS.get(r['kind'], r['kind'])}`")
        if r["note"]:
            cols[0].caption(f"　　{r['note']}")
        if cols[1].button("刪除", key=f"delfb{r['id']}"):
            try:
                api.delete_feedback(r["id"])
            except api.ApiError as exc:
                st.error(exc.message)
            else:
                st.rerun()


def render() -> None:
    _init_state()

    st.title("回報怪怪的")
    st.caption(
        "這不是客服信箱 —— 回報會被存成**可以直接轉成 eval 測試案例**的格式。"
        "記錄側目前沒有自動化檢查，所以每個 bug 都是手動踩到才發現的；"
        "累積一批回報之後就能建立測試集。"
    )

    target = st.session_state.report_target
    if target is None:
        _pick_from_logs()
        st.divider()
        if st.button("回報問答的答案（不是記錄的數字）"):
            st.session_state.report_target = {
                "source": "chat",
                "raw_input": "",
                "system_value": None,
                "context": {},
            }
            st.rerun()
    else:
        if st.button("← 換一筆"):
            st.session_state.report_target = None
            st.rerun()
        _report_form(target)

    _render_existing()
