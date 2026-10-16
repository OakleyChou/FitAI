"""今日總覽。純顯示——所有數字都是後端算好的。

這一頁刻意沒有任何算術。合計、剩餘額度、蛋白質目標全部來自
`GET /food_logs/summary`，前端只負責擺位置和挑顏色。

會想在這裡自己算是很自然的（`sum(m["calories"] for m in meals)` 一行就好），
但那樣「今天吃了多少」就有兩份實作，而只有後端那份有測試。
分岔的那天，使用者在對話頁看到「還剩 380 大卡」、在這一頁看到別的數字。
"""

import streamlit as st

import api


def _metric_row(summary: dict) -> None:
    totals = summary["totals"]
    goal = summary.get("calories_goal")
    remaining = summary.get("calories_remaining")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("今日熱量", f"{totals['calories']:.0f} 大卡")

    if goal is not None and remaining is not None:
        # delta_color="inverse" 讓負值顯示紅色——吃超過了是警訊而不是成就。
        # 預設的 normal 會把負數畫成綠色下降，在這個情境下意思剛好相反。
        c2.metric(
            "剩餘額度",
            f"{remaining:.0f} 大卡",
            delta=f"目標 {goal:.0f}",
            delta_color="off",
        )
        if remaining < 0:
            c2.caption(f"⚠️ 超出 {abs(remaining):.0f} 大卡")
    else:
        c2.metric("剩餘額度", "—")
        c2.caption("未設定每日目標")

    protein_goal = summary.get("protein_goal_g")
    if protein_goal:
        c3.metric(
            "蛋白質",
            f"{totals['protein_g']:.0f} g",
            delta=f"目標 約 {protein_goal:.0f} g",
            delta_color="off",
        )
        if summary.get("protein_goal_is_estimated"):
            # 這個目標是依體重推估的，不是使用者設定的。不標明的話
            # 使用者會以為那是他自己設過的數字。
            c3.caption("目標為推估值（體重 × 1.6 g/kg）")
    else:
        c3.metric("蛋白質", f"{totals['protein_g']:.0f} g")

    c4.metric("碳水 / 脂肪", f"{totals['carbs_g']:.0f} / {totals['fat_g']:.0f} g")


def render() -> None:
    st.title("今日總覽")

    days_back = st.selectbox(
        "日期",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda d: {0: "今天", 1: "昨天"}.get(d, f"{d} 天前"),
    )

    try:
        summary = api.day_summary(days_back)
    except api.ApiUnreachable as exc:
        st.error(f"{exc.message}\n\n請確認後端已啟動（`docker compose up -d`）。")
        return
    except api.ApiError as exc:
        st.error(exc.message)
        return

    st.caption(f"日期：{summary['date']}（台北時間）")
    _metric_row(summary)
    st.divider()

    meals = summary["meals"]
    if not meals:
        # 沒有紀錄不是錯誤，是一個正常狀態——後端也刻意不回 404。
        st.info("這一天還沒有飲食紀錄。到「記錄飲食」頁新增一筆。")
        return

    st.subheader(f"餐點明細（{len(meals)} 筆）")
    st.dataframe(
        [
            {
                "時間": m["logged_at_local"],
                "內容": m["description"],
                "熱量": round(m["calories"]),
                "蛋白質 (g)": round(m["protein_g"], 1),
                "碳水 (g)": round(m["carbs_g"], 1),
                "脂肪 (g)": round(m["fat_g"], 1),
            }
            for m in meals
        ],
        use_container_width=True,
        hide_index=True,
    )
