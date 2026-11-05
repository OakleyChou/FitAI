"""回報端點的整合測試。

這張表的欄位是照 **eval case 的需要**設計的，所以測試重點是
「存下來的東西夠不夠轉成一筆 eval」：輸入、系統算出什麼、可接受區間、擷取細節。
"""

import pytest

from app.models import FeedbackReport

VALID = {
    "source": "food_log",
    "raw_input": "五片豬肉片",
    "kind": "too_high",
    "system_value": 698.0,
    "expected_min": 180.0,
    "expected_max": 350.0,
    "note": "配到山豬肉片，而且一片不可能 30 公克",
    "context": {
        "items": [
            {"name": "豬肉片", "portion_g": 150.0, "grounded": True, "db_name": "山豬肉片"}
        ]
    },
}


def test_回報會存下轉成eval所需的全部欄位(client, user, db_session):
    r = client.post("/feedback", json=VALID)

    assert r.status_code == 200
    body = r.json()
    # 這四個欄位缺任何一個，這筆回報就不能轉成 eval case
    assert body["raw_input"] == "五片豬肉片"
    assert body["system_value"] == 698.0
    assert (body["expected_min"], body["expected_max"]) == (180.0, 350.0)
    assert body["context"]["items"][0]["db_name"] == "山豬肉片"
    assert body["converted_to_eval"] is False


def test_user_id不由客戶端決定(client, user, db_session):
    """跟 /chat 同樣的理由：讓客戶端指定就能寫到別人名下。"""
    r = client.post("/feedback", json={**VALID, "user_id": 999})

    assert r.json()["user_id"] == 1


def test_區間顛倒會被擋掉(client, user, db_session):
    """Pydantic 只驗單一欄位的範圍，跨欄位的約束要自己檢查。"""
    r = client.post("/feedback", json={**VALID, "expected_min": 500, "expected_max": 100})

    assert r.status_code == 422
    assert db_session.query(FeedbackReport).count() == 0


@pytest.mark.parametrize(
    "payload, 原因",
    [
        pytest.param({**VALID, "source": "nope"}, "來源不在列舉裡", id="bad_source"),
        pytest.param({**VALID, "kind": "nope"}, "類型不在列舉裡", id="bad_kind"),
        pytest.param({**VALID, "raw_input": ""}, "原始輸入不可為空", id="empty_input"),
        pytest.param({**VALID, "expected_min": -1}, "區間不可為負", id="negative"),
    ],
)
def test_不合法的回報回422(client, user, payload, 原因):
    r = client.post("/feedback", json=payload)

    assert r.status_code == 422, 原因
    assert r.json()["error"]["code"] == "validation_error"


def test_問答類的回報可以沒有數字(client, user, db_session):
    """問答的答案怪怪的時候沒有單一數字可填，不該因此擋掉回報。"""
    r = client.post(
        "/feedback",
        json={
            "source": "chat",
            "raw_input": "我吃過什麼蛋白質最高的食物？",
            "kind": "wrong_match",
            "note": "它答的是資料庫裡的食物，不是我吃過的",
        },
    )

    assert r.status_code == 200
    assert r.json()["system_value"] is None


def test_清單最新的在前(client, user, db_session):
    for text in ["第一筆", "第二筆", "第三筆"]:
        client.post("/feedback", json={**VALID, "raw_input": text})

    rows = client.get("/feedback").json()

    assert [x["raw_input"] for x in rows] == ["第三筆", "第二筆", "第一筆"]


def test_pending_only只看還沒轉成eval的(client, user, db_session):
    """累積一批之後要處理時用——已經轉過的不該再出現。"""
    client.post("/feedback", json={**VALID, "raw_input": "還沒處理"})
    client.post("/feedback", json={**VALID, "raw_input": "已經處理"})
    done = db_session.query(FeedbackReport).filter_by(raw_input="已經處理").one()
    done.converted_to_eval = True
    db_session.commit()

    rows = client.get("/feedback", params={"pending_only": True}).json()

    assert [x["raw_input"] for x in rows] == ["還沒處理"]


def test_刪除誤報(client, user, db_session):
    created = client.post("/feedback", json=VALID).json()

    assert client.delete(f"/feedback/{created['id']}").status_code == 204
    assert db_session.query(FeedbackReport).count() == 0


def test_刪除不存在的回報回404(client, user):
    r = client.delete("/feedback/999999")

    assert r.status_code == 404
    assert set(r.json()["error"]) == {"code", "message"}
