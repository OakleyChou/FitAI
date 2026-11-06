"""food_logs CRUD 的整合測試。

這組端點是專案裡最早寫的，一直沒有測試。它們的接縫在
**HTTP ↔ Pydantic ↔ SQLAlchemy ↔ 資料庫**，單元測試涵蓋不到：
欄位名對不上、response_model 漏欄位、commit 忘了寫，
三種都是「每個函式看起來都對」但整條路走不通。

LLM 在這裡是假的：新增紀錄會呼叫 extract_food_log 解析自然語言，
但這層要測的是「解析結果有沒有正確存進去並回傳」，不是解析得準不準。
"""

import pytest

from app.llm import FoodItem, FoodLogExtraction
from app.models import FoodLog as FoodLogModel
from app.models import FoodLogItem as FoodLogItemModel
from app.routers import food_logs as food_logs_module
from app.schemas import InputType

EXTRACTION = FoodLogExtraction(
    items=[
        FoodItem(
            name="雞胸肉",
            portion_g=100.0,
            calories=165,
            protein_g=31,
            carbs_g=0,
            fat_g=3.6,
            # 這一項有食藥署資料庫佐證。擷取改成兩階段之後，
            # 每一項都要記錄數值從哪裡來（見 app/extraction.py）。
            grounded=True,
            source_ref="I0410101",
            db_name="雞里肌肉",
        )
    ],
    food_description="雞胸肉 100 公克",
    calories=165,
    protein_g=31,
    carbs_g=0,
    fat_g=3.6,
)


@pytest.fixture
def fake_llm(monkeypatch):
    """把 extract_food_log 換成固定回傳值。

    patch 的是 food_logs 模組裡的名字而不是 llm 模組裡的——
    `from app.llm import extract_food_log` 會在 food_logs 模組建立自己的名字，
    改 llm.extract_food_log 對它沒有影響。**patch 使用它的地方，不是定義它的地方。**
    """
    monkeypatch.setattr(food_logs_module, "extract_food_log", lambda raw_text: EXTRACTION)


def _make_log(db_session, user_id: int = 1, **overrides) -> FoodLogModel:
    from datetime import datetime, timezone

    log = FoodLogModel(
        user_id=user_id,
        logged_at=datetime.now(timezone.utc),
        calories=overrides.get("calories", 500.0),
        protein_g=overrides.get("protein_g", 30.0),
        carbs_g=overrides.get("carbs_g", 50.0),
        fat_g=overrides.get("fat_g", 20.0),
        raw_input_text=overrides.get("raw_input_text", "測試餐"),
        input_type=InputType.text,
    )
    db_session.add(log)
    db_session.commit()
    db_session.refresh(log)
    return log


# ── 新增 ──────────────────────────────────────────────────────


def test_新增紀錄會存下LLM算出的營養素(client, user, db_session, fake_llm):
    r = client.post("/food_logs", json={"user_id": 1, "raw_text": "雞胸肉 100 公克"})

    assert r.status_code == 200
    body = r.json()
    assert body["calories"] == 165
    assert body["protein_g"] == 31
    # id 由資料庫產生，不是客戶端給的——這是 db.refresh 那一行在做的事，
    # 漏掉的話回應裡的 id 會是 None。
    assert body["id"] is not None

    # 回應正確不等於真的存進去了：少一個 db.commit() 兩者就會分岔。
    assert db_session.query(FoodLogModel).count() == 1


def test_回應帶上逐項明細與佐證狀態(client, user, db_session, fake_llm):
    """前端要靠 grounded 區分「資料庫數值」與「LLM 估的」。

    不給這個欄位的話，可靠的數字和猜測在畫面上看起來一模一樣，
    使用者就不知道哪一項需要核對。
    """
    r = client.post("/food_logs", json={"user_id": 1, "raw_text": "雞胸肉 100 公克"})

    body = r.json()
    assert body["grounded_count"] == 1
    item = body["items"][0]
    assert item["grounded"] is True
    assert item["source_ref"] == "I0410101"
    # db_name 要有：使用者說「雞胸肉」、資料庫叫「雞里肌肉」，
    # 不給他看正式名稱就無法判斷配對對不對。
    assert item["db_name"] == "雞里肌肉"
    assert item["portion_g"] == 100.0


def test_明細會落地到資料庫(client, user, db_session, fake_llm):
    """明細存進 food_log_items，合計仍然留在 food_logs。

    這個測試 2026-11-06 之前斷言的是相反的事（「明細不落地」），
    理由寫的是「那需要加欄位，而這個專案用 create_all 沒有 migration」。
    那個理由只對一半：create_all 不會修改既有表，但**會建立不存在的表**，
    所以新增子表不需要 migration。一個站不住腳的理由被寫成註解之後，
    就沒有人再質疑它——留著這段歷史是為了記住這件事。
    """
    client.post("/food_logs", json={"user_id": 1, "raw_text": "雞胸肉 100 公克"})

    row = db_session.query(FoodLogModel).one()
    # 合計仍然是權威值，不改成每次從明細加總（見 models.py 的說明）
    assert row.calories == 165

    assert len(row.items) == 1
    item = row.items[0]
    assert item.name == "雞胸肉"
    assert item.portion_g == 100.0
    assert item.calories == 165
    # 佐證狀態要落地，不能靠 source_ref is not None 推導：
    # 名稱守門可能在留著編號的情況下改用估算值（見 models.py）。
    assert item.grounded is True
    assert item.source_ref == "I0410101"
    assert item.db_name == "雞里肌肉"


def test_刪掉紀錄會一起刪掉明細(client, user, db_session, fake_llm):
    """孤兒明細會讓之後每一次「哪樣食物最高」都多算一筆查不到來源的資料。

    ORM 的 cascade 跟資料庫的 ON DELETE CASCADE 兩層都設了。
    這個測試跑在 SQLite 上，而 SQLite 預設**不強制外鍵**——
    所以它驗的是 ORM 那一層。只靠資料庫的話，這裡會通過而 production 不會。
    """
    log_id = client.post(
        "/food_logs", json={"user_id": 1, "raw_text": "雞胸肉 100 公克"}
    ).json()["id"]
    assert db_session.query(FoodLogItemModel).count() == 1

    assert client.delete(f"/food_logs/{log_id}").status_code == 204

    assert db_session.query(FoodLogModel).count() == 0
    assert db_session.query(FoodLogItemModel).count() == 0


def test_單筆查詢會帶出明細(client, user, db_session, fake_llm):
    """清單端點刻意不帶明細，單筆要帶——要核對數字的人會點進單一筆。"""
    log_id = client.post(
        "/food_logs", json={"user_id": 1, "raw_text": "雞胸肉 100 公克"}
    ).json()["id"]

    detail = client.get(f"/food_logs/{log_id}").json()
    assert [i["name"] for i in detail["items"]] == ["雞胸肉"]

    # 清單不帶 items，避免一次查詢把上百筆子資料列一起拖出來
    assert "items" not in client.get("/food_logs").json()[0]


def test_原始輸入會一起保留(client, user, db_session, fake_llm):
    """LLM 解析可能出錯，留著原文才有辦法事後修正或重跑。"""
    r = client.post("/food_logs", json={"user_id": 1, "raw_text": "雞胸肉 100 公克"})

    assert r.json()["raw_input_text"] == "雞胸肉 100 公克"
    assert r.json()["input_type"] == "text"


def test_不存在的使用者回404且不呼叫LLM(client, db_session, monkeypatch):
    """這是整合測試抓到的 bug（2026-09-22）。

    原本沒有這個檢查，user_id 不存在時外鍵約束會在 commit 才爆，
    回給客戶端的是 500——但這是客戶端給錯資料，500 等於叫對方重試，
    而重試永遠不會成功。而且那次 LLM 呼叫白花了錢。

    這裡刻意**不用 fake_llm**，改成一被呼叫就炸——
    「便宜的驗證要在昂貴的呼叫前面」這件事只有這樣才測得到。
    """

    def boom(raw_text):
        raise AssertionError("驗證失敗時不該呼叫 LLM")

    monkeypatch.setattr(food_logs_module, "extract_food_log", boom)

    r = client.post("/food_logs", json={"user_id": 999, "raw_text": "雞胸肉"})

    assert r.status_code == 404
    assert r.json()["error"]["code"] == "http_404"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"raw_text": "雞胸肉"}, id="缺user_id"),
        pytest.param({"user_id": 1}, id="缺raw_text"),
        pytest.param({"user_id": "abc", "raw_text": "雞胸肉"}, id="user_id型別錯"),
    ],
)
def test_不合法的新增請求回422(client, payload, fake_llm):
    r = client.post("/food_logs", json=payload)

    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


# ── 查詢 ──────────────────────────────────────────────────────


def test_可以用user_id篩選清單(client, user, db_session):
    _make_log(db_session, user_id=1)
    _make_log(db_session, user_id=1)

    assert len(client.get("/food_logs").json()) == 2
    assert len(client.get("/food_logs", params={"user_id": 1}).json()) == 2
    assert client.get("/food_logs", params={"user_id": 999}).json() == []


def test_查詢單筆(client, user, db_session):
    log = _make_log(db_session, calories=777.0)

    r = client.get(f"/food_logs/{log.id}")

    assert r.status_code == 200
    assert r.json()["calories"] == 777.0


# ── 更新 ──────────────────────────────────────────────────────


def test_更新只改有傳的欄位(client, user, db_session):
    """PUT 用的是 exclude_unset，沒傳的欄位要保持原值。

    沒有這個行為的話，只想修熱量的使用者會把其他三個營養素清成 null。
    """
    log = _make_log(db_session, calories=500.0, protein_g=30.0)

    r = client.put(f"/food_logs/{log.id}", json={"calories": 600.0})

    assert r.status_code == 200
    assert r.json()["calories"] == 600.0
    assert r.json()["protein_g"] == 30.0


def test_更新不接受負數(client, user, db_session):
    log = _make_log(db_session)

    r = client.put(f"/food_logs/{log.id}", json={"calories": -100})

    assert r.status_code == 422
    # 驗證失敗不能留下半套資料
    db_session.refresh(log)
    assert log.calories == 500.0


# ── 刪除 ──────────────────────────────────────────────────────


def test_刪除後查不到(client, user, db_session):
    log = _make_log(db_session)

    assert client.delete(f"/food_logs/{log.id}").status_code == 204
    assert client.get(f"/food_logs/{log.id}").status_code == 404
    assert db_session.query(FoodLogModel).count() == 0


# ── 找不到 ────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["get", "put", "delete"])
def test_不存在的紀錄一律回404且格式一致(client, user, method):
    kwargs = {"json": {"calories": 1.0}} if method == "put" else {}

    r = getattr(client, method)("/food_logs/999999", **kwargs)

    assert r.status_code == 404
    body = r.json()
    # 既有端點用的是 fastapi.HTTPException，但全域 handler 會把它
    # 轉成跟其他錯誤一樣的形狀——前端只要寫一條錯誤處理路徑。
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == "http_404"
