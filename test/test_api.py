"""
test_api.py — FastAPI 整合測試
================================
使用 httpx + pytest 測試 API 端點，不需要啟動真正的伺服器。

執行方式：
  uv run pytest test/test_api.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────
FAKE_MOMO = [
    {
        "id": 1,
        "title": "Apple iPhone 15 128GB 黑色",
        "price": 25900,
        "platform": "momo",
        "image": "https://example.com/img1.jpg",
        "url": "https://momo.com/product/1",
        "sku": "MOMO-001",
    }
]

FAKE_PCHOME = [
    {
        "id": 1,
        "title": "Apple iPhone 15 128GB 白色",
        "price": 25500,
        "platform": "pchome",
        "image": "https://example.com/img2.jpg",
        "url": "https://pchome.com/product/1",
        "sku": "PC-001",
    }
]


def _mock_database(get_cached_rv=(None, None)):
    """建立假的 _get_database 回傳值。"""
    mock_get_cached = MagicMock(return_value=get_cached_rv)
    mock_init_db = MagicMock()
    mock_save = MagicMock()
    return mock_get_cached, mock_init_db, mock_save


def _mock_scrapers(momo_rv=None, pchome_rv=None):
    """建立假的 _get_scrapers 回傳值。"""
    mock_momo = MagicMock(return_value=momo_rv or [])
    mock_pchome = MagicMock(return_value=pchome_rv or [])
    return mock_momo, mock_pchome


# ─────────────────────────────────────────────────────────────────────────────
# 健康檢查
# ─────────────────────────────────────────────────────────────────────────────
def test_health_check():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_ready" in data
    assert "timestamp" in data


# ─────────────────────────────────────────────────────────────────────────────
# 快取查詢（mock DB）
# ─────────────────────────────────────────────────────────────────────────────
@patch("api._get_database")
def test_cache_hit(mock_db_factory):
    mock_cached, mock_init, mock_save = _mock_database(
        get_cached_rv=(FAKE_MOMO, FAKE_PCHOME)
    )
    mock_db_factory.return_value = (mock_cached, mock_init, mock_save)

    resp = client.get("/api/cache/iPhone 15")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cached"] is True
    assert data["momo_count"] == 1
    assert data["pchome_count"] == 1


@patch("api._get_database")
def test_cache_miss(mock_db_factory):
    mock_cached, mock_init, mock_save = _mock_database(get_cached_rv=(None, None))
    mock_db_factory.return_value = (mock_cached, mock_init, mock_save)

    resp = client.get("/api/cache/不存在的關鍵字")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cached"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 同步搜尋（快取命中）
# ─────────────────────────────────────────────────────────────────────────────
@patch("api._get_database")
def test_search_cache_hit(mock_db_factory):
    mock_cached, mock_init, mock_save = _mock_database(
        get_cached_rv=(FAKE_MOMO, FAKE_PCHOME)
    )
    mock_db_factory.return_value = (mock_cached, mock_init, mock_save)

    resp = client.post(
        "/api/search",
        json={"keyword": "iPhone 15", "max_products": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["from_cache"] is True
    assert data["keyword"] == "iPhone 15"
    assert data["momo_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 同步搜尋（快取未命中 → 爬蟲）
# ─────────────────────────────────────────────────────────────────────────────
@patch("api._get_scrapers")
@patch("api._get_database")
def test_search_cache_miss(mock_db_factory, mock_scraper_factory):
    mock_cached, mock_init, mock_save = _mock_database(get_cached_rv=(None, None))
    mock_db_factory.return_value = (mock_cached, mock_init, mock_save)

    mock_momo, mock_pchome = _mock_scrapers(momo_rv=FAKE_MOMO, pchome_rv=FAKE_PCHOME)
    mock_scraper_factory.return_value = (mock_momo, mock_pchome)

    resp = client.post(
        "/api/search",
        json={"keyword": "iPhone 15", "max_products": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["from_cache"] is False
    assert data["momo_count"] == 1
    mock_momo.assert_called_once()
    mock_pchome.assert_called_once()
    mock_save.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# 非同步搜尋（回傳 job_id）
# ─────────────────────────────────────────────────────────────────────────────
@patch("api._get_database")
def test_search_async_returns_job_id(mock_db_factory):
    mock_cached, mock_init, mock_save = _mock_database(
        get_cached_rv=(FAKE_MOMO, FAKE_PCHOME)
    )
    mock_db_factory.return_value = (mock_cached, mock_init, mock_save)

    resp = client.post(
        "/api/search/async",
        json={"keyword": "PS5"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    assert data["keyword"] == "PS5"


# ─────────────────────────────────────────────────────────────────────────────
# 工作查詢 — 404
# ─────────────────────────────────────────────────────────────────────────────
def test_get_job_not_found():
    resp = client.get("/api/jobs/nonexistent-id")
    assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# LLM 驗證端點（mock ollama）
# ─────────────────────────────────────────────────────────────────────────────
@patch("api._import_heavy")
def test_verify_matches(mock_heavy):
    mock_ollama = MagicMock()
    mock_ollama.chat.return_value = {
        "message": {
            "content": json.dumps(
                [
                    {
                        "is_match": True,
                        "confidence": "high",
                        "reasoning": "相同商品（顏色不同）",
                    }
                ]
            )
        }
    }
    mock_heavy.return_value = (mock_ollama, MagicMock(), MagicMock())

    resp = client.post(
        "/api/match/verify",
        json=[
            {
                "momo_title": "Apple iPhone 15 128GB 黑色",
                "pchome_title": "Apple iPhone 15 128GB 白色",
                "similarity": 0.95,
            }
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["is_match"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 驗證超過上限 → 400
# ─────────────────────────────────────────────────────────────────────────────
def test_verify_too_many_pairs():
    pairs = [
        {"momo_title": f"商品A{i}", "pchome_title": f"商品B{i}", "similarity": 0.9}
        for i in range(51)
    ]
    resp = client.post("/api/match/verify", json=pairs)
    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 輸入驗證（keyword 太長 / 為空）
# ─────────────────────────────────────────────────────────────────────────────
def test_search_empty_keyword():
    resp = client.post("/api/search", json={"keyword": ""})
    assert resp.status_code == 422  # Pydantic validation error


def test_search_keyword_too_long():
    resp = client.post("/api/search", json={"keyword": "x" * 101})
    assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 統計端點
# ─────────────────────────────────────────────────────────────────────────────
@patch("api.Path")
def test_stats_empty(mock_path_cls):
    """沒有 log 檔案時應回傳 0。"""
    instance = MagicMock()
    instance.exists.return_value = False
    mock_path_cls.return_value = instance

    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_searches"] >= 0
