"""
API Tests — run with: pytest tests/test_api.py -v
"""

import sys
import os

# Add backend/ to path so `from app import create_app` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from app import create_app


@pytest.fixture
def client():
    """Test client with an in-memory SQLite database."""
    class TestConfig:
        TESTING = True
        SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
        SQLALCHEMY_TRACK_MODIFICATIONS = False
    app = create_app(TestConfig)
    with app.test_client() as test_client:
        yield test_client


# ─────────────────────── Health ─────────────────────────────────

def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


# ─────────────────────── Sort ───────────────────────────────────

def test_sort_prices(client):
    res = client.post("/api/sort", json={"prices": [300, 100, 200, 50, 400]})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["sorted_prices"] == [50, 100, 200, 300, 400]
    assert data["algorithm"] == "Quick Sort"


def test_sort_missing_field(client):
    res = client.post("/api/sort", json={})
    assert res.status_code == 400


# ─────────────────────── Search ─────────────────────────────────

def test_search_found(client):
    res = client.post("/api/search", json={
        "sorted_prices": [100, 200, 300, 400, 500],
        "target_price": 300,
    })
    data = res.get_json()
    assert data["success"] is True
    assert data["found"] is True
    assert data["index"] == 2


def test_search_not_found(client):
    res = client.post("/api/search", json={
        "sorted_prices": [100, 200, 400, 500],
        "target_price": 300,
    })
    data = res.get_json()
    assert data["success"] is True
    assert data["found"] is False


# ─────────────────────── Optimize ───────────────────────────────

def test_optimize_basic(client):
    res = client.post("/api/optimize", json={
        "cost_price": 1000,
        "fixed_costs": 50,
        "num_competitors": 20,
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "optimal_price" in data
    assert data["optimal_price"]["price"] > 1000


def test_optimize_invalid_cost(client):
    res = client.post("/api/optimize", json={"cost_price": 0})
    assert res.status_code == 400


def test_optimize_with_competitor_prices(client):
    res = client.post("/api/optimize", json={
        "cost_price": 500,
        "competitor_prices": [600, 650, 700, 750, 800],
    })
    data = res.get_json()
    assert data["success"] is True
    assert data["competitor_stats"]["count"] == 5


@pytest.mark.parametrize("payload", [
    {"cost_price": 0}, {"cost_price": 100, "fixed_costs": -1},
    {"cost_price": 100, "min_margin": 80, "max_margin": 10},
    {"cost_price": 100, "month": 13}, {"cost_price": 100, "rating": 6},
    {"cost_price": 100, "ad_spend": -1}, {"cost_price": 100, "category": "Food"},
    {"cost_price": 100, "available_units": -1},
    {"cost_price": 100, "competitor_prices": [0]},
    {"cost_price": 100, "competitor_prices": [100] * 5001},
    {"cost_price": 100, "num_competitors": 5001},
    {"cost_price": 100, "quality": "luxury"},
])
def test_optimize_validation_rules(client, payload):
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 400
    assert response.get_json()["errors"]


def test_optimize_get_docs(client):
    response = client.get("/api/optimize")
    assert response.status_code == 200
    assert response.get_json()["method"] == "POST"


def test_missing_model_fallback_and_warning(client, monkeypatch):
    from app.services import optimization_service
    from app.services.xgb_optimizer import XGBPriceOptimizer
    monkeypatch.setattr(optimization_service, "get_ml_optimizer", lambda: XGBPriceOptimizer(model_path="missing-model.json"))
    data = client.post("/api/optimize", json={"cost_price": 100, "competitor_prices": [110, 120]}).get_json()
    assert data["method"] == "heuristic-fallback"
    assert data["warnings"]


def test_available_units_caps_sales(client):
    result = client.post("/api/optimize", json={"cost_price": 100, "competitor_prices": [110, 120], "available_units": 3}).get_json()
    assert result["optimal_price"]["expected_units_sold"] <= 3


def test_generate_sample_seed_is_reproducible():
    from app.services.price_optimizer import generate_sample_data
    assert generate_sample_data(100, 10, seed=17) == generate_sample_data(100, 10, seed=17)


def test_quality_is_metadata_only(client):
    base = {"cost_price": 100, "competitor_prices": [110, 120], "seed": 123}
    budget = client.post("/api/optimize", json={**base, "quality": "budget"}).get_json()
    premium = client.post("/api/optimize", json={**base, "quality": "premium"}).get_json()
    assert budget["optimal_price"] == premium["optimal_price"]
    assert budget["quality"] == "budget"
    assert premium["quality"] == "premium"


def test_history_endpoint(client):
    client.post("/api/optimize", json={"cost_price": 100, "competitor_prices": [120], "product_name": "Widget", "quality": "premium"})
    history = client.get("/api/history").get_json()
    assert history["count"] == 1
    assert history["data"][0]["product_name"] == "Widget"
    assert history["data"][0]["quality"] == "premium"
    assert history["data"][0]["all_prices"]
    assert history["data"][0]["competitor_stats"]["simulated"] is False
    assert history["data"][0]["min_markup_pct"] == 5
    assert history["data"][0]["max_markup_pct"] == 80


def test_admin_rejects_missing_token(client):
    import os
    os.environ.pop("ADMIN_TOKEN", None)
    assert client.get("/admin/status").status_code == 401
