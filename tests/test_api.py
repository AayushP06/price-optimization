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
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    with app.test_client() as client:
        yield client


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
