import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import main
from app.api.deps import get_optimizer
from app.core.config import get_settings
from app.database.db import Base, get_db
from app.models.product import Optimization


@pytest.fixture
def client(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)

    def override_db():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    main.app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(get_settings(), "admin_token", "golden-token")
    with TestClient(main.app) as test_client:
        yield test_client
    main.app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)
    engine.dispose()


def _normalize(value):
    if isinstance(value, dict):
        return {
            key: _normalize(item)
            for key, item in value.items()
            if key not in {"id", "timestamp"}
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def test_golden_flask_contract(client):
    fixture_path = Path(__file__).parent / "golden" / "flask_api_contract.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    headers = {"Authorization": "Bearer golden-token"}
    for case in fixture["requests"]:
        if case["name"] == "optimize_get_docs":
            # Intentional change: endpoint help moved to FastAPI OpenAPI docs.
            continue
        request_payload = case.get("request")
        if case["name"] == "generate_sample":
            # The Phase 0 fixture was captured with the sample generator seeded at 42.
            request_payload = {**request_payload, "seed": 42}
        response = client.request(
            case["method"],
            case["path"],
            json=request_payload,
            headers=headers if case["name"] == "admin_status" else None,
        )
        assert response.status_code == case["status"], case["name"]
        if case["name"] == "admin_status":
            assert response.json() == {"status": "Database Connected", "total_records": 1}
            continue
        actual = response.json()
        if case["name"] in {"optimize_invalid", "generate_sample_invalid"}:
            assert actual["success"] is False
            assert "fields" in actual
            continue
        if case["name"] == "sort_invalid":
            assert actual["success"] is False
            assert "fields" in actual
            continue
        if case["name"] == "history":
            expected = case["response"]
            assert actual["count"] == expected["count"]
            assert len(actual["data"]) == len(expected["data"])
            actual = _normalize(actual)
            expected = _normalize(expected)
            assert actual == expected, case["name"]
            continue
        if case["name"] in {"sort", "search_found", "search_missing"}:
            actual["sort_time"] = 0 if "sort_time" in actual else actual.get("sort_time")
            actual["search_time"] = 0 if "search_time" in actual else actual.get("search_time")
            expected = dict(case["response"])
            if "sort_time" in expected:
                expected["sort_time"] = 0
            if "search_time" in expected:
                expected["search_time"] = 0
            assert actual == expected, case["name"]
            continue
        assert _normalize(actual) == _normalize(case["response"]), case["name"]


@pytest.mark.parametrize(
    "payload",
    [
        {"cost_price": 0},
        {"cost_price": 100, "fixed_costs": -1},
        {"cost_price": 100, "min_margin": 80, "max_margin": 10},
        {"cost_price": 100, "month": 13},
        {"cost_price": 100, "rating": 6},
        {"cost_price": 100, "ad_spend": -1},
        {"cost_price": 100, "category": "Food"},
        {"cost_price": 100, "available_units": -1},
        {"cost_price": 100, "competitor_prices": [0]},
        {"cost_price": 100, "competitor_prices": [100] * 5001},
        {"cost_price": 100, "num_competitors": 5001},
        {"cost_price": 100, "product_name": "x" * 101},
        {"cost_price": 100, "quality": "luxury"},
    ],
)
def test_optimize_validation_uses_400_field_envelope(client, payload):
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["fields"]


def test_quality_is_metadata_only_with_fixed_seed(client):
    payload = {"cost_price": 100, "competitor_prices": [110, 120], "seed": 123}
    budget = client.post("/api/optimize", json={**payload, "quality": "budget"}).json()
    premium = client.post("/api/optimize", json={**payload, "quality": "premium"}).json()
    assert budget["optimal_price"] == premium["optimal_price"]
    assert budget["quality"] == "budget"
    assert premium["quality"] == "premium"
    history = client.get("/api/history").json()
    assert {row["quality"] for row in history["data"]} == {"premium", "budget"}


def test_missing_model_falls_back_to_heuristic(client):
    from app.services.xgb_optimizer import XGBPriceOptimizer

    # Override the lifespan-loaded optimizer with the production fallback implementation.
    fallback = XGBPriceOptimizer(model_path="missing-model.json")
    client.app.dependency_overrides[get_optimizer] = lambda: fallback
    response = client.post("/api/optimize", json={"cost_price": 100, "competitor_prices": [110, 120]})
    assert response.json()["method"] == "heuristic-fallback"


def test_unknown_api_is_json_404(client):
    response = client.get("/api/unknown")
    assert response.status_code == 404
    assert response.json() == {"success": False, "error": "Not found"}


def test_spa_route_fallback_returns_index_html(client, monkeypatch, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>SPA</html>", encoding="utf-8")
    monkeypatch.setattr(main, "FRONTEND_DIST", dist)
    response = client.get("/dashboard/history")
    assert response.status_code == 200
    assert response.text == "<html>SPA</html>"


def test_admin_status_rejects_missing_token(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "admin_token", None)
    response = client.get("/admin/status")
    assert response.status_code == 401
    assert response.json()["success"] is False


def test_docs_can_be_hidden(monkeypatch):
    hidden_app = main.build_app(main.Settings(env="production", enable_docs=False))
    with TestClient(hidden_app) as hidden_client:
        assert hidden_client.get("/docs").status_code == 404
