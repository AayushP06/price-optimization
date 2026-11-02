import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.config import Settings
from app.database import db as database
from app.models.product import Optimization
from app import main


def test_settings_rewrite_legacy_postgres_url_and_docs_defaults():
    assert Settings(database_url="postgres://user:pass@host/db").database_url == "postgresql://user:pass@host/db"
    assert Settings(env="development").enable_docs is True
    assert Settings(env="production").enable_docs is False
    assert Settings(env="production", enable_docs=True).enable_docs is True


def test_engine_uses_pool_health_settings():
    assert database.engine.pool._pre_ping is True
    assert database.engine.pool._recycle == 300


def test_optimization_model_registers_the_legacy_table_and_columns():
    table = Optimization.__table__
    assert table.name == "optimizations"
    assert set(table.columns.keys()) == {
        "id", "product_id", "cost_price", "fixed_costs", "available_units", "category", "month",
        "rating", "ad_spend", "method", "product_name", "quality", "recommended_price",
        "expected_profit", "expected_units_sold", "competitor_prices", "competitor_stats",
        "price_analysis", "timestamp", "duration_ms",
    }
    assert {index.name for index in table.indexes} == {"ix_optimizations_product_id", "ix_optimizations_timestamp"}


def test_database_dependency_always_closes_its_session(monkeypatch):
    class FakeSession:
        closed = False

        def close(self):
            self.closed = True

    session = FakeSession()
    monkeypatch.setattr(database, "SessionLocal", lambda: session)
    dependency = database.get_db()
    assert next(dependency) is session
    with pytest.raises(StopIteration):
        next(dependency)
    assert session.closed is True


