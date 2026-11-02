import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.schemas import OptimizeRequest


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
def test_optimize_request_rejects_invalid_values(payload):
    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate(payload)


def test_quality_defaults_to_standard_and_is_metadata():
    request = OptimizeRequest(cost_price=100)
    assert request.quality == "standard"
    assert request.model_dump()["quality"] == "standard"
