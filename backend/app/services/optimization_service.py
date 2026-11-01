"""Application service for price recommendations."""
import json
from app.services.price_optimizer import generate_sample_data
from app.services.xgb_optimizer import get_ml_optimizer


def run_optimization(params: dict) -> dict:
    competitor_prices = params.get("competitor_prices") or generate_sample_data(
        params["cost_price"] * 1.5, params.get("num_competitors", 50)
    )
    result = get_ml_optimizer().recommend(
        cost_price=params["cost_price"], fixed_costs=params.get("fixed_costs", 0),
        competitor_prices=competitor_prices, min_margin=params.get("min_margin", 5),
        max_margin=params.get("max_margin", 80), category=params.get("category", "Electronics"),
        month=params.get("month", 6), rating=params.get("rating", 4.2),
        ad_spend=params.get("ad_spend", 1000),
    )
    # XGBoost and NumPy return scalar subclasses that Flask and SQLAlchemy's
    # JSON columns cannot serialize directly.
    return json.loads(json.dumps(result, default=lambda value: value.item()))
