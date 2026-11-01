"""Application service for price recommendations."""
import json
from app.services.price_optimizer import generate_sample_data
from app.services.xgb_optimizer import get_ml_optimizer


def run_optimization(params: dict) -> dict:
    competitor_prices = params.get("competitor_prices") or generate_sample_data(
        params["cost_price"] * 1.5, params.get("num_competitors", 50), seed=params.get("seed")
    )
    result = get_ml_optimizer().recommend(
        cost_price=params["cost_price"], fixed_costs=params.get("fixed_costs", 0),
        competitor_prices=competitor_prices, min_margin=params.get("min_markup_pct", 5),
        max_margin=params.get("max_markup_pct", 80), category=params.get("category", "Electronics"),
        month=params.get("month", 6), rating=params.get("rating", 4.2),
        ad_spend=params.get("ad_spend", 1000), available_units=params.get("available_units"),
    )
    # XGBoost and NumPy return scalar subclasses that Flask and SQLAlchemy's
    # JSON columns cannot serialize directly.
    result = json.loads(json.dumps(result, default=lambda value: value.item()))
    result["quality"] = params.get("quality", "standard")
    result["product_name"] = params.get("product_name") or "Product"
    result["min_markup_pct"] = params.get("min_markup_pct", 5)
    result["max_markup_pct"] = params.get("max_markup_pct", 80)
    result["min_margin"] = result["min_markup_pct"]
    result["max_margin"] = result["max_markup_pct"]
    price = result["optimal_price"]["price"]
    result["optimal_price"]["markup_pct"] = round((price - params["cost_price"]) / params["cost_price"] * 100, 2)
    result["optimal_price"]["margin_pct"] = result["optimal_price"]["margin"]
    result["optimal_price"]["markup"] = result["optimal_price"]["markup_pct"]
    for point in result["all_prices"]:
        point["margin_pct"] = round(max(0, point["price"] - params["cost_price"] - params.get("fixed_costs", 0)) / point["price"] * 100, 2)
    result["warnings"] = []
    if result["method"] == "heuristic-fallback":
        result["warnings"].append("The XGBoost model is unavailable; a heuristic fallback was used.")
    if not params.get("competitor_prices"):
        result["warnings"].append("Competitor prices were generated for simulation.")
    if price in (result["all_prices"][0]["price"], result["all_prices"][-1]["price"]):
        result["warnings"].append("The recommended price is at the edge of the requested markup range.")
    result["simulated_competitors"] = not bool(params.get("competitor_prices"))
    return result
