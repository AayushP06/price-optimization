import logging
import time

from flask import Blueprint, jsonify, request
from app.services.price_optimizer import binary_search, generate_sample_data, quick_sort

bp = Blueprint("utils", __name__)
logger = logging.getLogger(__name__)


@bp.post("/api/generate-sample")
def generate_sample():
    data = request.get_json(silent=True) or {}
    try:
        prices = generate_sample_data(float(data.get("base_price", 1500)), int(data.get("num_competitors", 50)), float(data.get("variation", .3)))
        return jsonify({"success": True, "prices": prices})
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid sample data parameters"}), 400


@bp.post("/api/sort")
def sort_prices():
    data = request.get_json(silent=True) or {}
    prices = data.get("prices")
    if not isinstance(prices, list):
        return jsonify({"success": False, "errors": {"prices": ["must be a list"]}}), 400
    if len(prices) > 5000:
        return jsonify({"success": False, "errors": {"prices": ["must contain at most 5000 items"]}}), 400
    try:
        prices = [float(p) for p in prices]
    except (TypeError, ValueError):
        return jsonify({"success": False, "errors": {"prices": ["must contain numbers"]}}), 400
    start = time.time()
    quick_sort(prices, 0, len(prices) - 1)
    return jsonify({"success": True, "sorted_prices": prices, "sort_time": round(time.time() - start, 4), "algorithm": "Quick Sort"})


@bp.post("/api/search")
def search_price():
    data = request.get_json(silent=True) or {}
    if not isinstance(data.get("sorted_prices"), list) or "target_price" not in data:
        return jsonify({"success": False, "error": "Missing required fields"}), 400
    try:
        values = [float(v) for v in data["sorted_prices"]]
        target = float(data["target_price"])
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Prices must be numeric"}), 400
    start = time.time()
    index = binary_search(values, target)
    return jsonify({"success": True, "index": index, "search_time": round(time.time() - start, 6), "found": index < len(values) and values[index] == target})
