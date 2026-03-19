"""
Flask routes — all API endpoints.
The broken /api/optimize is fully fixed here.
"""

import json
import os
import time
from datetime import datetime

from flask import Blueprint, jsonify, request, send_file

from app.database.db import db
from app.models.product import Optimization
from app.services.price_optimizer import (
    PriceOptimizerAPI,
    generate_sample_data,
    quick_sort,
    binary_search,
)
from app.services.xgb_optimizer import get_ml_optimizer

api_bp = Blueprint("api", __name__)

_HISTORY_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "stored_data.json"
)


def _store_request_data(input_data: dict, result_data: dict):
    """Append a run to stored_data.json for simple JSON-based history."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": input_data,
        "result": result_data,
    }
    try:
        with open(_HISTORY_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        data = []
    data.append(entry)
    with open(_HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=4)


@api_bp.route("/")
def index():
    html_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "price-optimization-website.html"
    )
    return send_file(os.path.abspath(html_path))


@api_bp.route("/api/optimize", methods=["GET", "POST"])
def optimize_price():
    if request.method == "GET":
        return jsonify({
            "endpoint": "/api/optimize",
            "method": "POST",
            "description": "Get optimal price recommendation",
        }), 200

    data = request.json or {}

    try:
        cost_price = float(data.get("cost_price", 0))
        if cost_price <= 0:
            return jsonify({"success": False, "error": "cost_price must be > 0"}), 400

        fixed_costs = float(data.get("fixed_costs", 0))
        available_units = data.get("available_units")
        if available_units is not None:
            available_units = int(available_units)

        min_margin = float(data.get("min_margin", 15))
        max_margin = float(data.get("max_margin", 35))
        
        category = data.get("category", "Electronics")
        month = int(data.get("month", 6))
        rating = float(data.get("rating", 4.2))
        ad_spend = float(data.get("ad_spend", 1000.0))

        competitor_prices = data.get("competitor_prices", [])
        if not competitor_prices:
            num_competitors = int(data.get("num_competitors", 50))
            base_price = cost_price * 1.5
            competitor_prices = generate_sample_data(base_price, num_competitors)

        optimizer = get_ml_optimizer()
        result = optimizer.recommend(
            cost_price=cost_price,
            fixed_costs=fixed_costs,
            competitor_prices=competitor_prices,
            min_margin=min_margin,
            max_margin=max_margin,
            category=category,
            month=month,
            rating=rating,
            ad_spend=ad_spend
        )
        try:
            _store_request_data(data, result)
        except Exception:
            pass
        try:
            opt = Optimization(
                product_id=data.get("product_id", f"product_{int(time.time())}"),
                cost_price=cost_price,
                fixed_costs=fixed_costs,
                available_units=available_units,
                recommended_price=result["optimal_price"]["price"],
                expected_profit=result["optimal_price"]["expected_profit"],
                competitor_prices=result["competitor_prices"],
                competitor_stats=result["competitor_stats"],
                price_analysis=result["all_prices"],
            )
            db.session.add(opt)
            db.session.commit()
            result["id"] = opt.id
        except Exception as e:
            db.session.rollback()
            print(f"Database error: {e}")

        return jsonify({"success": True, **result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/optimize/ml", methods=["GET", "POST"])
def optimize_price_ml():
    """XGBoost-powered price recommendation using the trained model."""
    if request.method == "GET":
        return jsonify({
            "endpoint":    "/api/optimize/ml",
            "method":      "POST",
            "description": "XGBoost price recommendation",
            "model":       "XGBoost Regressor",
        }), 200

    data = request.json or {}
    try:
        cost_price = float(data.get("cost_price", 0))
        if cost_price <= 0:
            return jsonify({"success": False, "error": "cost_price must be > 0"}), 400

        fixed_costs       = float(data.get("fixed_costs", 0))
        min_margin        = float(data.get("min_margin", 5))
        max_margin        = float(data.get("max_margin", 80))
        
        category = data.get("category", "Electronics")
        month = int(data.get("month", 6))
        rating = float(data.get("rating", 4.2))
        ad_spend = float(data.get("ad_spend", 1000.0))
        
        competitor_prices = data.get("competitor_prices", [])

        if not competitor_prices:
            num_competitors   = int(data.get("num_competitors", 50))
            base_price        = cost_price * 1.5
            competitor_prices = generate_sample_data(base_price, num_competitors)

        optimizer = get_ml_optimizer()
        result    = optimizer.recommend(
            cost_price=cost_price,
            fixed_costs=fixed_costs,
            competitor_prices=competitor_prices,
            min_margin=min_margin,
            max_margin=max_margin,
            category=category,
            month=month,
            rating=rating,
            ad_spend=ad_spend
        )

        try:
            _store_request_data(data, result)
        except Exception:
            pass

        return jsonify({"success": True, **result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500




@api_bp.route("/api/generate-sample", methods=["POST"])
def generate_sample():
    try:
        data = request.json or {}
        base_price = float(data.get("base_price", 1500))
        num_competitors = int(data.get("num_competitors", 50))
        variation = float(data.get("variation", 0.3))
        prices = generate_sample_data(base_price, num_competitors, variation)
        return jsonify({"success": True, "prices": prices})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/sort", methods=["POST"])
def sort_prices():
    data = request.json
    if not data or "prices" not in data:
        return jsonify({"error": "Missing required field: prices"}), 400
    prices = data["prices"].copy()
    start = time.time()
    quick_sort(prices, 0, len(prices) - 1)
    return jsonify({
        "success": True,
        "sorted_prices": prices,
        "sort_time": round(time.time() - start, 4),
        "algorithm": "Quick Sort",
    })


@api_bp.route("/api/search", methods=["POST"])
def search_price():
    data = request.json
    if not data or "sorted_prices" not in data or "target_price" not in data:
        return jsonify({"error": "Missing required fields"}), 400
    sorted_prices = data["sorted_prices"]
    target = float(data["target_price"])
    start = time.time()
    idx = binary_search(sorted_prices, target)
    return jsonify({
        "success": True,
        "index": idx,
        "search_time": round(time.time() - start, 6),
        "found": idx < len(sorted_prices) and sorted_prices[idx] == target,
    })

@api_bp.route("/api/health")
def health():
    try:
        db.session.execute(db.text("SELECT 1"))
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@api_bp.route("/api/history", methods=["GET"])
def get_history():
    try:
        limit = int(request.args.get("limit", 50))
        product_id = request.args.get("product_id")
        query = Optimization.query
        if product_id:
            query = query.filter_by(product_id=product_id)
        records = query.order_by(Optimization.timestamp.desc()).limit(limit).all()
        return jsonify({"count": len(records), "data": [r.to_dict() for r in records]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/stats", methods=["GET"])
def get_stats():
    try:
        total = Optimization.query.count()
        avg_profit = db.session.query(
            db.func.avg(Optimization.expected_profit)
        ).scalar() or 0
        return jsonify({
            "total_optimizations": total,
            "average_profit": round(avg_profit, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/products", methods=["GET"])
def get_products():
    try:
        products = db.session.query(
            Optimization.product_id,
            db.func.count(Optimization.id).label("count"),
        ).group_by(Optimization.product_id).all()
        return jsonify({
            "products": [
                {"product_id": p.product_id, "optimization_count": p.count}
                for p in products
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/view-history", methods=["GET"])
def view_history():
    try:
        with open(_HISTORY_FILE, "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "history": data})
    except Exception:
        return jsonify({"success": False, "error": "No data found"})

@api_bp.route("/admin/status")
def admin_status():
    try:
        count = Optimization.query.count()
        return f"""
        <html>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1 style="color: green;">✅ Database Connected!</h1>
            <p style="font-size: 24px;">Total records: {count}</p>
            <a href="/api/health">View Health JSON</a> |
            <a href="/api/history">View History</a>
        </body>
        </html>
        """
    except Exception as e:
        return f"""
        <html>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1 style="color: red;">❌ Database Error</h1>
            <p>{str(e)}</p>
        </body>
        </html>
        """, 500
