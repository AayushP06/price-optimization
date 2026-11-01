import logging
import uuid

from flask import Blueprint, jsonify, request
from app.database.db import db
from app.models.product import Optimization

bp = Blueprint("history", __name__)
logger = logging.getLogger(__name__)


def _server_error():
    error_id = str(uuid.uuid4())
    logger.exception("History request failed; error_id=%s", error_id)
    return jsonify({"error": "Request failed", "error_id": error_id}), 500


@bp.get("/api/history")
def get_history():
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 500)
        query = Optimization.query
        if request.args.get("product_id"):
            query = query.filter_by(product_id=request.args["product_id"])
        records = query.order_by(Optimization.timestamp.desc()).limit(limit).all()
        return jsonify({"count": len(records), "data": [record.to_dict() for record in records]})
    except Exception:
        return _server_error()


@bp.get("/api/stats")
def get_stats():
    try:
        total = Optimization.query.count()
        avg_profit = db.session.query(db.func.avg(Optimization.expected_profit)).scalar() or 0
        return jsonify({"total_optimizations": total, "average_profit": round(avg_profit, 2)})
    except Exception:
        return _server_error()


@bp.get("/api/products")
def get_products():
    try:
        products = db.session.query(Optimization.product_id, db.func.count(Optimization.id).label("count")).group_by(Optimization.product_id).all()
        return jsonify({"products": [{"product_id": item.product_id, "optimization_count": item.count} for item in products]})
    except Exception:
        return _server_error()
