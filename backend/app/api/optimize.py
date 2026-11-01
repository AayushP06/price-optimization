import logging
import uuid
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from app.database.db import db
from app.models.product import Optimization
from app.schemas import OptimizeRequest, validation_errors
from app.services.optimization_service import run_optimization

bp = Blueprint("optimize", __name__)
logger = logging.getLogger(__name__)


@bp.route("/api/optimize", methods=["GET", "POST"])
def optimize():
    if request.method == "GET":
        return jsonify({"endpoint": "/api/optimize", "method": "POST", "description": "Get optimal price recommendation"})
    data = request.get_json(silent=True)
    try:
        params = OptimizeRequest.model_validate(data or {}).model_dump()
    except ValidationError as exc:
        return jsonify({"success": False, "errors": validation_errors(exc)}), 400
    try:
        result = run_optimization(params)
        try:
            opt = Optimization(
                product_id=params.get("product_id") or f"product_{int(datetime.now(timezone.utc).timestamp())}",
                cost_price=params["cost_price"], fixed_costs=params["fixed_costs"],
                recommended_price=result["optimal_price"]["price"],
                expected_profit=result["optimal_price"]["expected_profit"],
                expected_units_sold=result["optimal_price"].get("expected_units_sold"),
                competitor_prices=result["competitor_prices"], competitor_stats={**result["competitor_stats"], "simulated": result["simulated_competitors"]},
                price_analysis=result["all_prices"],
                category=params["category"], month=params["month"], rating=params["rating"],
                ad_spend=params["ad_spend"], method=result["method"],
                available_units=params["available_units"], product_name=params["product_name"],
                quality=params["quality"],
            )
            db.session.add(opt)
            db.session.commit()
            result["id"] = opt.id
        except Exception:
            db.session.rollback()
            logger.exception("Unable to persist optimization history")
        return jsonify({"success": True, **result})
    except Exception:
        error_id = str(uuid.uuid4())
        logger.exception("Optimization failed; error_id=%s", error_id)
        return jsonify({"success": False, "error": "Optimization failed", "error_id": error_id}), 500
