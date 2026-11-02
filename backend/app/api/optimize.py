import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_optimizer
from app.database.db import get_db
from app.models.product import Optimization
from app.schemas import APIResponse, OptimizeRequest
from app.services.optimization_service import run_optimization

router = APIRouter(tags=["optimization"])
logger = logging.getLogger(__name__)


@router.post("/api/optimize", response_model=APIResponse, summary="Recommend a selling price")
def optimize(
    payload: OptimizeRequest,
    db: Session = Depends(get_db),
    optimizer=Depends(get_optimizer),
):
    params = payload.model_dump()
    result = run_optimization(params, optimizer=optimizer)
    try:
        record = Optimization(
                product_id=params.get("product_id") or f"product_{int(datetime.now(timezone.utc).timestamp())}",
                cost_price=params["cost_price"], fixed_costs=params["fixed_costs"],
                recommended_price=result["optimal_price"]["price"],
                expected_profit=result["optimal_price"]["expected_profit"],
                expected_units_sold=result["optimal_price"].get("expected_units_sold"),
                competitor_prices=result["competitor_prices"],
                competitor_stats={**result["competitor_stats"], "simulated": result["simulated_competitors"]},
                price_analysis=result["all_prices"], category=params["category"], month=params["month"],
                rating=params["rating"], ad_spend=params["ad_spend"], method=result["method"],
                available_units=params["available_units"], product_name=params["product_name"],
                quality=params["quality"],
            )
        db.add(record)
        db.commit()
        result["id"] = record.id
    except Exception:
        db.rollback()
        logger.exception("Unable to persist optimization history")
    return {"success": True, **result}
