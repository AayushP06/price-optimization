from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.models.product import Optimization
from app.schemas import APIResponse

router = APIRouter(tags=["history"])


@router.get("/api/history", response_model=APIResponse, summary="List optimization history")
def get_history(
    limit: str = "50",
    product_id: str | None = None,
    db: Session = Depends(get_db),
):
    limit = min(max(int(limit), 1), 500)
    statement = select(Optimization)
    if product_id:
        statement = statement.where(Optimization.product_id == product_id)
    records = db.scalars(statement.order_by(Optimization.timestamp.desc()).limit(limit)).all()
    return {"count": len(records), "data": [record.to_dict() for record in records]}


@router.get("/api/stats", response_model=APIResponse, summary="Summarize optimization history")
def get_stats(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Optimization.id))) or 0
    average_profit = db.scalar(select(func.avg(Optimization.expected_profit))) or 0
    return {"total_optimizations": total, "average_profit": round(average_profit, 2)}


@router.get("/api/products", response_model=APIResponse, summary="List products with optimization counts")
def get_products(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Optimization.product_id, func.count(Optimization.id).label("count"))
        .group_by(Optimization.product_id)
    ).all()
    return {
        "products": [
            {"product_id": row.product_id, "optimization_count": row.count} for row in rows
        ]
    }
