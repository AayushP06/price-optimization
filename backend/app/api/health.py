from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import text, select, func
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.db import get_db
from app.models.product import Optimization
from app.schemas import APIResponse

router = APIRouter(tags=["system"])


@router.get("/api/health", response_model=APIResponse, summary="Check API and database health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {
        "status": "healthy",
        "database": "connected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/admin/status", response_model=APIResponse, summary="Check protected database status")
def admin_status(authorization: str | None = Header(default=None), db: Session = Depends(get_db)):
    token = get_settings().admin_token
    if not token or authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    count = db.scalar(select(func.count(Optimization.id))) or 0
    return {"status": "Database Connected", "total_records": count}
