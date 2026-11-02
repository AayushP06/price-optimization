"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.services.xgb_optimizer import XGBPriceOptimizer

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        optimizer = XGBPriceOptimizer()
        if not optimizer.is_ml_ready:
            logger.warning("XGBoost model unavailable; heuristic fallback will be used")
    except Exception:
        logger.warning("Unable to load XGBoost optimizer; using heuristic fallback", exc_info=True)
        optimizer = XGBPriceOptimizer(model_path="")
    app.state.optimizer = optimizer
    yield


app = FastAPI(
    title="Price Optimizer API",
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)
