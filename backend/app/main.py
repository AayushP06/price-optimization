"""FastAPI application entry point."""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import health, history, optimize, utils
from app.core.config import Settings, get_settings
from app.schemas import validation_errors
from app.services.xgb_optimizer import XGBPriceOptimizer

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


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


def build_app(settings: Settings) -> FastAPI:
    app = FastAPI(
        title="Price Optimizer API",
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    app.include_router(optimize.router)
    app.include_router(utils.router)
    app.include_router(history.router)
    app.include_router(health.router)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        fields = validation_errors_from_fastapi(exc)
        return JSONResponse(status_code=400, content={"success": False, "error": "Validation failed", "fields": fields})

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        detail = exc.detail
        fields = detail.get("fields") if isinstance(detail, dict) else None
        error = (detail.get("error") or "Validation failed") if isinstance(detail, dict) else detail
        body = {"success": False, "error": str(error)}
        if fields:
            body["fields"] = fields
        return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        error_id = str(uuid.uuid4())
        logger.exception("Unhandled API error; error_id=%s", error_id, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error", "error_id": error_id},
        )

    if FRONTEND_DIST.is_dir():
        assets_dir = FRONTEND_DIST / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str):
        if path == "api" or path.startswith(("api/", "docs/", "redoc/")) or path in {"docs", "redoc", "openapi.json"}:
            return JSONResponse(status_code=404, content={"success": False, "error": "Not found"})
        index_file = FRONTEND_DIST / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return JSONResponse(status_code=404, content={"success": False, "error": "Not found"})

    return app


def validation_errors_from_fastapi(exc: RequestValidationError) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for issue in exc.errors():
        path = ".".join(str(part) for part in issue["loc"] if part not in {"body", "query", "path", "header"})
        path = path or "request"
        message = issue["msg"].replace("Value error, ", "")
        fields.setdefault(path, []).append(message)
    return fields


app = build_app(get_settings())
