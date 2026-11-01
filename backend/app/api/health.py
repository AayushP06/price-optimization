import html
import logging
import os
import uuid
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from app.database.db import db
from app.models.product import Optimization

bp = Blueprint("health", __name__)
logger = logging.getLogger(__name__)


@bp.get("/api/health")
def health():
    try:
        db.session.execute(db.text("SELECT 1"))
        return jsonify({"status": "healthy", "database": "connected", "timestamp": datetime.now(timezone.utc).isoformat()})
    except Exception:
        error_id = str(uuid.uuid4())
        logger.exception("Health check failed; error_id=%s", error_id)
        return jsonify({"status": "unhealthy", "error": "Database unavailable", "error_id": error_id}), 500


@bp.get("/admin/status")
def admin_status():
    token = os.environ.get("ADMIN_TOKEN")
    if not token or request.headers.get("Authorization") != f"Bearer {token}":
        return jsonify({"error": "Unauthorized"}), 401
    try:
        count = Optimization.query.count()
        return ("<html><body><h1>Database Connected</h1><p>Total records: " + html.escape(str(count)) + "</p></body></html>")
    except Exception:
        error_id = str(uuid.uuid4())
        logger.exception("Admin status failed; error_id=%s", error_id)
        return jsonify({"error": "Database unavailable", "error_id": error_id}), 500
