"""Flask app factory."""

import os

from flask import Flask, abort, send_from_directory
from flask_cors import CORS

from app.api.core.config import Config
from app.database.db import db


def create_app(config_class=Config):
    app = Flask(__name__, static_folder=None)
    app.config.from_object(config_class)

    frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))

    @app.get("/")
    def frontend_index():
        if not os.path.isfile(os.path.join(frontend_dist, "index.html")):
            abort(404)
        return send_from_directory(frontend_dist, "index.html")

    @app.get("/<path:filename>")
    def frontend_assets(filename):
        if not os.path.isfile(os.path.join(frontend_dist, "index.html")):
            abort(404)
        target = os.path.join(frontend_dist, filename)
        if os.path.isfile(target):
            return send_from_directory(frontend_dist, filename)
        if os.path.splitext(filename)[1]:
            abort(404)
        return send_from_directory(frontend_dist, "index.html")

    # Extensions
    CORS(app)
    db.init_app(app)

    # Create tables
    with app.app_context():
        # Import models so SQLAlchemy registers them before create_all
        from app.models.product import Optimization  # noqa: F401
        db.create_all()

    # Register blueprints
    from app.api import register_blueprints
    register_blueprints(app)

    # Verify/load model files and the saved feature contract during app startup.
    from app.services.xgb_optimizer import get_ml_optimizer
    get_ml_optimizer()

    return app
