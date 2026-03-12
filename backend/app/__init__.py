"""Flask app factory."""

from flask import Flask
from flask_cors import CORS

from app.core.config import Config
from app.database.db import db


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Extensions
    CORS(app)
    db.init_app(app)

    # Create tables
    with app.app_context():
        # Import models so SQLAlchemy registers them before create_all
        from app.models.product import Optimization  # noqa: F401
        db.create_all()

    # Register blueprints
    from app.api.routes import api_bp
    app.register_blueprint(api_bp)

    return app
