"""API blueprints and registration."""


def register_blueprints(app):
    from app.api.health import bp as health_bp
    from app.api.history import bp as history_bp
    from app.api.optimize import bp as optimize_bp
    from app.api.utils import bp as utils_bp

    for blueprint in (optimize_bp, utils_bp, history_bp, health_bp):
        app.register_blueprint(blueprint)
