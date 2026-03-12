from datetime import datetime
from app.database.db import db


class Optimization(db.Model):
    """Stores each price optimization result."""
    __tablename__ = "optimizations"

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(100), index=True)
    cost_price = db.Column(db.Float, nullable=False)
    fixed_costs = db.Column(db.Float, default=0)
    available_units = db.Column(db.Integer, nullable=True)

    # Results
    recommended_price = db.Column(db.Float)
    expected_profit = db.Column(db.Float)
    expected_units_sold = db.Column(db.Integer)

    # Complex data stored as JSON
    competitor_prices = db.Column(db.JSON)
    competitor_stats = db.Column(db.JSON)
    price_analysis = db.Column(db.JSON)

    # Metadata
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    duration_ms = db.Column(db.Float)

    def to_dict(self):
        return {
            "id": self.id,
            "product_id": self.product_id,
            "cost_price": self.cost_price,
            "recommended_price": self.recommended_price,
            "expected_profit": self.expected_profit,
            "competitor_prices": self.competitor_prices,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
