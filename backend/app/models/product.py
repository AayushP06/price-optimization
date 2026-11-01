from datetime import datetime, timezone
from app.database.db import db


class Optimization(db.Model):
    """Stores each price optimization result."""
    __tablename__ = "optimizations"

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(100), index=True)
    cost_price = db.Column(db.Float, nullable=False)
    fixed_costs = db.Column(db.Float, default=0)
    available_units = db.Column(db.Integer, nullable=True)
    category = db.Column(db.String(40))
    month = db.Column(db.Integer)
    rating = db.Column(db.Float)
    ad_spend = db.Column(db.Float)
    method = db.Column(db.String(40))
    product_name = db.Column(db.String(200))
    quality = db.Column(db.String(20), default="standard")  # metadata only; not used by the model. Future option: filter competitor prices by tier.

    # Results
    recommended_price = db.Column(db.Float)
    expected_profit = db.Column(db.Float)
    expected_units_sold = db.Column(db.Integer)

    # Complex data stored as JSON
    competitor_prices = db.Column(db.JSON)
    competitor_stats = db.Column(db.JSON)
    price_analysis = db.Column(db.JSON)

    # Metadata
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    duration_ms = db.Column(db.Float)

    def to_dict(self):
        return {
            "id": self.id,
            "product_id": self.product_id,
            "cost_price": self.cost_price,
            "recommended_price": self.recommended_price,
            "expected_profit": self.expected_profit,
            "competitor_prices": self.competitor_prices,
            "expected_units_sold": self.expected_units_sold,
            "fixed_costs": self.fixed_costs,
            "category": self.category,
            "month": self.month,
            "rating": self.rating,
            "ad_spend": self.ad_spend,
            "method": self.method,
            "available_units": self.available_units,
            "product_name": self.product_name,
            "quality": self.quality,
            "competitor_stats": self.competitor_stats,
            "all_prices": self.price_analysis,
            "simulated_competitors": bool((self.competitor_stats or {}).get("simulated", False)),
            "min_markup_pct": round((self.price_analysis[0]["price"] / self.cost_price - 1) * 100, 2) if self.price_analysis and self.cost_price else None,
            "max_markup_pct": round((self.price_analysis[-1]["price"] / self.cost_price - 1) * 100, 2) if self.price_analysis and self.cost_price else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
