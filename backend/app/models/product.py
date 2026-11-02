from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.db import Base


class Optimization(Base):
    """Stores each price optimization result."""

    __tablename__ = "optimizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[str | None] = mapped_column(String(100), index=True)
    cost_price: Mapped[float] = mapped_column(Float, nullable=False)
    fixed_costs: Mapped[float | None] = mapped_column(Float, default=0)
    available_units: Mapped[int | None] = mapped_column(Integer, nullable=True)
    category: Mapped[str | None] = mapped_column(String(40))
    month: Mapped[int | None] = mapped_column(Integer)
    rating: Mapped[float | None] = mapped_column(Float)
    ad_spend: Mapped[float | None] = mapped_column(Float)
    method: Mapped[str | None] = mapped_column(String(40))
    product_name: Mapped[str | None] = mapped_column(String(200))
    quality: Mapped[str | None] = mapped_column(String(20), default="standard")  # metadata only; not used by the model. Future option: filter competitor prices by tier.

    recommended_price: Mapped[float | None] = mapped_column(Float)
    expected_profit: Mapped[float | None] = mapped_column(Float)
    expected_units_sold: Mapped[int | None] = mapped_column(Integer)

    competitor_prices: Mapped[list[float] | None] = mapped_column(JSON)
    competitor_stats: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    price_analysis: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)

    timestamp: Mapped[datetime | None] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    duration_ms: Mapped[float | None] = mapped_column(Float)

    def to_dict(self) -> dict[str, Any]:
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
            "simulated_competitors": (self.competitor_stats or {}).get("simulated"),
            "min_markup_pct": round((self.price_analysis[0]["price"] / self.cost_price - 1) * 100, 2) if self.price_analysis and self.cost_price else None,
            "max_markup_pct": round((self.price_analysis[-1]["price"] / self.cost_price - 1) * 100, 2) if self.price_analysis and self.cost_price else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
