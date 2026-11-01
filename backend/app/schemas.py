"""Request validation for public API payloads."""
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

CATEGORIES = {"Electronics", "Clothing", "Home", "Beauty", "Sports"}


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cost_price: float = Field(gt=0)
    fixed_costs: float = Field(default=0, ge=0)
    min_margin: float = 5
    max_margin: float = 80
    month: int = Field(default=6, ge=1, le=12)
    rating: float = Field(default=4.2, ge=1, le=5)
    ad_spend: float = Field(default=1000, ge=0)
    category: str = "Electronics"
    available_units: int | None = Field(default=None, ge=0)
    competitor_prices: list[float] = Field(default_factory=list, max_length=5000)
    num_competitors: int = Field(default=50, ge=1, le=5000)
    product_id: str | None = None
    product_name: str | None = None
    quality: float | None = None

    @field_validator("category")
    @classmethod
    def valid_category(cls, value):
        if value not in CATEGORIES:
            raise ValueError("must be one of Electronics, Clothing, Home, Beauty, Sports")
        return value

    @field_validator("competitor_prices")
    @classmethod
    def positive_prices(cls, value):
        if any(price <= 0 for price in value):
            raise ValueError("each price must be > 0")
        return value

    @model_validator(mode="after")
    def ordered_margins(self):
        if self.min_margin >= self.max_margin:
            raise ValueError("min_margin must be less than max_margin")
        return self


def validation_errors(error: ValidationError) -> dict:
    result = {}
    for item in error.errors():
        loc = ".".join(str(part) for part in item["loc"] if part != "") or "request"
        result.setdefault(loc, []).append(item["msg"].replace("Value error, ", ""))
    return result
