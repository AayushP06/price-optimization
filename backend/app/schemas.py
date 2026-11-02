"""Pydantic request and response models for the HTTP API."""

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

CATEGORIES = {"Electronics", "Clothing", "Home", "Beauty", "Sports"}


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cost_price: float = Field(gt=0)
    fixed_costs: float = Field(default=0, ge=0)
    min_markup_pct: float = Field(default=5, validation_alias=AliasChoices("min_markup_pct", "min_margin"))
    max_markup_pct: float = Field(default=80, validation_alias=AliasChoices("max_markup_pct", "max_margin"))
    month: int = Field(default=6, ge=1, le=12)
    rating: float = Field(default=4.2, ge=1, le=5)
    ad_spend: float = Field(default=1000, ge=0)
    category: str = "Electronics"
    available_units: int | None = Field(default=None, ge=0)
    competitor_prices: list[float] = Field(default_factory=list, max_length=5000)
    num_competitors: int = Field(default=50, ge=1, le=5000)
    seed: int | None = None
    product_id: str | None = None
    product_name: str | None = Field(default=None, max_length=100)
    quality: str = "standard"  # metadata only; not used by the model. Future option: filter competitor prices by tier.

    @field_validator("category")
    @classmethod
    def valid_category(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError("must be one of Electronics, Clothing, Home, Beauty, Sports")
        return value

    @field_validator("competitor_prices")
    @classmethod
    def positive_prices(cls, value: list[float]) -> list[float]:
        if any(price <= 0 for price in value):
            raise ValueError("each price must be > 0")
        return value

    @field_validator("quality")
    @classmethod
    def valid_quality(cls, value: str) -> str:
        if value not in {"budget", "standard", "premium"}:
            raise ValueError("must be one of budget, standard, premium")
        return value

    @model_validator(mode="after")
    def ordered_margins(self):
        if self.min_markup_pct >= self.max_markup_pct:
            raise ValueError("min_markup_pct must be less than max_markup_pct")
        return self


class APIResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


def validation_errors(error: ValidationError) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for item in error.errors():
        loc = ".".join(str(part) for part in item["loc"] if part != "") or "request"
        result.setdefault(loc, []).append(item["msg"].replace("Value error, ", ""))
    return result
