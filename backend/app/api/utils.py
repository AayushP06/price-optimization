import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.schemas import APIResponse
from app.services.price_optimizer import binary_search, generate_sample_data, quick_sort

router = APIRouter(tags=["utilities"])


class GenerateSampleRequest(BaseModel):
    base_price: float = Field(default=1500, gt=0)
    num_competitors: int = Field(default=50, ge=1, le=5000)
    variation: float = Field(default=0.3, ge=0, le=1)
    seed: int | None = None


@router.post("/api/generate-sample", response_model=APIResponse, summary="Generate sample competitor prices")
def generate_sample(payload: GenerateSampleRequest):
    prices = generate_sample_data(
        payload.base_price, payload.num_competitors, payload.variation, seed=payload.seed
    )
    return {"success": True, "prices": prices}


@router.post("/api/sort", response_model=APIResponse, summary="Sort competitor prices")
def sort_prices(data: dict):
    prices = data.get("prices")
    if not isinstance(prices, list):
        raise HTTPException(status_code=400, detail={"fields": {"prices": ["must be a list"]}})
    if len(prices) > 100000:
        raise HTTPException(status_code=400, detail={"fields": {"prices": ["must contain at most 100000 items"]}})
    try:
        values = [float(value) for value in prices]
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail={"fields": {"prices": ["must contain numbers"]}})
    started = time.perf_counter()
    if len(values) > 5000:
        values = sorted(values)
    else:
        quick_sort(values, 0, len(values) - 1)
    elapsed = time.perf_counter() - started
    return {
        "success": True,
        "sorted_prices": values,
        "sort_time": round(elapsed, 4),
        "algorithm": "Quick Sort",
    }


@router.post("/api/search", response_model=APIResponse, summary="Find a price in a sorted list")
def search_price(data: dict):
    if not isinstance(data.get("sorted_prices"), list) or "target_price" not in data:
        raise HTTPException(status_code=400, detail="Missing required fields")
    try:
        values = [float(value) for value in data["sorted_prices"]]
        target = float(data["target_price"])
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Prices must be numeric")
    started = time.perf_counter()
    index = binary_search(values, target)
    elapsed = time.perf_counter() - started
    return {
        "success": True,
        "index": index,
        "search_time": round(elapsed, 6),
        "found": index < len(values) and values[index] == target,
    }
