"""
Core DS algorithms and price optimization logic.

Algorithms used:
  - Quick Sort  → sort competitor prices
  - Binary Search → find optimal price range
"""

import random
import time
from typing import List, Dict


def quick_sort(prices: List[float], low: int, high: int) -> List[float]:
    if low < high:
        pivot_idx = _partition(prices, low, high)
        quick_sort(prices, low, pivot_idx - 1)
        quick_sort(prices, pivot_idx + 1, high)
    return prices


def _partition(prices: List[float], low: int, high: int) -> int:
    pivot = prices[high]
    i = low - 1
    for j in range(low, high):
        if prices[j] < pivot:
            i += 1
            prices[i], prices[j] = prices[j], prices[i]
    prices[i + 1], prices[high] = prices[high], prices[i + 1]
    return i + 1

def binary_search(sorted_prices: List[float], target: float) -> int:
    """Returns the insertion index of `target` in `sorted_prices`."""
    left, right = 0, len(sorted_prices) - 1
    while left <= right:
        mid = (left + right) // 2
        if sorted_prices[mid] == target:
            return mid
        elif sorted_prices[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left


# ─────────────────────── PRICE OPTIMIZER ────────────────────────

class PriceOptimizerAPI:

    def __init__(self, cost_price: float, fixed_costs: float = 0,
                 available_units: int = None):
        self.cost_price = cost_price
        self.fixed_costs = fixed_costs
        self.available_units = available_units
        self.sorted_prices: List[float] = []

    def add_prices(self, prices: List[float]):
        """Validate, copy, and sort the supplied competitor prices."""
        if not isinstance(prices, list):
            prices = []
        try:
            price_list = [float(p) for p in prices if p is not None]
        except (ValueError, TypeError):
            price_list = []

        if price_list:
            self.sorted_prices = price_list.copy()
            quick_sort(self.sorted_prices, 0, len(self.sorted_prices) - 1)
        else:
            self.sorted_prices = []

    # ── Metrics ─────────────────────────────────────────────────

    def calculate_profit_margin(self, selling_price: float) -> float:
        if selling_price <= self.cost_price:
            return 0.0
        profit = selling_price - self.cost_price - self.fixed_costs
        return (profit / selling_price) * 100

    def estimate_demand(self, selling_price: float) -> float:
        """Estimate demand (0–100 units) based on price percentile."""
        if not self.sorted_prices:
            return 50.0
        position = binary_search(self.sorted_prices, selling_price)
        percentile = (position / len(self.sorted_prices)) * 100
        demand_units = (100 - percentile) / 100 * 100
        return max(10.0, demand_units)

    def calculate_expected_profit(self, selling_price: float) -> float:
        profit_per_unit = selling_price - self.cost_price - self.fixed_costs
        if profit_per_unit <= 0:
            return 0.0
        demand_units = self.estimate_demand(selling_price)
        if self.available_units is not None:
            demand_units = min(demand_units, self.available_units)
        return max(0.0, profit_per_unit * demand_units)

    # ── Main Analysis ────────────────────────────────────────────

    def analyze(self, min_margin: float = 15, max_margin: float = 35) -> Dict:
        """Run full optimization and return results dict."""
        min_price = self.cost_price * (1 + min_margin / 100)
        max_price = self.cost_price * (1 + max_margin / 100)

        start = time.time()
        binary_search(self.sorted_prices, min_price)
        binary_search(self.sorted_prices, max_price)
        search_time = time.time() - start

        # Test 13 evenly-spaced price points across the margin range
        step = (max_price - min_price) / 12
        test_prices = [min_price + step * i for i in range(13)]

        price_analysis = [
            {
                "price": round(p, 2),
                "margin": round(self.calculate_profit_margin(p), 2),
                "demand": round(self.estimate_demand(p), 2),
                "expected_profit": round(self.calculate_expected_profit(p), 2),
            }
            for p in test_prices
        ]

        price_analysis.sort(key=lambda x: x["expected_profit"], reverse=True)
        optimal = price_analysis[0]

        n = len(self.sorted_prices)
        stats = (
            {"count": 0, "min": 0, "max": 0, "median": 0, "mean": 0}
            if n == 0
            else {
                "count": n,
                "min": round(self.sorted_prices[0], 2),
                "max": round(self.sorted_prices[-1], 2),
                "median": round(self.sorted_prices[n // 2], 2),
                "mean": round(sum(self.sorted_prices) / n, 2),
            }
        )

        return {
            "optimal_price": optimal,
            "all_prices": price_analysis,
            "competitor_stats": stats,
            "competitor_prices": self.sorted_prices,
            "performance": {
                "sort_time": 0.0,          # sorting done in add_prices
                "search_time": round(search_time, 4),
            },
        }


# ─────────────────────── SAMPLE GENERATOR ───────────────────────

def generate_sample_data(base_price: float = 1500,
                         num_competitors: int = 50,
                         variation: float = 0.3,
                         use_fixed_seed: bool = False) -> List[float]:
    """Generate random competitor prices around a base price."""
    if use_fixed_seed:
        random.seed(42)
    prices = [
        round(base_price * (1 + random.uniform(-variation, variation)), 2)
        for _ in range(num_competitors)
    ]
    if use_fixed_seed:
        random.seed()
    return prices
