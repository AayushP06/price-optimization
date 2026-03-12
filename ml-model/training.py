"""
Standalone ML / Data Science demo script.

This runs the price optimization algorithms directly in the terminal
with matplotlib visualizations — no Flask required.

Usage:
    python ml-model/training.py
"""

import random
import time
import json
import matplotlib.pyplot as plt
from typing import List, Dict


# ─────────────────────── ALGORITHMS ─────────────────────────────

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


# ─────────────────────── OPTIMIZER CLASS ────────────────────────

class PriceOptimizer:

    def __init__(self, cost_price: float, fixed_costs: float = 0):
        self.cost_price = cost_price
        self.fixed_costs = fixed_costs
        self.competitor_prices: List[float] = []
        self.sorted_prices: List[float] = []
        self.analysis_results: Dict = {}

    def add_competitor_prices(self, prices: List[float]):
        self.competitor_prices.extend(prices)
        print(f"✓ Added {len(prices)} competitor prices")

    def sort_prices(self) -> float:
        self.sorted_prices = self.competitor_prices.copy()
        start = time.time()
        quick_sort(self.sorted_prices, 0, len(self.sorted_prices) - 1)
        sort_time = time.time() - start
        print(f"✓ Sorted {len(self.sorted_prices)} prices in {sort_time:.4f}s")
        return sort_time

    def calculate_profit_margin(self, selling_price: float) -> float:
        if selling_price <= self.cost_price:
            return 0
        profit = selling_price - self.cost_price - self.fixed_costs
        return max(0, (profit / selling_price) * 100)

    def estimate_demand(self, selling_price: float) -> float:
        if not self.sorted_prices:
            return 50
        position = binary_search(self.sorted_prices, selling_price)
        percentile = (position / len(self.sorted_prices)) * 100
        return 100 * (100 - percentile) / 100

    def calculate_expected_profit(self, selling_price: float) -> float:
        profit_per_unit = selling_price - self.cost_price - self.fixed_costs
        demand = self.estimate_demand(selling_price)
        return max(0, profit_per_unit * demand)

    def analyze_price_points(self, min_margin: float = 15, max_margin: float = 35):
        if not self.sorted_prices:
            print("⚠  Sort prices first!")
            return

        min_price = self.cost_price * (1 + min_margin / 100)
        max_price = self.cost_price * (1 + max_margin / 100)

        start = time.time()
        start_idx = binary_search(self.sorted_prices, min_price)
        end_idx = binary_search(self.sorted_prices, max_price)
        search_time = time.time() - start
        print(f"✓ Binary Search completed in {search_time:.4f}s")

        price_range = self.sorted_prices[start_idx:end_idx] or [
            min_price + (max_price - min_price) * i / 10 for i in range(11)
        ]

        price_analysis = [
            {
                "price": round(p, 2),
                "margin": round(self.calculate_profit_margin(p), 2),
                "demand": round(self.estimate_demand(p), 2),
                "expected_profit": round(self.calculate_expected_profit(p), 2),
            }
            for p in price_range[:10]
        ]
        price_analysis.sort(key=lambda x: x["expected_profit"], reverse=True)

        self.analysis_results = {
            "optimal_price": price_analysis[0],
            "all_prices": price_analysis,
            "search_time": search_time,
            "competitor_stats": self.get_competitor_stats(),
        }
        return price_analysis[0]

    def get_competitor_stats(self) -> Dict:
        n = len(self.sorted_prices)
        if n == 0:
            return {}
        return {
            "count": n,
            "min": round(self.sorted_prices[0], 2),
            "max": round(self.sorted_prices[-1], 2),
            "median": round(self.sorted_prices[n // 2], 2),
            "mean": round(sum(self.sorted_prices) / n, 2),
        }

    def display_results(self):
        if not self.analysis_results:
            print("⚠  No analysis performed yet!")
            return
        optimal = self.analysis_results["optimal_price"]
        stats = self.analysis_results["competitor_stats"]
        print("\n" + "=" * 60)
        print("OPTIMAL PRICE RECOMMENDATION")
        print("=" * 60)
        print(f"\n💰 Recommended Price: ₹{optimal['price']}")
        print(f"📊 Profit Margin:     {optimal['margin']}%")
        print(f"📈 Estimated Demand:  {optimal['demand']} units")
        print(f"💵 Expected Profit:   ₹{optimal['expected_profit']}")
        print("\n" + "-" * 60)
        print("COMPETITOR ANALYSIS")
        print("-" * 60)
        print(f"Total Competitors: {stats.get('count', 0)}")
        print(f"Price Range:       ₹{stats.get('min')} – ₹{stats.get('max')}")
        print(f"Median Price:      ₹{stats.get('median')}")
        print(f"Average Price:     ₹{stats.get('mean')}")
        print("=" * 60 + "\n")

    def visualize_results(self):
        if not self.analysis_results:
            print("⚠  No analysis to visualize!")
            return
        prices = [p["price"] for p in self.analysis_results["all_prices"]]
        margins = [p["margin"] for p in self.analysis_results["all_prices"]]
        profits = [p["expected_profit"] for p in self.analysis_results["all_prices"]]
        optimal_price = self.analysis_results["optimal_price"]["price"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Dynamic Price Optimization Analysis", fontsize=16, fontweight="bold")

        ax1.plot(prices, margins, "o-", color="#f39c12", linewidth=2, markersize=8)
        ax1.axvline(optimal_price, color="#e67e22", linestyle="--", linewidth=2,
                    label=f"Optimal: ₹{optimal_price}")
        ax1.set_xlabel("Price (₹)"); ax1.set_ylabel("Profit Margin (%)")
        ax1.set_title("Price vs Profit Margin"); ax1.grid(True, alpha=0.3); ax1.legend()

        ax2.plot(prices, profits, "s-", color="#27ae60", linewidth=2, markersize=8)
        ax2.axvline(optimal_price, color="#e67e22", linestyle="--", linewidth=2,
                    label=f"Optimal: ₹{optimal_price}")
        ax2.set_xlabel("Price (₹)"); ax2.set_ylabel("Expected Profit (₹)")
        ax2.set_title("Price vs Expected Profit"); ax2.grid(True, alpha=0.3); ax2.legend()

        plt.tight_layout()
        plt.savefig("ml-model/price_optimization_analysis.png", dpi=300, bbox_inches="tight")
        print("✓ Visualization saved as 'ml-model/price_optimization_analysis.png'")
        plt.show()

    def export_results(self, filename: str = "ml-model/optimization_results.json"):
        if not self.analysis_results:
            return
        with open(filename, "w") as f:
            json.dump(
                {"cost_price": self.cost_price,
                 "fixed_costs": self.fixed_costs,
                 "analysis": self.analysis_results},
                f, indent=4,
            )
        print(f"✓ Results exported to '{filename}'")


# ─────────────────────── SAMPLE GENERATOR ───────────────────────

def generate_sample_competitor_data(num_competitors: int = 50,
                                    base_price: float = 1500,
                                    variation: float = 0.3) -> List[float]:
    return [
        round(base_price * (1 + random.uniform(-variation, variation)), 2)
        for _ in range(num_competitors)
    ]


# ─────────────────────────── MAIN ───────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("DYNAMIC PRICE OPTIMIZATION SYSTEM")
    print("Using Quick Sort & Binary Search Algorithms")
    print("=" * 60 + "\n")

    cost_price = 1000
    fixed_costs = 50

    optimizer = PriceOptimizer(cost_price, fixed_costs)

    print("📊 Generating competitor pricing data...")
    competitor_prices = generate_sample_competitor_data(
        num_competitors=100, base_price=1500, variation=0.25
    )
    optimizer.add_competitor_prices(competitor_prices)

    print("\n🔄 Sorting prices using Quick Sort...")
    optimizer.sort_prices()

    print("\n🔍 Analyzing price points...")
    optimizer.analyze_price_points(min_margin=15, max_margin=35)

    optimizer.display_results()

    print("📈 Generating visualizations...")
    optimizer.visualize_results()

    optimizer.export_results()

    print("✅ Analysis Complete!\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
