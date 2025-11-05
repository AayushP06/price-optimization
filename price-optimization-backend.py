# Dynamic Price Optimization System
# Author: Aayush Prasad
# Institution: VSSUT Burla
# Course: Data Structures & Algorithms

import random
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json

# ============================================
# SORTING ALGORITHMS
# ============================================

def quick_sort(prices: List[float], low: int, high: int) -> List[float]:
    """
    Quick Sort implementation for sorting competitor prices
    Time Complexity: O(n log n) average, O(n²) worst case
    Space Complexity: O(log n) due to recursion stack
    """
    if low < high:
        pivot_idx = partition(prices, low, high)
        quick_sort(prices, low, pivot_idx - 1)
        quick_sort(prices, pivot_idx + 1, high)
    return prices


def partition(prices: List[float], low: int, high: int) -> int:
    """
    Partition function for Quick Sort
    Selects last element as pivot and arranges elements
    """
    pivot = prices[high]
    i = low - 1
    
    for j in range(low, high):
        if prices[j] < pivot:
            i += 1
            prices[i], prices[j] = prices[j], prices[i]
    
    prices[i + 1], prices[high] = prices[high], prices[i + 1]
    return i + 1


# ============================================
# SEARCHING ALGORITHMS
# ============================================

def binary_search(sorted_prices: List[float], target_price: float) -> int:
    """
    Binary Search to find target price or nearest position
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(sorted_prices) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if sorted_prices[mid] == target_price:
            return mid
        elif sorted_prices[mid] < target_price:
            left = mid + 1
        else:
            right = mid - 1
    
    return left  # Return insertion position if not found


def find_optimal_price_range(sorted_prices: List[float], 
                             min_price: float, 
                             max_price: float) -> List[float]:
    """
    Find all prices within the optimal range using binary search
    """
    start_idx = binary_search(sorted_prices, min_price)
    end_idx = binary_search(sorted_prices, max_price)
    
    # Adjust indices to include boundary values
    if start_idx > 0 and sorted_prices[start_idx] > min_price:
        start_idx -= 1
    if end_idx < len(sorted_prices) and sorted_prices[end_idx] < max_price:
        end_idx += 1
    
    return sorted_prices[start_idx:end_idx]


# ============================================
# PROFIT CALCULATION ENGINE
# ============================================

class PriceOptimizer:
    """
    Main Price Optimization Engine
    Analyzes competitor prices and recommends optimal pricing
    """
    
    def __init__(self, cost_price: float, fixed_costs: float = 0):
        self.cost_price = cost_price
        self.fixed_costs = fixed_costs
        self.competitor_prices = []
        self.sorted_prices = []
        self.analysis_results = {}
    
    
    def add_competitor_prices(self, prices: List[float]):
        """Add competitor pricing data"""
        self.competitor_prices.extend(prices)
        print(f"✓ Added {len(prices)} competitor prices")
    
    
    def sort_prices(self) -> float:
        """Sort prices using Quick Sort and measure time"""
        self.sorted_prices = self.competitor_prices.copy()
        
        start_time = time.time()
        quick_sort(self.sorted_prices, 0, len(self.sorted_prices) - 1)
        sort_time = time.time() - start_time
        
        print(f"✓ Sorted {len(self.sorted_prices)} prices in {sort_time:.4f} seconds")
        return sort_time
    
    
    def calculate_profit_margin(self, selling_price: float) -> float:
        """Calculate profit margin percentage"""
        if selling_price <= self.cost_price:
            return 0
        profit = selling_price - self.cost_price - self.fixed_costs
        margin = (profit / selling_price) * 100
        return max(0, margin)
    
    
    def estimate_demand(self, selling_price: float) -> float:
        """
        Estimate demand based on price competitiveness
        Lower prices = higher demand (simplified model)
        """
        if not self.sorted_prices:
            return 50  # Default demand
        
        # Calculate percentile rank
        position = binary_search(self.sorted_prices, selling_price)
        percentile = (position / len(self.sorted_prices)) * 100
        
        # Demand decreases as price increases
        demand_factor = 100 - percentile
        base_demand = 100
        
        return base_demand * (demand_factor / 100)
    
    
    def calculate_expected_profit(self, selling_price: float) -> float:
        """Calculate expected profit considering demand"""
        margin = self.calculate_profit_margin(selling_price)
        demand = self.estimate_demand(selling_price)
        
        # Expected profit = profit per unit × estimated sales
        profit_per_unit = selling_price - self.cost_price - self.fixed_costs
        expected_profit = profit_per_unit * demand
        
        return max(0, expected_profit)
    
    
    def analyze_price_points(self, min_margin: float = 15, max_margin: float = 35):
        """
        Analyze multiple price points and find optimal price
        """
        if not self.sorted_prices:
            print("⚠ Please sort prices first!")
            return
        
        print("\n" + "="*60)
        print("PRICE ANALYSIS IN PROGRESS...")
        print("="*60)
        
        # Define price range based on desired margins
        min_price = self.cost_price * (1 + min_margin/100)
        max_price = self.cost_price * (1 + max_margin/100)
        
        # Search for prices in optimal range
        start_time = time.time()
        price_range = find_optimal_price_range(self.sorted_prices, min_price, max_price)
        search_time = time.time() - start_time
        
        print(f"✓ Binary Search completed in {search_time:.4f} seconds")
        print(f"✓ Found {len(price_range)} prices in optimal range")
        
        # Analyze each price point
        price_analysis = []
        
        # Generate test price points
        test_prices = []
        if price_range:
            test_prices = price_range[:min(10, len(price_range))]
        else:
            # Fallback: generate prices within range
            test_prices = [min_price + (max_price - min_price) * i / 10 
                          for i in range(11)]
        
        for price in test_prices:
            margin = self.calculate_profit_margin(price)
            demand = self.estimate_demand(price)
            expected_profit = self.calculate_expected_profit(price)
            
            price_analysis.append({
                'price': round(price, 2),
                'margin': round(margin, 2),
                'demand': round(demand, 2),
                'expected_profit': round(expected_profit, 2)
            })
        
        # Sort by expected profit
        price_analysis.sort(key=lambda x: x['expected_profit'], reverse=True)
        
        self.analysis_results = {
            'optimal_price': price_analysis[0],
            'all_prices': price_analysis,
            'search_time': search_time,
            'competitor_stats': self.get_competitor_stats()
        }
        
        return price_analysis[0]
    
    
    def get_competitor_stats(self) -> Dict:
        """Get statistical summary of competitor prices"""
        if not self.sorted_prices:
            return {}
        
        n = len(self.sorted_prices)
        return {
            'count': n,
            'min': round(self.sorted_prices[0], 2),
            'max': round(self.sorted_prices[-1], 2),
            'median': round(self.sorted_prices[n//2], 2),
            'mean': round(sum(self.sorted_prices) / n, 2)
        }
    
    
    def display_results(self):
        """Display comprehensive analysis results"""
        if not self.analysis_results:
            print("⚠ No analysis performed yet!")
            return
        
        optimal = self.analysis_results['optimal_price']
        stats = self.analysis_results['competitor_stats']
        
        print("\n" + "="*60)
        print("OPTIMAL PRICE RECOMMENDATION")
        print("="*60)
        print(f"\n💰 Recommended Price: ₹{optimal['price']}")
        print(f"📊 Profit Margin: {optimal['margin']}%")
        print(f"📈 Estimated Demand: {optimal['demand']} units")
        print(f"💵 Expected Profit: ₹{optimal['expected_profit']}")
        
        print("\n" + "-"*60)
        print("COMPETITOR ANALYSIS")
        print("-"*60)
        print(f"Total Competitors Analyzed: {stats['count']}")
        print(f"Price Range: ₹{stats['min']} - ₹{stats['max']}")
        print(f"Median Price: ₹{stats['median']}")
        print(f"Average Price: ₹{stats['mean']}")
        
        print("\n" + "-"*60)
        print("TOP 5 PRICE OPTIONS")
        print("-"*60)
        print(f"{'Price':<12} {'Margin %':<12} {'Demand':<12} {'Exp. Profit':<15}")
        print("-"*60)
        
        for i, price_data in enumerate(self.analysis_results['all_prices'][:5], 1):
            print(f"₹{price_data['price']:<10} {price_data['margin']:<12} "
                  f"{price_data['demand']:<12} ₹{price_data['expected_profit']:<13}")
        
        print("="*60 + "\n")
    
    
    def visualize_results(self):
        """Generate visualization of price vs profit analysis"""
        if not self.analysis_results:
            print("⚠ No analysis to visualize!")
            return
        
        prices = [p['price'] for p in self.analysis_results['all_prices']]
        margins = [p['margin'] for p in self.analysis_results['all_prices']]
        expected_profits = [p['expected_profit'] for p in self.analysis_results['all_prices']]
        
        optimal_price = self.analysis_results['optimal_price']['price']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Dynamic Price Optimization Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Price vs Profit Margin
        ax1.plot(prices, margins, 'o-', color='#f39c12', 
                linewidth=2, markersize=8, label='Profit Margin')
        ax1.axvline(optimal_price, color='#e67e22', 
                   linestyle='--', linewidth=2, label=f'Optimal: ₹{optimal_price}')
        ax1.set_xlabel('Price (₹)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Profit Margin (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Price vs Profit Margin', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Price vs Expected Profit
        ax2.plot(prices, expected_profits, 's-', color='#27ae60', 
                linewidth=2, markersize=8, label='Expected Profit')
        ax2.axvline(optimal_price, color='#e67e22', 
                   linestyle='--', linewidth=2, label=f'Optimal: ₹{optimal_price}')
        ax2.set_xlabel('Price (₹)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Expected Profit (₹)', fontsize=12, fontweight='bold')
        ax2.set_title('Price vs Expected Profit', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('price_optimization_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved as 'price_optimization_analysis.png'")
        plt.show()
    
    
    def export_results(self, filename: str = 'optimization_results.json'):
        """Export analysis results to JSON file"""
        if not self.analysis_results:
            print("⚠ No results to export!")
            return
        
        export_data = {
            'product_info': {
                'cost_price': self.cost_price,
                'fixed_costs': self.fixed_costs
            },
            'analysis': self.analysis_results
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=4)
        
        print(f"✓ Results exported to '{filename}'")


# ============================================
# SAMPLE DATA GENERATOR
# ============================================

def generate_sample_competitor_data(num_competitors: int = 50, 
                                   base_price: float = 1500,
                                   variation: float = 0.3) -> List[float]:
    """
    Generate realistic competitor pricing data
    """
    prices = []
    for _ in range(num_competitors):
        # Random variation around base price
        price = base_price * (1 + random.uniform(-variation, variation))
        prices.append(round(price, 2))
    
    return prices


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("DYNAMIC PRICE OPTIMIZATION SYSTEM")
    print("Using Quick Sort & Binary Search Algorithms")
    print("="*60 + "\n")
    
    # Initialize the optimizer
    cost_price = 1000  # Product cost
    fixed_costs = 50   # Fixed costs per unit
    
    optimizer = PriceOptimizer(cost_price, fixed_costs)
    
    # Generate sample competitor data
    print("📊 Generating competitor pricing data...")
    competitor_prices = generate_sample_competitor_data(
        num_competitors=100,
        base_price=1500,
        variation=0.25
    )
    
    optimizer.add_competitor_prices(competitor_prices)
    
    # Sort the prices
    print("\n🔄 Sorting prices using Quick Sort...")
    sort_time = optimizer.sort_prices()
    
    # Analyze and find optimal price
    print("\n🔍 Analyzing price points...")
    optimal = optimizer.analyze_price_points(min_margin=15, max_margin=35)
    
    # Display results
    optimizer.display_results()
    
    # Generate visualization
    print("\n📈 Generating visualizations...")
    optimizer.visualize_results()
    
    # Export results
    optimizer.export_results()
    
    print("\n✅ Analysis Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
