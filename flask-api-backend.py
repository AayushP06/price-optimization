# Flask API Backend for Price Optimization System
# Author: Aayush Prasad
# Institution: VSSUT Burla

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import random
import time
from typing import List, Dict

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# ============================================
# CORE ALGORITHMS (Same as before)
# ============================================

def quick_sort(prices: List[float], low: int, high: int) -> List[float]:
    """Quick Sort implementation"""
    if low < high:
        pivot_idx = partition(prices, low, high)
        quick_sort(prices, low, pivot_idx - 1)
        quick_sort(prices, pivot_idx + 1, high)
    return prices


def partition(prices: List[float], low: int, high: int) -> int:
    """Partition function for Quick Sort"""
    pivot = prices[high]
    i = low - 1
    
    for j in range(low, high):
        if prices[j] < pivot:
            i += 1
            prices[i], prices[j] = prices[j], prices[i]
    
    prices[i + 1], prices[high] = prices[high], prices[i + 1]
    return i + 1


def binary_search(sorted_prices: List[float], target_price: float) -> int:
    """Binary Search implementation"""
    left, right = 0, len(sorted_prices) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if sorted_prices[mid] == target_price:
            return mid
        elif sorted_prices[mid] < target_price:
            left = mid + 1
        else:
            right = mid - 1
    
    return left


class PriceOptimizerAPI:
    """Price Optimizer for API usage"""
    
    def __init__(self, cost_price: float, fixed_costs: float = 0):
        self.cost_price = cost_price
        self.fixed_costs = fixed_costs
        self.competitor_prices = []
        self.sorted_prices = []
    
    def add_prices(self, prices: List[float]):
        """Add competitor prices"""
        self.competitor_prices = prices
    
    def sort_prices(self) -> float:
        """Sort prices and return time taken"""
        self.sorted_prices = self.competitor_prices.copy()
        start_time = time.time()
        quick_sort(self.sorted_prices, 0, len(self.sorted_prices) - 1)
        return time.time() - start_time
    
    def calculate_profit_margin(self, selling_price: float) -> float:
        """Calculate profit margin"""
        if selling_price <= self.cost_price:
            return 0
        profit = selling_price - self.cost_price - self.fixed_costs
        return (profit / selling_price) * 100
    
    def estimate_demand(self, selling_price: float) -> float:
        """Estimate demand based on price position"""
        if not self.sorted_prices:
            return 50
        
        position = binary_search(self.sorted_prices, selling_price)
        percentile = (position / len(self.sorted_prices)) * 100
        demand_factor = 100 - percentile
        return 100 * (demand_factor / 100)
    
    def calculate_expected_profit(self, selling_price: float) -> float:
        """Calculate expected profit"""
        profit_per_unit = selling_price - self.cost_price - self.fixed_costs
        demand = self.estimate_demand(selling_price)
        return max(0, profit_per_unit * demand)
    
    def analyze(self, min_margin: float = 15, max_margin: float = 35) -> Dict:
        """Perform complete analysis"""
        # Sort prices
        sort_time = self.sort_prices()
        
        # Define price range
        min_price = self.cost_price * (1 + min_margin/100)
        max_price = self.cost_price * (1 + max_margin/100)
        
        # Binary search for optimal range
        start_time = time.time()
        start_idx = binary_search(self.sorted_prices, min_price)
        end_idx = binary_search(self.sorted_prices, max_price)
        search_time = time.time() - start_time
        
        # Generate test prices
        step = (max_price - min_price) / 12
        test_prices = [min_price + step * i for i in range(13)]
        
        # Analyze each price
        price_analysis = []
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
        
        # Find optimal
        price_analysis.sort(key=lambda x: x['expected_profit'], reverse=True)
        optimal = price_analysis[0]
        
        # Competitor stats
        n = len(self.sorted_prices)
        stats = {
            'count': n,
            'min': round(self.sorted_prices[0], 2),
            'max': round(self.sorted_prices[-1], 2),
            'median': round(self.sorted_prices[n//2], 2),
            'mean': round(sum(self.sorted_prices) / n, 2)
        }
        
        return {
            'optimal_price': optimal,
            'all_prices': price_analysis,
            'competitor_stats': stats,
            'performance': {
                'sort_time': round(sort_time, 4),
                'search_time': round(search_time, 4)
            }
        }


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('price-optimization-website.html')


@app.route('/api/optimize', methods=['POST'])
def optimize_price():
    """
    Main optimization endpoint
    
    Expected JSON:
    {
        "cost_price": 1000,
        "fixed_costs": 50,
        "competitor_prices": [1200, 1300, 1400, ...],
        "min_margin": 15,
        "max_margin": 35
    }
    """
    try:
        data = request.json
        
        # Validate input
        if not data or 'cost_price' not in data:
            return jsonify({'error': 'Missing required field: cost_price'}), 400
        
        product_name = data.get('product_name', '')
        quality = str(data.get('quality', 'standard')).lower()
        cost_price = float(data['cost_price'])
        fixed_costs = float(data.get('fixed_costs', 0))
        competitor_prices = data.get('competitor_prices', [])
        min_margin = float(data.get('min_margin', 15))
        max_margin = float(data.get('max_margin', 35))
        num_competitors = int(data.get('num_competitors', 50))
        
        # Generate sample data if none provided
        # Determine variation based on quality
        quality_variation = {
            'budget': 0.35,
            'standard': 0.25,
            'premium': 0.18
        }.get(quality, 0.25)

        # Use provided competitor prices, else generate sample around market base
        if not competitor_prices:
            # assume market base ~ cost with typical markup influenced by quality
            market_base = max(cost_price * (1 + min_margin/100), cost_price * (1 + 0.2))
            competitor_prices = generate_sample_data(
                base_price=market_base,
                num_competitors=num_competitors,
                variation=quality_variation
            )
        
        # Create optimizer and analyze
        optimizer = PriceOptimizerAPI(cost_price, fixed_costs)
        optimizer.add_prices(competitor_prices)
        results = optimizer.analyze(min_margin, max_margin)
        
        return jsonify({
            'success': True,
            'product': {
                'name': product_name,
                'quality': quality
            },
            'data': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/generate-sample', methods=['POST'])
def generate_sample():
    """
    Generate sample competitor data
    
    Expected JSON:
    {
        "base_price": 1500,
        "num_competitors": 50,
        "variation": 0.3
    }
    """
    try:
        data = request.json or {}
        
        base_price = float(data.get('base_price', 1500))
        num_competitors = int(data.get('num_competitors', 50))
        variation = float(data.get('variation', 0.3))
        
        prices = generate_sample_data(base_price, num_competitors, variation)
        
        return jsonify({
            'success': True,
            'prices': prices,
            'count': len(prices)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sort', methods=['POST'])
def sort_prices():
    """
    Sort prices using Quick Sort
    
    Expected JSON:
    {
        "prices": [1200, 1300, 1400, ...]
    }
    """
    try:
        data = request.json
        
        if not data or 'prices' not in data:
            return jsonify({'error': 'Missing required field: prices'}), 400
        
        prices = data['prices'].copy()
        
        start_time = time.time()
        quick_sort(prices, 0, len(prices) - 1)
        sort_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'sorted_prices': prices,
            'sort_time': round(sort_time, 4),
            'algorithm': 'Quick Sort'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/search', methods=['POST'])
def search_price():
    """
    Search for a price using Binary Search
    
    Expected JSON:
    {
        "sorted_prices": [1200, 1300, 1400, ...],
        "target_price": 1550
    }
    """
    try:
        data = request.json
        
        if not data or 'sorted_prices' not in data or 'target_price' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        sorted_prices = data['sorted_prices']
        target_price = float(data['target_price'])
        
        start_time = time.time()
        index = binary_search(sorted_prices, target_price)
        search_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'index': index,
            'search_time': round(search_time, 6),
            'algorithm': 'Binary Search',
            'found': index < len(sorted_prices) and sorted_prices[index] == target_price
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Price Optimization API',
        'version': '1.0.0'
    })




def generate_sample_data(base_price: float = 1500, 
                         num_competitors: int = 50,
                         variation: float = 0.3) -> List[float]:
    """Generate realistic sample competitor prices"""
    prices = []
    for _ in range(num_competitors):
        price = base_price * (1 + random.uniform(-variation, variation))
        prices.append(round(price, 2))
    return prices


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("PRICE OPTIMIZATION API SERVER")
    print("="*60)
    print("\nAvailable Endpoints:")
    print("  POST /api/optimize        - Run complete optimization")
    print("  POST /api/generate-sample - Generate sample data")
    print("  POST /api/sort            - Sort prices using Quick Sort")
    print("  POST /api/search          - Search using Binary Search")
    print("  GET  /api/health          - Health check")
    print("\nStarting server on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
