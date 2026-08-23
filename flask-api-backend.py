from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import random
import time
from typing import List, Dict
import json
from datetime import datetime
import os
import sys

from dotenv import load_dotenv

load_dotenv()


import numpy as np

app = Flask(__name__)
CORS(app)

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL', 
    'sqlite:///local.db'  
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class Optimization(db.Model):
    __tablename__ = 'optimizations'
    
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(100), index=True)
    cost_price = db.Column(db.Float, nullable=False)
    fixed_costs = db.Column(db.Float, default=0)
    available_units = db.Column(db.Integer, nullable=True)
    
    recommended_price = db.Column(db.Float)
    expected_profit = db.Column(db.Float)
    expected_units_sold = db.Column(db.Integer)
    
 
    competitor_prices = db.Column(db.JSON)
    competitor_stats = db.Column(db.JSON)
    price_analysis = db.Column(db.JSON)
    

    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    duration_ms = db.Column(db.Float)
    
    def to_dict(self):
        return {
            'id': self.id,
            'product_id': self.product_id,
            'cost_price': self.cost_price,
            'recommended_price': self.recommended_price,
            'expected_profit': self.expected_profit,
            'competitor_prices': self.competitor_prices,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

with app.app_context():
    db.create_all()

def quick_sort(prices: List[float], low: int, high: int) -> List[float]:
    if low < high:
        pivot_idx = partition(prices, low, high)
        quick_sort(prices, low, pivot_idx - 1)
        quick_sort(prices, pivot_idx + 1, high)
    return prices

def partition(prices: List[float], low: int, high: int) -> int:
    pivot = prices[high]
    i = low - 1
    
    for j in range(low, high):
        if prices[j] < pivot:
            i += 1
            prices[i], prices[j] = prices[j], prices[i]
    
    prices[i + 1], prices[high] = prices[high], prices[i + 1]
    return i + 1



def binary_search(sorted_prices: List[float], target_price: float) -> int:
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
    
    def __init__(self, cost_price: float, fixed_costs: float = 0, available_units: int = None):
        self.cost_price = cost_price
        self.fixed_costs = fixed_costs
        self.available_units = available_units   # <-- ADDED
        self.sorted_prices = []
    
    def add_prices(self, prices: List[float]):
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
    
    def sort_prices(self) -> float:
        if not self.sorted_prices or len(self.sorted_prices) == 0:
            return 0.0
        return 0.0
    
    def calculate_profit_margin(self, selling_price: float) -> float:
        if selling_price <= self.cost_price:
            return 0
        profit = selling_price - self.cost_price - self.fixed_costs
        return (profit / selling_price) * 100
    
    def estimate_demand(self, selling_price: float) -> float:
        if not self.sorted_prices or len(self.sorted_prices) == 0:
            return 50.0
        
        position = binary_search(self.sorted_prices, selling_price)
        percentile = (position / len(self.sorted_prices)) * 100
        demand_factor = (100 - percentile) / 100
        base_demand = 100.0
        demand_units = base_demand * demand_factor
        return max(10.0, demand_units)
    

    def calculate_expected_profit(self, selling_price: float) -> float:
        profit_per_unit = selling_price - self.cost_price - self.fixed_costs
        if profit_per_unit <= 0:
            return 0.0

        demand_units = self.estimate_demand(selling_price)

  
        if self.available_units is not None:
            actual_units_sold = min(demand_units, self.available_units)
        else:
            actual_units_sold = demand_units
        
        return max(0, profit_per_unit * actual_units_sold)
    
    def analyze(self, min_margin: float = 15, max_margin: float = 35) -> Dict:
        sort_time = self.sort_prices()
        
        min_price = self.cost_price * (1 + min_margin / 100)
        max_price = self.cost_price * (1 + max_margin / 100)
        
        start_time = time.time()
        start_idx = binary_search(self.sorted_prices, min_price)
        end_idx = binary_search(self.sorted_prices, max_price)
        search_time = time.time() - start_time
        
        step = (max_price - min_price) / 12
        test_prices = [min_price + step * i for i in range(13)]
        
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
        
        optimal = max(price_analysis, key=lambda x: x['expected_profit'])
        price_analysis.sort(key=lambda x: x['price'])
        
        n = len(self.sorted_prices)
        if n == 0:
            stats = {'count': 0, 'min': 0, 'max': 0, 'median': 0, 'mean': 0}
        else:
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
            'competitor_prices': self.sorted_prices,
            'performance': {
                'sort_time': round(sort_time, 4),
                'search_time': round(search_time, 4)
            }
        }

def store_request_data(input_data, result_data):
    """Store user input and results in stored_data.json"""

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": input_data,
        "result": result_data
    }

    try:
       
        with open("stored_data.json", "r") as f:
            data = json.load(f)
    except:
    
        data = []

 
    data.append(entry)

    with open("stored_data.json", "w") as f:
        json.dump(data, f, indent=4)



@app.route('/')
def index():
    return send_file('price-optimization-website.html')


@app.route('/api/optimize', methods=['GET', 'POST'])
def optimize_price():

    if request.method == 'GET':
        return jsonify({
            'endpoint': '/api/optimize',
            'method': 'POST',
            'description': 'Get optimal price recommendation',
        }), 200

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Missing JSON payload'}), 400
    try:
        cost_price = float(data['cost_price'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid or missing cost_price'}), 400
    fixed_costs = float(data.get('fixed_costs', 0))
    available_units = data.get('available_units')
    if available_units is not None:
        try:
            available_units = int(available_units)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Invalid available_units'}), 400

    competitor_prices = data.get('competitor_prices')
    if competitor_prices:
        try:
            competitor_prices = [float(p) for p in competitor_prices if p is not None]
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Invalid competitor_prices'}), 400
    else:
        base_price = float(data.get('base_price', 1500))
        num_competitors = int(data.get('num_competitors', 50))
        variation = float(data.get('variation', 0.3))
        competitor_prices = generate_sample_data(base_price, num_competitors, variation)

    category = str(data.get('category', 'Electronics'))
    month = int(data.get('month', 6))
    rating = float(data.get('rating', 4.2))
    ad_spend = float(data.get('ad_spend', 1000))

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))
    from app.services.xgb_optimizer import get_ml_optimizer
    ml_optimizer = get_ml_optimizer()
    start_time = time.time()
    result = ml_optimizer.recommend(
        cost_price=cost_price,
        fixed_costs=fixed_costs,
        competitor_prices=competitor_prices if len(competitor_prices) > 0 else None,
        category=category,
        month=month,
        rating=rating,
        ad_spend=ad_spend,
    )
    duration_ms = (time.time() - start_time) * 1000
    result['success'] = True
    result['performance'] = {'duration_ms': round(duration_ms, 2)}
    result = convert_numpy(result)
    competitor_prices = convert_numpy(competitor_prices)

    try:
        optimization = Optimization(
            product_id=data.get('product_id', f'product_{int(time.time())}'),
            cost_price=cost_price,
            fixed_costs=fixed_costs,
            available_units=available_units,
            recommended_price=result['optimal_price']['price'],
            expected_profit=result['optimal_price']['expected_profit'],
            expected_units_sold=result['optimal_price'].get('expected_units_sold'),
            competitor_prices=competitor_prices,
            competitor_stats=result.get('competitor_stats'),
            price_analysis=result.get('all_prices'),
            duration_ms=duration_ms
        )
        db.session.add(optimization)
        db.session.commit()
        result['id'] = optimization.id
    except Exception as e:
        db.session.rollback()
        print(f"Database error: {e}")

    return jsonify(result)


@app.route('/api/generate-sample', methods=['POST'])
def generate_sample():
    try:
        data = request.json or {}
        
        base_price = float(data.get('base_price', 1500))
        num_competitors = int(data.get('num_competitors', 50))
        variation = float(data.get('variation', 0.3))
        
        prices = generate_sample_data(base_price, num_competitors, variation)
        
        return jsonify({'success': True, 'prices': prices})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sort', methods=['POST'])
def sort_prices():
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


@app.route('/api/search', methods=['POST'])
def search_price():
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
        'found': index < len(sorted_prices) and sorted_prices[index] == target_price
    })


@app.route('/api/health')
def health():
    """Health check endpoint"""
    try:
        db.session.execute(db.text('SELECT 1'))
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get optimization history"""
    try:
        limit = int(request.args.get('limit', 50))
        product_id = request.args.get('product_id')
        
        query = Optimization.query
        if product_id:
            query = query.filter_by(product_id=product_id)
        
        optimizations = query.order_by(
            Optimization.timestamp.desc()
        ).limit(limit).all()
        
        return jsonify({
            "count": len(optimizations),
            "data": [opt.to_dict() for opt in optimizations]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics"""
    try:
        total = Optimization.query.count()
        avg_profit = db.session.query(
            db.func.avg(Optimization.expected_profit)
        ).scalar() or 0
        
        return jsonify({
            "total_optimizations": total,
            "average_profit": round(avg_profit, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    """List unique products"""
    try:
        products = db.session.query(
            Optimization.product_id,
            db.func.count(Optimization.id).label('count')
        ).group_by(Optimization.product_id).all()
        
        return jsonify({
            "products": [
                {"product_id": p.product_id, "optimization_count": p.count}
                for p in products
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/view-history', methods=['GET'])
def view_history():
    try:
        with open("stored_data.json", "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "history": data})
    except:
        return jsonify({"success": False, "error": "No data found"})

@app.route('/admin/status')
def admin_status():
    """Quick visual check"""
    try:
        with app.app_context():
            count = Optimization.query.count()
            return f"""
            <html>
            <body style="font-family: Arial; padding: 40px; text-align: center;">
                <h1 style="color: green;">✅ Database Connected!</h1>
                <p style="font-size: 24px;">Total records: {count}</p>
                <a href="/api/health">View Health JSON</a> | 
                <a href="/api/history">View History</a>
            </body>
            </html>
            """
    except Exception as e:
        return f"""
        <html>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1 style="color: red;">❌ Database Error</h1>
            <p>{str(e)}</p>
        </body>
        </html>
        """, 500

def generate_sample_data(base_price: float = 1500,
                         num_competitors: int = 50,
                         variation: float = 0.3,
                         use_fixed_seed: bool = False) -> List[float]:
    
    if use_fixed_seed:
        random.seed(42)
    
    prices = []
    for _ in range(num_competitors):
        price = base_price * (1 + random.uniform(-variation, variation))
        prices.append(round(price, 2))
    
    if use_fixed_seed:
        random.seed()
    
    return prices

# ------------------------------ SERVER START -----------------------------

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=PORT, debug=False)