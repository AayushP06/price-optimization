from __future__ import annotations

import os
import sys
import random
from typing import List, Optional

import joblib
import numpy as np
import xgboost as xgb

_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SERVICE_DIR, "..", "..", ".."))
_ML_DIR = os.path.join(_PROJECT_ROOT, "ml-model")

MODEL_PATH = os.path.join(_ML_DIR, "xgb_model.pkl")

class XGBPriceOptimizer:
    def __init__(self, model_path: str = MODEL_PATH):
        self._model: Optional[xgb.XGBRegressor] = None
        self._loaded: bool = False

        if os.path.exists(model_path):
            try:
                self._model = joblib.load(model_path)
                self._loaded = True
            except Exception:
                pass

    @property
    def is_ml_ready(self) -> bool:
        return self._loaded and self._model is not None

    def recommend(
        self,
        cost_price: float,
        fixed_costs: float = 0.0,
        competitor_prices: List[float] = None,
        min_margin: float = 5.0,
        max_margin: float = 80.0,
        category: str = 'Electronics',
        month: int = 6,
        rating: float = 4.2,
        ad_spend: float = 1000.0,
    ) -> dict:
        
        comp = sorted(competitor_prices) if competitor_prices else \
               sorted(round(cost_price * (1 + random.uniform(-0.25, 0.25)), 2)
                      for _ in range(50))

        price_lo = cost_price * (1 + min_margin / 100)
        price_hi = cost_price * (1 + max_margin / 100)
        c_min = comp[0]
        c_max = comp[-1]
        c_med = comp[len(comp) // 2]

        if self.is_ml_ready:
            best_price, expected_units, all_prices_data = self._ml_search(
                cost_price, fixed_costs, price_lo, price_hi,
                c_min, c_max, c_med,
                category, month, rating, ad_spend
            )
            method = "xgboost"
        else:
            best_price, all_prices_data = self._analytical_search(
                cost_price, fixed_costs, price_lo, price_hi,
                c_min, c_max
            )
            expected_units = self._demand_score(best_price, c_min, c_max) * 100
            method = "heuristic-fallback"

        info = self._describe(best_price, cost_price, fixed_costs, c_min, c_max, expected_units)

        return {
            "optimal_price":    info,
            "method":           method,
            "competitor_stats": {
                "count":  len(comp),
                "min":    round(c_min, 2),
                "max":    round(c_max, 2),
                "median": round(c_med, 2),
                "mean":   round(sum(comp) / len(comp), 2),
            },
            "competitor_prices": comp,
            "all_prices": all_prices_data,
        }

    def _demand_score(self, price: float, comp_min: float, comp_max: float) -> float:
        if comp_max <= comp_min:
            return 0.5
        pos = (price - comp_min) / (comp_max - comp_min)
        return max(0.05, min(1.0, 1.0 - pos))

    def _describe(
        self, price: float, cost: float, fixed: float,
        comp_min: float, comp_max: float, expected_units: float
    ) -> dict:
        profit_pu = max(0, price - cost - fixed)
        margin    = (profit_pu / price * 100) if price > 0 else 0
        return {
            "price":           round(price, 2),
            "profit_per_unit": round(profit_pu, 2),
            "margin":          round(margin, 2),
            "expected_units_sold": int(expected_units),
            "expected_profit": round(profit_pu * expected_units, 2),
        }

    def _ml_search(
        self, cost: float, fixed: float,
        price_lo: float, price_hi: float,
        c_min: float, c_max: float, c_med: float,
        category: str, month: int, rating: float, ad_spend: float
    ):
        best_price = price_lo
        best_profit = -1e9
        best_demand = 0
        all_prices_data = []
        
        n_sweep = 200
        prices = np.linspace(price_lo, price_hi, n_sweep)
        features = []
        
        cat_map = {'Electronics': 0, 'Clothing': 1, 'Home': 2, 'Beauty': 3, 'Sports': 4}
        cat_enc = cat_map.get(category, 0)
        
        for p in prices:
            margin_pct = (p - cost) / p if p > 0 else 0
            price_to_comp_med = p / c_med if c_med > 0 else 1.0
            comp_position = (p - c_min) / max(1e-9, c_max - c_min)
            
            features.append([
                cost,
                c_min,
                c_med,
                c_max,
                p,
                margin_pct,
                price_to_comp_med,
                comp_position,
                cat_enc,
                month,
                rating,
                ad_spend
            ])
            
        X_infer = np.array(features)
        y_log_pred = self._model.predict(X_infer)
        demand_pred = np.expm1(y_log_pred)
        
        for i, p in enumerate(prices):
            profit_pu = max(0, p - cost - fixed)
            demand = max(0, demand_pred[i])
            profit = profit_pu * demand
            
            margin_pct = (p - cost) / p if p > 0 else 0
            all_prices_data.append({
                "price": round(p, 2),
                "margin": round(margin_pct * 100, 2),
                "expected_profit": round(profit, 2)
            })
            
            if profit > best_profit:
                best_profit = profit
                best_price = p
                best_demand = demand

        return round(best_price, 2), max(1, best_demand), all_prices_data

    def _analytical_search(
        self, cost: float, fixed: float,
        price_lo: float, price_hi: float,
        comp_min: float, comp_max: float,
    ) -> float:
        best_price, best_profit = price_lo, -1e9
        all_prices_data = []
        for i in range(201):
            p = price_lo + (price_hi - price_lo) * i / 200
            profit_pu = max(0, p - cost - fixed)
            demand = self._demand_score(p, comp_min, comp_max) * 100
            profit = profit_pu * demand
            
            margin_pct = (p - cost) / p if p > 0 else 0
            all_prices_data.append({
                "price": round(p, 2),
                "margin": round(margin_pct * 100, 2),
                "expected_profit": round(profit, 2)
            })
            
            if profit > best_profit:
                best_profit, best_price = profit, p
        return round(best_price, 2), all_prices_data


_optimizer_instance: Optional[XGBPriceOptimizer] = None

def get_ml_optimizer() -> XGBPriceOptimizer:
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = XGBPriceOptimizer()
    return _optimizer_instance
