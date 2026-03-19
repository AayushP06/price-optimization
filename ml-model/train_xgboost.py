import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "cleaned_products.csv")
MODEL_PATH = os.path.join(_HERE, "xgb_model.pkl")

def prepare_data():
    if not os.path.exists(DATA_PATH):
        print(f"{DATA_PATH} not found.")
        sys.exit(1)
        
    df = pd.read_csv(DATA_PATH)
    
    features = []
    targets = []
    
    for _, row in df.iterrows():
        comp_prices_str = row.get("competitor_prices", "[]")
        try:
            comps = json.loads(comp_prices_str)
        except:
            comps = []
            
        if len(comps) == 0:
            continue
            
        c_min = min(comps)
        c_max = max(comps)
        c_med = np.median(comps)
        
        cost = row["cost_price"]
        price = row["median_price"]
        demand = row["transaction_count"]
        
        cat_map = {'Electronics': 0, 'Clothing': 1, 'Home': 2, 'Beauty': 3, 'Sports': 4}
        category_encoded = cat_map.get(row.get("category", "Electronics"), 0)
        month = int(row.get("month", 6))
        rating = float(row.get("rating", 4.2))
        ad_spend = float(row.get("ad_spend", 1000.0))
        
        if c_max <= c_min or price <= cost:
            continue
            
        margin_pct = (price - cost) / price
        price_to_comp_med = price / c_med
        comp_position = (price - c_min) / (c_max - c_min + 0.0001)
        
        features.append([
            cost,
            c_min,
            c_med,
            c_max,
            price,
            margin_pct,
            price_to_comp_med,
            comp_position,
            category_encoded,
            month,
            rating,
            ad_spend
        ])
        targets.append(demand)
        
    X = np.array(features)
    y = np.array(targets)
    
    y_log = np.log1p(y)
    
    return train_test_split(X, y_log, test_size=0.2, random_state=42)

def train_model():
    X_train, X_test, y_train, y_test = prepare_data()
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    
    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    train_model()
