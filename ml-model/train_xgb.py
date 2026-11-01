"""Train and save the JSON XGBoost model and its feature contract."""
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
from app.services.xgb_optimizer import FEATURE_COLUMNS, MODEL_PATH, FEATURE_PATH

DATA_PATH = os.path.join(HERE, "cleaned_products.csv")


def prepare_data():
    df = pd.read_csv(DATA_PATH)
    rows, targets = [], []
    cat_map = {"Electronics": 0, "Clothing": 1, "Home": 2, "Beauty": 3, "Sports": 4}
    for _, item in df.iterrows():
        try:
            comps = json.loads(item.get("competitor_prices", "[]"))
        except (TypeError, json.JSONDecodeError):
            continue
        if not comps:
            continue
        low, high, median = min(comps), max(comps), float(np.median(comps))
        cost, price, demand = float(item["cost_price"]), float(item["median_price"]), float(item["transaction_count"])
        if high <= low or price <= cost:
            continue
        values = {"cost_price": cost, "competitor_min": low, "competitor_median": median,
                  "competitor_max": high, "price": price, "margin_pct": (price-cost)/price,
                  "price_to_competitor_median": price/median,
                  "competitor_position": (price-low)/(high-low+0.0001),
                  "category_encoded": cat_map.get(item.get("category", "Electronics"), 0),
                  "month": int(item.get("month", 6)), "rating": float(item.get("rating", 4.2)),
                  "ad_spend": float(item.get("ad_spend", 1000))}
        rows.append([values[column] for column in FEATURE_COLUMNS])
        targets.append(np.log1p(demand))
    return train_test_split(np.asarray(rows), np.asarray(targets), test_size=.2, random_state=42)


def train_model():
    X_train, X_test, y_train, y_test = prepare_data()
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=300,
                             learning_rate=.05, max_depth=6, subsample=.8,
                             colsample_bytree=.8, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    model.save_model(MODEL_PATH)
    with open(FEATURE_PATH, "w", encoding="utf-8") as feature_file:
        json.dump(FEATURE_COLUMNS, feature_file, indent=2)
    return model


if __name__ == "__main__":
    train_model()
