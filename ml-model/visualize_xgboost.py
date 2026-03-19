

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(_HERE, "cleaned_products.csv")
MODEL_PATH = os.path.join(_HERE, "xgb_model.pkl")
CHART_PATH = os.path.join(_HERE, "xgb_visualizations.png")

FEATURE_NAMES = [
    "Cost Price",
    "Competitor Min",
    "Competitor Median",
    "Competitor Max",
    "Product Price",
    "Margin %",
    "Price / Comp Median",
    "Market Position",
]

def prepare_data():
    df = pd.read_csv(DATA_PATH)
    features, targets = [], []

    for _, row in df.iterrows():
        try:
            comps = json.loads(row.get("competitor_prices", "[]"))
        except Exception:
            comps = []
        if len(comps) == 0:
            continue

        c_min  = min(comps)
        c_max  = max(comps)
        c_med  = float(np.median(comps))
        cost   = float(row["cost_price"])
        price  = float(row["median_price"])
        demand = float(row["transaction_count"])

        if c_max <= c_min or price <= cost:
            continue

        margin_pct       = (price - cost) / price
        price_to_comp_med = price / c_med if c_med > 0 else 1.0
        comp_position    = (price - c_min) / (c_max - c_min + 1e-9)

        features.append([cost, c_min, c_med, c_max, price,
                         margin_pct, price_to_comp_med, comp_position])
        targets.append(demand)

    X = np.array(features)
    y = np.log1p(np.array(targets))
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    print("\n" + "=" * 60)
    print("  XGBoost Visualizations")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. Run train_xgboost.py first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    print("✓ Model loaded")

    X_train, X_test, y_train, y_test = prepare_data()
    y_pred_log = model.predict(X_test)

    y_actual  = np.expm1(y_test)
    y_pred    = np.expm1(y_pred_log)
    residuals = y_actual - y_pred

    mse = mean_squared_error(y_test, y_pred_log)
    r2  = r2_score(y_test, y_pred_log)
    print(f"  Test MSE (log scale) : {mse:.4f}")
    print(f"  Test R²  (log scale) : {r2:.4f}\n")

   
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("XGBoost Price Optimization — Model Diagnostics",
                 fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

   
    ax1 = fig.add_subplot(gs[0, 0])
    importance = model.get_booster().get_score(importance_type="gain")
  
    imp_values = [importance.get(f"f{i}", 0) for i in range(len(FEATURE_NAMES))]
    total      = sum(imp_values) or 1
    imp_norm   = [v / total * 100 for v in imp_values]
    sorted_pairs = sorted(zip(FEATURE_NAMES, imp_norm), key=lambda x: x[1])
    names, vals  = zip(*sorted_pairs)
    bars = ax1.barh(names, vals, color=plt.cm.viridis(
        [v / max(vals) for v in vals]))
    ax1.set_xlabel("Relative Importance (% gain)")
    ax1.set_title("Feature Importance")
    for bar, val in zip(bars, vals):
        ax1.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="x")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_test, y_pred_log, alpha=0.3, s=12,
                color="#3498db", edgecolors="none")
    lo = min(y_test.min(), y_pred_log.min()) - 0.2
    hi = max(y_test.max(), y_pred_log.max()) + 0.2
    ax2.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax2.set_xlabel("Actual log(demand)"); ax2.set_ylabel("Predicted log(demand)")
    ax2.set_title(f"Predicted vs Actual\n(log scale, R²={r2:.3f})")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=60, color="#e67e22", edgecolor="white",
             linewidth=0.4, alpha=0.85)
    ax3.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax3.set_xlabel("Residual (actual − predicted demand)")
    ax3.set_ylabel("Count")
    ax3.set_title("Residuals Distribution")
    ax3.grid(True, alpha=0.3)
    ax4 = fig.add_subplot(gs[1, 0:2])
    df = pd.read_csv(DATA_PATH)
    sample = df.iloc[0]
    comps  = json.loads(sample["competitor_prices"])
    c_min, c_max, c_med = min(comps), max(comps), float(np.median(comps))
    cost   = float(sample["cost_price"])
    fixed  = float(sample["fixed_costs"])

    price_lo = cost * 1.05
    price_hi = cost * 1.80
    sweep    = np.linspace(price_lo, price_hi, 200)
    feats    = []
    for p in sweep:
        margin          = (p - cost) / p if p > 0 else 0
        p_to_cm         = p / c_med if c_med > 0 else 1.0
        comp_pos        = (p - c_min) / max(1e-9, c_max - c_min)
        feats.append([cost, c_min, c_med, c_max, p, margin, p_to_cm, comp_pos])

    pred_demand = np.expm1(model.predict(np.array(feats)))
    profit_pu   = np.maximum(0, sweep - cost - fixed)
    exp_profit  = profit_pu * pred_demand
    best_idx    = np.argmax(exp_profit)

    ax4_twin = ax4.twinx()
    ax4.plot(sweep, pred_demand, color="#3498db", linewidth=2, label="Predicted Demand (units)")
    ax4_twin.plot(sweep, exp_profit, color="#27ae60", linewidth=2,
                  linestyle="--", label="Expected Profit (£)")
    ax4.axvline(sweep[best_idx], color="#e74c3c", linewidth=1.8, linestyle=":",
                label=f"Optimal Price £{sweep[best_idx]:.2f}")
    ax4.set_xlabel("Price (£)"); ax4.set_ylabel("Predicted Demand (units)", color="#3498db")
    ax4_twin.set_ylabel("Expected Profit (£)", color="#27ae60")
    ax4.set_title(f"Price vs Demand & Profit  —  {sample['Description'][:40]}")
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax4.grid(True, alpha=0.3)
    ax5 = fig.add_subplot(gs[1, 2])
    buckets = pd.qcut(y_actual, q=5, labels=["Very Low", "Low", "Mid", "High", "Very High"])
    bucket_rmse = (
        pd.DataFrame({"bucket": buckets,
                      "abs_err": np.abs(residuals)})
          .groupby("bucket", observed=True)["abs_err"]
          .mean()
    )
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bucket_rmse)))
    ax5.bar(bucket_rmse.index, bucket_rmse.values, color=colors, edgecolor="white")
    ax5.set_xlabel("Demand Bucket"); ax5.set_ylabel("Mean Absolute Error (units)")
    ax5.set_title("MAE by Demand Bucket")
    ax5.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    print(f"✓ Saved → {CHART_PATH}")
    plt.show()
    print("✅ Done!\n" + "=" * 60)


if __name__ == "__main__":
    main()
