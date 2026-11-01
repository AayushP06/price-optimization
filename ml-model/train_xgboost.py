"""Backward-compatible entry point; use train_xgb.py for XGBoost training."""
from train_xgb import train_model

if __name__ == "__main__":
    train_model()
