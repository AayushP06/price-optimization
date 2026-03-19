import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

_HERE       = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH   = os.path.join(_HERE, "online_retail_II.xlsx")
OUTPUT_PATH = os.path.join(_HERE, "cleaned_products.csv")

NON_PRODUCT_CODES = {
    "POST", "D", "M", "BANK CHARGES", "PADS", "DOT",
    "AMAZONFEE", "S", "ADJUST", "ADJUST2", "TEST001",
    "TEST002", "CRUK", "gift_0001_40", "gift_0001_30",
}
MIN_TRANSACTIONS = 10
COST_PRICE_RATIO = 0.60
FIXED_COST_RATIO = 0.05
MAX_COMP_PRICES = 50

def load_and_clean_sheet(path, sheet_name):
    print(f"Loading sheet: {sheet_name}...")
    df = pd.read_excel(path, sheet_name=sheet_name, dtype={"Invoice": str})

    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]
    df = df.dropna(subset=["Description"])
    df["Description"] = df["Description"].str.strip().str.title()

    keep_cols = ["StockCode", "Description", "Quantity", "Price"]
    df = df[keep_cols]

    print(f"  {sheet_name}: {len(df)} clean rows")
    return df

def aggregate_products(df):
    agg = df.groupby("StockCode").agg(
        Description       = ("Description", "first"),
        transaction_count = ("Quantity",    "count"),
        total_units_sold  = ("Quantity",    "sum"),
        min_price         = ("Price",       "min"),
        max_price         = ("Price",       "max"),
        median_price      = ("Price",       "median"),
        mean_price        = ("Price",       "mean"),
        std_price         = ("Price",       "std"),
    ).reset_index()

    agg["std_price"] = agg["std_price"].fillna(0.0)
    agg = agg[agg["transaction_count"] >= MIN_TRANSACTIONS]

    np.random.seed(42)
    agg["cost_price"]  = (agg["median_price"] * COST_PRICE_RATIO).round(4)
    agg["fixed_costs"] = (agg["cost_price"]   * FIXED_COST_RATIO).round(4)

    price_lists = (
        df.groupby("StockCode")["Price"]
          .apply(lambda x: sorted(x.drop_duplicates().tolist()))
          .reset_index()
          .rename(columns={"Price": "all_prices"})
    )
    agg = agg.merge(price_lists, on="StockCode", how="left")

    def _sample_prices(prices):
        if not isinstance(prices, list):
            return "[]"
        if len(prices) > MAX_COMP_PRICES:
            step = len(prices) // MAX_COMP_PRICES
            prices = prices[::step][:MAX_COMP_PRICES]
        return json.dumps([round(p, 4) for p in prices])

    agg["competitor_prices"] = agg["all_prices"].apply(_sample_prices)
    agg = agg.drop(columns=["all_prices"])

    for col in ["min_price", "max_price", "median_price", "mean_price",
                "std_price", "cost_price", "fixed_costs"]:
        agg[col] = agg[col].round(4)


    np.random.seed(101)
    num_rows = len(agg)
    

    categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports']
    agg['category'] = np.random.choice(categories, size=num_rows)
    

    agg['month'] = np.random.randint(1, 13, size=num_rows)
    

    ratings = np.random.normal(4.2, 0.8, size=num_rows)
    agg['rating'] = np.clip(ratings, 1.0, 5.0).round(1)
    

    agg['ad_spend'] = np.round(np.random.uniform(0, 5000, size=num_rows), 2)

    return agg

def main():
    if not os.path.exists(XLSX_PATH):
        print(f"File not found: {XLSX_PATH}")
        sys.exit(1)

    sheet1 = load_and_clean_sheet(XLSX_PATH, "Year 2009-2010")
    sheet2 = load_and_clean_sheet(XLSX_PATH, "Year 2010-2011")
    combined = pd.concat([sheet1, sheet2], ignore_index=True)
    del sheet1, sheet2

    print(f"Total clean rows: {len(combined)}")

    products = aggregate_products(combined)
    del combined

    products.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"Saved {len(products)} products to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
