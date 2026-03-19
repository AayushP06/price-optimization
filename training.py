"""
training.py — RL-based Dynamic Price Optimization (Real Data)

Trains a Q-Learning agent on real product data from the Online Retail II dataset.
Run data_cleaning.py first to generate cleaned_products.csv.

Workflow:
  1. Load cleaned_products.csv (one row per real product)
  2. For each product, build a PricingEnvironment from its real price distribution
  3. Train the shared Q-Learning agent across all products
  4. Plot learning curve, Q-value heatmap, and profit landscape
  5. Save trained Q-table → ml-model/q_table.npy

Usage:
    python training.py
"""

import sys
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ML_DIR = os.path.join(_HERE, "ml-model")
for _p in [_HERE, _ML_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl_environment import PricingEnvironment
from rl_agent import QLearningAgent, train as rl_train


# ─── CONFIG ──────────────────────────────────────────────────────

CLEANED_CSV   = os.path.join(_ML_DIR, "cleaned_products.csv")
Q_TABLE_PATH  = os.path.join(_ML_DIR, "q_table.npy")
CHART_PATH    = os.path.join(_ML_DIR, "rl_training_results.png")

N_EPISODES_PER_PRODUCT = 300   # episodes per product (× n_products = total)
LOG_EVERY              = 500   # log every N episodes across all products
MIN_MARGIN             = 5.0   # % above cost — lower price bound
MAX_MARGIN             = 80.0  # % above cost — upper price bound

# How many products to train on (None = all)
MAX_PRODUCTS = None


# ─── DATA LOADING ────────────────────────────────────────────────

def load_products(path: str) -> list:
    """Load cleaned products CSV and return list of product dicts."""
    if not os.path.exists(path):
        print(f"\n❌ {path} not found.")
        print("   Please run:  python ml-model/data_cleaning.py  first.\n")
        sys.exit(1)

    df = pd.read_csv(path)
    products = []
    for _, row in df.iterrows():
        try:
            comp_prices = json.loads(row["competitor_prices"])
        except Exception:
            comp_prices = []
        if not comp_prices or row["cost_price"] <= 0:
            continue
        products.append({
            "stock_code":   row["StockCode"],
            "description":  row["Description"],
            "cost_price":   float(row["cost_price"]),
            "fixed_costs":  float(row["fixed_costs"]),
            "median_price": float(row["median_price"]),
            "competitor_prices": comp_prices,
        })
    return products


# ─── ROLLING MEAN ────────────────────────────────────────────────

def rolling_mean(data: list, window: int = 200) -> np.ndarray:
    out = np.empty(len(data))
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out[i] = np.mean(data[lo : i + 1])
    return out


# ─── VISUALISATION ───────────────────────────────────────────────

def plot_results(episode_rewards: list, agent: QLearningAgent,
                 sample_env: PricingEnvironment, n_products: int):
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        f"Q-Learning Price Optimization — {n_products} Real Products "
        f"({len(episode_rewards):,} Total Episodes)",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 3, wspace=0.38)

    # ── 1. Learning Curve ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    rewards    = np.array(episode_rewards)
    rm_window  = max(50, len(rewards) // 20)
    rm         = rolling_mean(episode_rewards, window=rm_window)
    ax1.plot(rewards, alpha=0.2, color="#3498db", linewidth=0.5, label="Episode Reward")
    ax1.plot(rm, color="#e67e22", linewidth=2.0, label=f"{rm_window}-ep Rolling Avg")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward (£)")
    ax1.set_title("Learning Curve"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ── 2. Q-value Heatmap ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    q_slice = agent.q_table[:, :, 5, :].max(axis=2)   # comp_bin=5 slice
    im = ax2.imshow(q_slice.T, origin="lower", aspect="auto",
                    cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax2, shrink=0.8)
    ax2.set_xlabel("Price Bin (low→high)")
    ax2.set_ylabel("Demand Bin (low→high)")
    ax2.set_title("Q-value Heatmap\n(best action, comp_bin=5)")

    # ── 3. Profit Landscape for sample product ───────────────────
    ax3 = fig.add_subplot(gs[2])
    prices  = np.linspace(sample_env.price_lo, sample_env.price_hi, 200)
    profits = [sample_env.describe_price(p)["expected_profit"] for p in prices]
    opt_p   = sample_env.get_optimal_price()
    opt_r   = sample_env.describe_price(opt_p)["expected_profit"]
    ax3.plot(prices, profits, color="#27ae60", linewidth=2)
    ax3.axvline(opt_p, color="#e74c3c", linestyle="--", linewidth=1.8,
                label=f"Optimal £{opt_p:.2f}")
    ax3.scatter([opt_p], [opt_r], color="#e74c3c", zorder=5, s=80)
    ax3.set_xlabel("Price (£)"); ax3.set_ylabel("Expected Profit (£)")
    ax3.set_title("Profit Landscape\n(sample product)"); ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    print(f"✓ Chart saved → {CHART_PATH}")
    plt.show()


# ─── DEMO ────────────────────────────────────────────────────────

def run_demo(agent: QLearningAgent, env: PricingEnvironment, product: dict):
    print("\n" + "=" * 60)
    print("  DEMO — Trained Agent on a Real Product")
    print("=" * 60)
    print(f"  Product  : {product['description']}  [{product['stock_code']}]")
    print(f"  Cost     : £{env.cost_price:.4f}")
    print(f"  Fixed    : £{env.fixed_costs:.4f}")
    print(f"  Market   : £{env.comp_min:.4f} – £{env.comp_max:.4f}  "
          f"(median £{env.comp_med:.4f})")

    state = env.reset()
    best_price  = env.price
    best_profit = env.describe_price(env.price)["expected_profit"]
    for _ in range(50):
        action              = agent.best_action(state)
        state, _, done      = env.step(action)
        info                = env.describe_price(env.price)
        if info["expected_profit"] > best_profit:
            best_profit = info["expected_profit"]
            best_price  = env.price
        if done:
            break

    rec = env.describe_price(best_price)
    ana = env.describe_price(env.get_optimal_price())

    print(f"\n  ── RL Agent Recommendation ──")
    print(f"  Recommended Price : £{rec['price']:.4f}")
    print(f"  Profit Margin     : {rec['margin_pct']:.1f} %")
    print(f"  Est. Demand       : {rec['demand_score']:.1f} units")
    print(f"  Expected Profit   : £{rec['expected_profit']:.2f}")
    print(f"\n  ── Analytical Optimum (ground truth) ──")
    print(f"  Optimal Price     : £{ana['price']:.4f}")
    print(f"  Expected Profit   : £{ana['expected_profit']:.2f}")
    gap = abs(rec["price"] - ana["price"]) / max(1e-9, ana["price"]) * 100
    print(f"\n  Price Gap vs Analytical : {gap:.1f} %")
    print("=" * 60)


# ─── MAIN ────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Q-LEARNING PRICE OPTIMIZATION — Real Product Data")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    # 1. Load products
    products = load_products(CLEANED_CSV)
    if MAX_PRODUCTS:
        products = products[:MAX_PRODUCTS]
    n = len(products)
    print(f"\n✓ Loaded {n:,} products from cleaned dataset")
    print(f"  Episodes per product : {N_EPISODES_PER_PRODUCT}")
    print(f"  Total episodes       : {n * N_EPISODES_PER_PRODUCT:,}\n")

    # 2. Build a sample environment (for plotting later)
    sample_p = products[0]
    sample_env = PricingEnvironment(
        cost_price=sample_p["cost_price"],
        fixed_costs=sample_p["fixed_costs"],
        competitor_prices=sample_p["competitor_prices"],
        min_margin=MIN_MARGIN,
        max_margin=MAX_MARGIN,
    )

    # 3. Build shared agent
    agent = QLearningAgent(
        state_shape=sample_env.state_shape,
        n_actions=sample_env.n_actions,
        alpha=0.10,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
    )
    total_episodes = n * N_EPISODES_PER_PRODUCT
    agent.epsilon_decay = (agent.epsilon - agent.epsilon_min) / total_episodes
    print(f"✓ Q-table shape: {agent.q_table.shape}")
    print(f"  Training…\n")

    # 4. Train across all products
    all_rewards = []
    for idx, product in enumerate(products):
        env = PricingEnvironment(
            cost_price=product["cost_price"],
            fixed_costs=product["fixed_costs"],
            competitor_prices=product["competitor_prices"],
            min_margin=MIN_MARGIN,
            max_margin=MAX_MARGIN,
        )
        ep_rewards = rl_train(agent, env, n_episodes=N_EPISODES_PER_PRODUCT, log_every=999999)
        all_rewards.extend(ep_rewards)

        # Progress log every ~10% of products
        if (idx + 1) % max(1, n // 10) == 0 or idx == n - 1:
            avg = np.mean(all_rewards[-500:]) if len(all_rewards) >= 500 else np.mean(all_rewards)
            print(
                f"  Product {idx+1:>4}/{n}  |  "
                f"ε = {agent.epsilon:.3f}  |  "
                f"Avg Reward (last 500 eps): {avg:>10.2f}"
            )

    # 5. Stats and save
    print("\n" + "-" * 60)
    print("  Q-TABLE STATS:")
    for k, v in agent.q_value_summary().items():
        print(f"    {k:<24}: {v}")
    agent.save(Q_TABLE_PATH)

    # 6. Demo on the last product seen
    run_demo(agent, sample_env, sample_p)

    # 7. Plots
    print("\n📊 Generating plots…")
    plot_results(all_rewards, agent, sample_env, n)

    print("\n✅ Training complete!\n" + "=" * 60)


if __name__ == "__main__":
    main()
