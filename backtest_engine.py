"""
🦇 BATMAN AI TRADING SYSTEM — BACKTEST ENGINE
================================================
Pandas-based trade simulation on test set. Equity curve, slippage,
commissions, Sharpe ratio, max drawdown, win rate, profit factor.
The Batcave never lies.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from model_core import load_saved_model, load_config

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] 🦇 %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BatmanBacktest")


def generate_test_predictions(model, X_test, y_test):
    """
    Generate probability predictions on test set (binary classification).

    Args:
        model: Trained PyTorch model (binary classifier with Sigmoid output)
        X_test: Test input sequences
        y_test: Test target values (binary 0/1)

    Returns:
        Tuple: (probabilities_array, actuals_array) both in [0, 1] space
    """
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        probabilities = model(X_test_t).cpu().numpy().flatten()
    actuals = y_test.flatten()
    return probabilities, actuals


def run_backtest(probabilities, actuals, config=None, prices=None):
    """
    Simulate trades on test set predictions (binary classification).

    Strategy:
        - BUY when probability >= prob_threshold (default 0.75)
        - SELL when probability <= (1 - prob_threshold) (default 0.25)
        - Apply slippage and commission per trade
        - Track equity curve from initial capital

    Args:
        probabilities: Array of UP probabilities (0-1)
        actuals: Array of actual binary outcomes (0/1)
        config: Config dict
        prices: Optional array of actual close prices for equity simulation

    Returns:
        Dict with equity_curve, trades, metrics DataFrames
    """
    if config is None:
        config = load_config()

    initial_capital = config.get("account_balance", 1000)
    slippage = config.get("slippage", 0.0005)
    commission = config.get("commission", 0.001)
    prob_thresh = config.get("prob_threshold", 0.75)
    stop_pct = config.get("stop_loss_pct", 0.015)
    tp_pct = config.get("take_profit_pct", 0.03)

    n = len(probabilities)
    equity = initial_capital
    cash = initial_capital
    position = 0  # shares held
    entry_price = 0.0
    trades = []
    equity_curve = [initial_capital]

    # If prices not provided, simulate simple directional returns
    has_prices = prices is not None and len(prices) == n

    for i in range(n - 1):
        prob = probabilities[i]
        actual_dir = actuals[i]  # 1 = up, 0 = down

        if has_prices:
            current_price = prices[i]
            next_price = prices[i + 1]

            # Check stop-loss / take-profit for open position
            if position > 0:
                if next_price <= entry_price * (1 - stop_pct):
                    sell_price = entry_price * (1 - stop_pct) * (1 - slippage)
                    proceeds = position * sell_price * (1 - commission)
                    pnl = proceeds - (position * entry_price)
                    trades.append({"step": i+1, "type": "STOP_LOSS", "price": sell_price,
                                   "shares": position, "pnl": pnl, "equity": cash + proceeds})
                    cash += proceeds
                    position = 0
                    entry_price = 0.0
                    equity = cash
                    equity_curve.append(equity)
                    continue
                if next_price >= entry_price * (1 + tp_pct):
                    sell_price = entry_price * (1 + tp_pct) * (1 - slippage)
                    proceeds = position * sell_price * (1 - commission)
                    pnl = proceeds - (position * entry_price)
                    trades.append({"step": i+1, "type": "TAKE_PROFIT", "price": sell_price,
                                   "shares": position, "pnl": pnl, "equity": cash + proceeds})
                    cash += proceeds
                    position = 0
                    entry_price = 0.0
                    equity = cash
                    equity_curve.append(equity)
                    continue

            # Generate signal from probability
            if prob >= prob_thresh and position == 0:
                buy_price = next_price * (1 + slippage)
                cost_per_share = buy_price * (1 + commission)
                shares = int(cash / cost_per_share)
                if shares > 0:
                    total_cost = shares * cost_per_share
                    cash -= total_cost
                    position = shares
                    entry_price = buy_price
                    trades.append({"step": i+1, "type": "BUY", "price": buy_price,
                                   "shares": shares, "pnl": 0, "equity": cash + position * next_price})
            elif prob <= (1.0 - prob_thresh) and position > 0:
                sell_price = next_price * (1 - slippage)
                proceeds = position * sell_price * (1 - commission)
                pnl = proceeds - (position * entry_price)
                cash += proceeds
                trades.append({"step": i+1, "type": "SELL", "price": sell_price,
                               "shares": position, "pnl": pnl, "equity": cash})
                position = 0
                entry_price = 0.0

            equity = cash + position * next_price
            equity_curve.append(equity)
        else:
            # Simple directional equity sim when no prices available
            # Bet direction based on probability, get +1% if correct, -1% if wrong
            predicted_dir = 1 if prob >= 0.5 else 0
            correct = (predicted_dir == actual_dir)
            daily_ret = 0.01 if correct else -0.01
            pnl = equity * daily_ret
            equity = equity * (1 + daily_ret)
            equity_curve.append(equity)
            
            # Log virtual trade for metrics correctly
            trades.append({
                "step": i+1,
                "type": "VIRTUAL_WIN" if correct else "VIRTUAL_LOSS", 
                "price": 0,
                "shares": 0, 
                "pnl": pnl, 
                "equity": equity
            })

    # Close any remaining position at end
    if has_prices and position > 0:
        final_price = prices[-1] * (1 - slippage)
        proceeds = position * final_price * (1 - commission)
        pnl = proceeds - (position * entry_price)
        cash += proceeds
        trades.append({
            "step": n - 1, "type": "CLOSE", "price": final_price,
            "shares": position, "pnl": pnl, "equity": cash,
        })
        position = 0
        equity = cash
        equity_curve[-1] = equity

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["step", "type", "price", "shares", "pnl", "equity"]
    )

    logger.info(f"✅ Backtest complete: {len(trades)} trades executed")

    return {
        "equity_curve": np.array(equity_curve),
        "trades": trades_df,
        "final_equity": equity,
        "initial_capital": initial_capital,
        "actuals": actuals,
        "predictions": probabilities,
    }


def compute_metrics(backtest_result, config=None):
    """
    Compute performance metrics from backtest results.

    Metrics: Total Return, Sharpe Ratio, Max Drawdown,
             Win Rate, Profit Factor, Total Trades

    Args:
        backtest_result: Dict from run_backtest
        config: Config dict

    Returns:
        Dict with computed metrics
    """
    if config is None:
        config = load_config()

    equity = backtest_result["equity_curve"]
    trades_df = backtest_result["trades"]
    initial = backtest_result["initial_capital"]
    final = backtest_result["final_equity"]
    risk_free = config.get("risk_free_rate", 0.02)

    # Total Return
    total_return = ((final - initial) / initial) * 100

    # Daily Returns
    daily_returns = np.diff(equity) / equity[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    # Sharpe Ratio (annualized, assuming 252 trading days)
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        excess_return = np.mean(daily_returns) - risk_free / 252
        sharpe = (excess_return / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = float(np.max(drawdown)) * 100 if len(drawdown) > 0 else 0.0

    # Trade Metrics
    if len(trades_df) > 0 and "pnl" in trades_df.columns:
        # Filter for trades that actually closed a position
        # Included VIRTUAL types for directional sim
        valid_types = ["SELL", "STOP_LOSS", "TAKE_PROFIT", "CLOSE", "VIRTUAL_WIN", "VIRTUAL_LOSS"]
        closed_trades = trades_df[trades_df["type"].isin(valid_types)]
        
        if len(closed_trades) > 0:
            wins = closed_trades[closed_trades["pnl"] > 0]
            losses = closed_trades[closed_trades["pnl"] <= 0]
            win_rate = (len(wins) / len(closed_trades)) * 100
            gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            win_rate = 0.0
            profit_factor = 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0

    total_trades = len(trades_df[trades_df["type"] == "BUY"]) if len(trades_df) > 0 else 0

    metrics = {
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "total_trades": total_trades,
        "final_equity": round(final, 2),
        "initial_capital": initial,
    }

    logger.info(f"📊 Metrics — Return: {total_return:+.2f}% | Sharpe: {sharpe:.3f} | "
                f"MaxDD: {max_dd:.2f}% | WinRate: {win_rate:.1f}% | PF: {profit_factor:.2f}")

    if sharpe < 1.5:
        logger.warning(f"⚠️ Sharpe ({sharpe:.3f}) below 1.5 target — review strategy")

    return metrics


def plot_backtest(backtest_result, config=None):
    """
    Plot equity curve and trades overlay on price chart.

    Args:
        backtest_result: Dict from run_backtest
        config: Config dict

    Returns:
        matplotlib Figure object
    """
    if config is None:
        config = load_config()

    equity = backtest_result["equity_curve"]
    trades_df = backtest_result["trades"]
    actuals = backtest_result["actuals"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]})

    # ─── Price + Trades ───
    ax1 = axes[0]
    ax1.plot(actuals, color="#00aaff", linewidth=1.2, label="Actual Price")

    if len(trades_df) > 0:
        buys = trades_df[trades_df["type"] == "BUY"]
        sells = trades_df[trades_df["type"].isin(["SELL", "TAKE_PROFIT", "CLOSE"])]
        stops = trades_df[trades_df["type"] == "STOP_LOSS"]

        if len(buys) > 0:
            ax1.scatter(buys["step"], buys["price"], marker="^", color="#f1c40f",
                        s=120, zorder=5, label="BUY", edgecolors="#000")
        if len(sells) > 0:
            ax1.scatter(sells["step"], sells["price"], marker="v", color="#2ecc71",
                        s=120, zorder=5, label="SELL/TP", edgecolors="#000")
        if len(stops) > 0:
            ax1.scatter(stops["step"], stops["price"], marker="x", color="#c0392b",
                        s=120, zorder=5, label="STOP-LOSS", linewidths=2)

    ax1.set_title("Batman Backtest — Price & Trades", fontsize=14, fontweight="bold", color="#f1c40f")
    ax1.set_ylabel("Price", color="#e0e0e0")
    ax1.legend(loc="upper left", facecolor="#1a1a2e", edgecolor="#333", labelcolor="#e0e0e0")
    ax1.set_facecolor("#0a0a0a")
    ax1.tick_params(colors="#e0e0e0")
    for spine in ax1.spines.values():
        spine.set_color("#333333")

    # ─── Equity Curve ───
    ax2 = axes[1]
    eq_x = np.arange(len(equity))
    ax2.fill_between(eq_x, equity, alpha=0.3, color="#00aaff")
    ax2.plot(eq_x, equity, color="#00aaff", linewidth=1.5, label="Equity")
    ax2.axhline(y=backtest_result["initial_capital"], color="#f1c40f",
                linestyle="--", alpha=0.5, label="Initial Capital")
    ax2.set_title("Equity Curve", fontsize=12, fontweight="bold", color="#f1c40f")
    ax2.set_xlabel("Time Steps", color="#e0e0e0")
    ax2.set_ylabel("Equity (£)", color="#e0e0e0")
    ax2.legend(loc="upper left", facecolor="#1a1a2e", edgecolor="#333", labelcolor="#e0e0e0")
    ax2.set_facecolor("#0a0a0a")
    ax2.tick_params(colors="#e0e0e0")
    for spine in ax2.spines.values():
        spine.set_color("#333333")

    fig.patch.set_facecolor("#0a0a0a")
    plt.tight_layout()

    return fig


def save_backtest_plot(fig, config=None):
    """Save backtest plot to file."""
    if config is None:
        config = load_config()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, config.get("backtest_plot_path", "backtest_plot.png"))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"💾 Backtest plot saved to {path}")


def run_full_backtest(config=None):
    """
    Full backtest pipeline: load model → predict on test → simulate → metrics → plot.

    Args:
        config: Config dict

    Returns:
        Tuple: (backtest_result, metrics, fig)
    """
    if config is None:
        config = load_config()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    logger.info(f"{'='*60}")
    logger.info("🦇 BATMAN BACKTEST ENGINE")
    logger.info(f"{'='*60}")

    # Load model
    model = load_saved_model(config)

    # Load test data
    seq_path = os.path.join(base_dir, config.get("sequences_path", "sequences.npz"))
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Sequences not found at {seq_path}. Run data pipeline first.")

    data = np.load(seq_path)
    X_test, y_test = data["X_test"], data["y_test"]

    # Generate predictions (probabilities, no inverse transform)
    predictions, actuals = generate_test_predictions(model, X_test, y_test)

    # Run backtest
    backtest_result = run_backtest(predictions, actuals, config)

    # Compute metrics
    metrics = compute_metrics(backtest_result, config)

    # Plot
    fig = plot_backtest(backtest_result, config)
    save_backtest_plot(fig, config)

    return backtest_result, metrics, fig


def run_walk_forward_backtest(res_df, config=None):
    """
    Simulate trading stats based on Walk-Forward Validation results.
    
    Args:
        res_df: DataFrame with 'Predicted', 'Actual', 'Date' index
        config: Config dict
        
    Returns:
        Dict with equity curve, metrics
    """
    if config is None:
        config = load_config()
        
    initial_capital = config.get("account_balance", 1000)
    
    equity = initial_capital
    equity_curve = [initial_capital]
    market_equity = initial_capital 
    
    # Simulate Long/Short logic for maximum realism based on direction
    # Pred > Prev => Long
    # Pred < Prev => Short
    
    for i in range(1, len(res_df)):
        today_row = res_df.iloc[i]
        
        pred_price = today_row["Predicted"]
        prev_close = today_row["Prev"]
        actual_close = today_row["Actual"]
        
        signal = 1 if pred_price > prev_close else -1
        daily_ret = (actual_close - prev_close) / prev_close
        trade_ret = daily_ret * signal
        
        equity = equity * (1 + trade_ret)
        market_equity = market_equity * (1 + daily_ret)
        equity_curve.append(equity)
        
    equity_curve = np.array(equity_curve)
    rolling_acc = res_df["Direction_Hit"].rolling(window=20).mean()
    
    return {
        "equity_curve": equity_curve,
        "final_equity": equity,
        "rolling_accuracy": rolling_acc,
        "total_return_pct": (equity - initial_capital) / initial_capital * 100
    }


if __name__ == "__main__":
    _, metrics, _ = run_full_backtest()
    print(f"\n{'='*50}")
    print("  🦇 BACKTEST RESULTS")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}")
