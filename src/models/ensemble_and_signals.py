"""
Ensemble & Trading Signal Generation Module (Track B - Daf).

Features:
- Weighted model ensembling (LightGBM + LSTM/GRU)
- P(Up) to BUY / HOLD / SELL signal conversion
- Threshold tuning on Validation set
- Realistic trading backtest simulation vs Buy & Hold benchmark
- Calculates Sharpe Ratio, Cumulative Return, Max Drawdown, Win Rate
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def generate_trading_signals(
    probabilities: np.ndarray,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts probabilities P(Up) into binary predictions and discrete trading signals.
    
    Returns:
    --------
    Tuple: (predictions: int [0 or 1], signals: str ['BUY', 'HOLD', 'SELL'])
    """
    probs = np.asarray(probabilities, dtype=float)
    predictions = (probs >= 0.50).astype(int)
    
    signals = np.full(len(probs), "HOLD", dtype=object)
    signals[probs >= buy_threshold] = "BUY"
    signals[probs <= sell_threshold] = "SELL"
    
    return predictions, signals


def compute_ensemble_probabilities(
    prob_dict: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Computes weighted average ensemble probability P(Up).
    """
    if weights is None:
        # Equal weights
        n = len(prob_dict)
        weights = {k: 1.0 / n for k in prob_dict}
        
    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    norm_weights = {k: v / total_weight for k, v in weights.items()}
    
    first_key = list(prob_dict.keys())[0]
    ensemble_prob = np.zeros(len(prob_dict[first_key]), dtype=float)
    
    for model_name, probs in prob_dict.items():
        w = norm_weights.get(model_name, 0.0)
        ensemble_prob += w * np.asarray(probs, dtype=float)
        
    return np.clip(ensemble_prob, 0.0, 1.0)


def backtest_trading_strategy(
    df: pd.DataFrame,
    signals: np.ndarray,
    transaction_cost_pct: float = 0.0015  # 0.15% IDX fee per trade
) -> Dict[str, float]:
    """
    Simulates a daily trading strategy based on BUY/HOLD/SELL signals.
    
    Multi-ticker safe:
    - Tracks positions per ticker independently (eliminates cross-ticker position bleed).
    - Aggregates equal-weighted daily returns across active tickers.
    - Compares performance against Buy & Hold benchmark on the same universe & timeframe.
    
    Strategy rules:
    - BUY  -> Long position (weight = +1.0)
    - HOLD -> Maintain previous position for that ticker
    - SELL -> Cash / Flat position (weight = 0.0)
    - Fee  -> 0.15% IDX fee on position change (|pos_t - pos_{t-1}| * 0.0015)
    
    Returns:
    --------
    Dictionary with key performance indicators (Sharpe, Returns, Drawdown, Win Rate, etc.)
    """
    df = df.copy().reset_index(drop=True)
    df["signal"] = signals
    df["date"] = pd.to_datetime(df["date"])
    
    has_ticker = "ticker" in df.columns
    
    # Track positions and trade costs per ticker independently
    if has_ticker:
        processed_dfs = []
        for ticker, grp in df.groupby("ticker", sort=False):
            grp = grp.sort_values(by="date").copy()
            position = 0.0
            positions = []
            for sig in grp["signal"]:
                if sig == "BUY":
                    position = 1.0
                elif sig == "SELL":
                    position = 0.0
                # If HOLD, position remains unchanged for this ticker
                positions.append(position)
            
            grp["position"] = positions
            pos_change = np.abs(np.diff(positions, prepend=0.0))
            grp["pos_change"] = pos_change
            grp["cost"] = pos_change * transaction_cost_pct
            
            actual_ret = grp["forward_return_1d"].fillna(0.0).values
            grp["strategy_ret"] = (grp["position"].values * actual_ret) - grp["cost"].values
            grp["benchmark_ret"] = actual_ret
            processed_dfs.append(grp)
            
        sim_df = pd.concat(processed_dfs, ignore_index=True)
    else:
        sim_df = df.sort_values(by="date").copy()
        position = 0.0
        positions = []
        for sig in sim_df["signal"]:
            if sig == "BUY":
                position = 1.0
            elif sig == "SELL":
                position = 0.0
            positions.append(position)
        sim_df["position"] = positions
        pos_change = np.abs(np.diff(positions, prepend=0.0))
        sim_df["pos_change"] = pos_change
        sim_df["cost"] = pos_change * transaction_cost_pct
        actual_ret = sim_df["forward_return_1d"].fillna(0.0).values
        sim_df["strategy_ret"] = (sim_df["position"].values * actual_ret) - sim_df["cost"].values
        sim_df["benchmark_ret"] = actual_ret

    # Daily portfolio-level aggregation (Equal-weight across active tickers per date)
    daily_portfolio = sim_df.groupby("date").agg({
        "strategy_ret": "mean",
        "benchmark_ret": "mean"
    }).sort_index()
    
    strategy_daily_return = daily_portfolio["strategy_ret"].values
    benchmark_daily_return = daily_portfolio["benchmark_ret"].values
    
    # Cumulative curves
    strategy_cum = (1.0 + strategy_daily_return).cumprod()
    benchmark_cum = (1.0 + benchmark_daily_return).cumprod()
    
    # Total returns
    strategy_total_return = float(strategy_cum[-1] - 1.0) if len(strategy_cum) > 0 else 0.0
    benchmark_total_return = float(benchmark_cum[-1] - 1.0) if len(benchmark_cum) > 0 else 0.0
    
    # Annualized Sharpe (assuming 252 trading days)
    strat_mean = np.mean(strategy_daily_return) if len(strategy_daily_return) > 0 else 0.0
    strat_std = np.std(strategy_daily_return) + 1e-10 if len(strategy_daily_return) > 0 else 1e-10
    sharpe_ratio = float((strat_mean / strat_std) * np.sqrt(252))
    
    bench_mean = np.mean(benchmark_daily_return) if len(benchmark_daily_return) > 0 else 0.0
    bench_std = np.std(benchmark_daily_return) + 1e-10 if len(benchmark_daily_return) > 0 else 1e-10
    bench_sharpe = float((bench_mean / bench_std) * np.sqrt(252))
    
    # Max Drawdown
    cum_max = np.maximum.accumulate(strategy_cum) if len(strategy_cum) > 0 else np.array([1.0])
    drawdowns = (strategy_cum - cum_max) / cum_max if len(strategy_cum) > 0 else np.array([0.0])
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    bench_cum_max = np.maximum.accumulate(benchmark_cum) if len(benchmark_cum) > 0 else np.array([1.0])
    bench_drawdowns = (benchmark_cum - bench_cum_max) / bench_cum_max if len(benchmark_cum) > 0 else np.array([0.0])
    bench_max_drawdown = float(np.min(bench_drawdowns)) if len(bench_drawdowns) > 0 else 0.0
    
    # Trade statistics
    num_trades = int(np.sum(sim_df["pos_change"] > 0))
    active_days = int(np.sum(sim_df["position"] > 0))
    win_trades = int(np.sum((sim_df["position"] > 0) & (sim_df["forward_return_1d"] > 0)))
    win_rate = float(win_trades / active_days) if active_days > 0 else 0.0
    
    return {
        "strategy_total_return": strategy_total_return,
        "benchmark_total_return": benchmark_total_return,
        "excess_return": strategy_total_return - benchmark_total_return,
        "strategy_sharpe": sharpe_ratio,
        "benchmark_sharpe": bench_sharpe,
        "strategy_max_drawdown": max_drawdown,
        "benchmark_max_drawdown": bench_max_drawdown,
        "win_rate": win_rate,
        "num_trades": num_trades,
        "active_days": active_days,
        "total_days": int(sim_df["date"].nunique())
    }

