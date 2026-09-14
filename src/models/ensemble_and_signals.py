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
    - Calculates position and fees independently per ticker.
    - Aggregates daily returns across active stocks with equal weighting.
    - Calculates portfolio-level metrics (Cumulative Return, Sharpe, Drawdown, Win Rate).
    
    Strategy rules:
    - BUY  -> Long position (weight = +1.0)
    - HOLD -> Maintain previous position
    - SELL -> Cash / Flat position (weight = 0.0)
    - Compares performance against Buy & Hold benchmark.
    
    Returns:
    --------
    Dictionary with key performance indicators (Sharpe, Returns, Drawdown, etc.)
    """
    df = df.copy().reset_index(drop=True)
    df["signal"] = signals
    
    has_ticker = "ticker" in df.columns
    has_date = "date" in df.columns
    
    if has_ticker and has_date:
        df["date"] = pd.to_datetime(df["date"])
        ticker_groups = []
        total_trades = 0
        total_active_days = 0
        total_win_trades = 0
        
        for ticker, grp in df.groupby("ticker"):
            grp = grp.sort_values("date").copy()
            position = 0.0
            positions = []
            for sig in grp["signal"].values:
                if sig == "BUY":
                    position = 1.0
                elif sig == "SELL":
                    position = 0.0
                # If HOLD, maintain previous position
                positions.append(position)
            grp["position"] = positions
            pos_change = np.abs(np.diff(positions, prepend=0.0))
            grp["cost"] = pos_change * transaction_cost_pct
            
            actual_return = grp["forward_return_1d"].fillna(0.0).values
            grp["strategy_ret"] = (grp["position"].values * actual_return) - grp["cost"].values
            grp["benchmark_ret"] = actual_return
            
            total_trades += int(np.sum(pos_change > 0))
            active_mask = grp["position"].values > 0
            total_active_days += int(np.sum(active_mask))
            total_win_trades += int(np.sum(active_mask & (actual_return > 0)))
            
            ticker_groups.append(grp)
            
        full_df = pd.concat(ticker_groups, ignore_index=True)
        
        # Aggregate daily portfolio returns (equal-weighted across tickers on that date)
        daily_portfolio = full_df.groupby("date")[["strategy_ret", "benchmark_ret"]].mean().sort_index()
        strategy_daily_return = daily_portfolio["strategy_ret"].values
        benchmark_daily_return = daily_portfolio["benchmark_ret"].values
        
        strategy_cum = np.cumprod(1.0 + strategy_daily_return)
        benchmark_cum = np.cumprod(1.0 + benchmark_daily_return)
        
        strategy_total_return = float(strategy_cum[-1] - 1.0)
        benchmark_total_return = float(benchmark_cum[-1] - 1.0)
        
        strat_mean = np.mean(strategy_daily_return)
        strat_std = np.std(strategy_daily_return) + 1e-10
        sharpe_ratio = float((strat_mean / strat_std) * np.sqrt(252))
        
        bench_mean = np.mean(benchmark_daily_return)
        bench_std = np.std(benchmark_daily_return) + 1e-10
        bench_sharpe = float((bench_mean / bench_std) * np.sqrt(252))
        
        cum_max = np.maximum.accumulate(strategy_cum)
        drawdowns = (strategy_cum - cum_max) / cum_max
        max_drawdown = float(np.min(drawdowns))
        
        bench_cum_max = np.maximum.accumulate(benchmark_cum)
        bench_drawdowns = (benchmark_cum - bench_cum_max) / bench_cum_max
        bench_max_drawdown = float(np.min(bench_drawdowns))
        
        win_rate = float(total_win_trades / total_active_days) if total_active_days > 0 else 0.0
        
        return {
            "strategy_total_return": strategy_total_return,
            "benchmark_total_return": benchmark_total_return,
            "excess_return": strategy_total_return - benchmark_total_return,
            "strategy_sharpe": sharpe_ratio,
            "benchmark_sharpe": bench_sharpe,
            "strategy_max_drawdown": max_drawdown,
            "benchmark_max_drawdown": bench_max_drawdown,
            "win_rate": win_rate,
            "num_trades": total_trades,
            "active_days": total_active_days,
            "total_days": len(daily_portfolio)
        }
    
    # Fallback for single-series without ticker column
    position = 0.0
    positions = []
    for sig in signals:
        if sig == "BUY":
            position = 1.0
        elif sig == "SELL":
            position = 0.0
        positions.append(position)
        
    df["position"] = positions
    pos_change = np.abs(np.diff(positions, prepend=0.0))
    df["cost"] = pos_change * transaction_cost_pct
    
    actual_return = df["forward_return_1d"].fillna(0.0).values
    strategy_daily_return = (df["position"].values * actual_return) - df["cost"].values
    benchmark_daily_return = actual_return
    
    df["strategy_cum"] = (1.0 + strategy_daily_return).cumprod()
    df["benchmark_cum"] = (1.0 + benchmark_daily_return).cumprod()
    
    strategy_total_return = float(df["strategy_cum"].iloc[-1] - 1.0)
    benchmark_total_return = float(df["benchmark_cum"].iloc[-1] - 1.0)
    
    strat_mean = np.mean(strategy_daily_return)
    strat_std = np.std(strategy_daily_return) + 1e-10
    sharpe_ratio = float((strat_mean / strat_std) * np.sqrt(252))
    
    bench_mean = np.mean(benchmark_daily_return)
    bench_std = np.std(benchmark_daily_return) + 1e-10
    bench_sharpe = float((bench_mean / bench_std) * np.sqrt(252))
    
    cum_max = np.maximum.accumulate(df["strategy_cum"].values)
    drawdowns = (df["strategy_cum"].values - cum_max) / cum_max
    max_drawdown = float(np.min(drawdowns))
    
    bench_cum_max = np.maximum.accumulate(df["benchmark_cum"].values)
    bench_drawdowns = (df["benchmark_cum"].values - bench_cum_max) / bench_cum_max
    bench_max_drawdown = float(np.min(bench_drawdowns))
    
    num_trades = int(np.sum(pos_change > 0))
    active_days = int(np.sum(df["position"] > 0))
    win_trades = int(np.sum((df["position"] > 0) & (df["forward_return_1d"] > 0)))
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
        "total_days": len(df)
    }
