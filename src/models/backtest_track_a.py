"""
Track A Trading Strategy Backtesting Pipeline (Phase 10 - Kevin).

Simulates realistic daily trading for Track A Classical ML Models:
- Logistic Regression
- Random Forest
- XGBoost
- Buy & Hold (Benchmark)

Features:
- IDX transaction fees (0.15% per trade)
- Position sizing: BUY -> Long (+1.0), SELL -> Cash (0.0), HOLD -> Maintain
- Metrics: Total Return, Excess Return, Sharpe Ratio, Max Drawdown, Win Rate, Trades
- Generates reports/trading_backtest_track_a.csv and reports/trading_backtest_all.csv
"""

import logging
import os
import sys
from typing import Dict, List
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.preparation import split_time_series
from src.models.ensemble_and_signals import backtest_trading_strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_track_a_backtesting(
    data_path: str = "data/processed/modeling_data_advanced.csv",
    reports_dir: str = "reports"
) -> pd.DataFrame:
    """
    Runs trading simulation for all Track A models on the Test Set.
    """
    os.makedirs(reports_dir, exist_ok=True)
    logger.info("=== Running Track A Trading Strategy Backtest (Test Set: 2025+) ===")
    
    # Load dataset and get test split
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    _, _, test_df = split_time_series(df)
    test_df = test_df.sort_values(by=["date", "ticker"]).reset_index(drop=True)
    
    models = ["logistic_regression", "random_forest", "xgboost"]
    results = []
    
    benchmark_calculated = False
    
    for model_name in models:
        pred_csv = os.path.join(reports_dir, f"predictions_{model_name}.csv")
        if not os.path.exists(pred_csv):
            logger.warning(f"Prediction file {pred_csv} not found, skipping...")
            continue
            
        pred_df = pd.read_csv(pred_csv)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        
        merged = pd.merge(
            test_df[["date", "ticker", "forward_return_1d"]],
            pred_df[["date", "ticker", "signal"]],
            on=["date", "ticker"],
            how="inner"
        ).sort_values(by=["date", "ticker"]).reset_index(drop=True)
        
        bt = backtest_trading_strategy(merged, merged["signal"].values)
        
        # Add Benchmark once
        if not benchmark_calculated:
            results.append({
                "Model": "Buy & Hold (Benchmark)",
                "Total Return (%)": bt["benchmark_total_return"] * 100,
                "Excess Return (%)": 0.0,
                "Sharpe Ratio": bt["benchmark_sharpe"],
                "Max Drawdown (%)": bt["benchmark_max_drawdown"] * 100,
                "Win Rate (%)": 0.0,
                "Trades": 0
            })
            benchmark_calculated = True
            
        display_name = {
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
            "xgboost": "XGBoost"
        }.get(model_name, model_name)
        
        results.append({
            "Model": display_name,
            "Total Return (%)": bt["strategy_total_return"] * 100,
            "Excess Return (%)": bt["excess_return"] * 100,
            "Sharpe Ratio": bt["strategy_sharpe"],
            "Max Drawdown (%)": bt["strategy_max_drawdown"] * 100,
            "Win Rate (%)": bt["win_rate"] * 100,
            "Trades": bt["num_trades"]
        })
        
    backtest_df = pd.DataFrame(results)
    out_path = os.path.join(reports_dir, "trading_backtest_track_a.csv")
    backtest_df.to_csv(out_path, index=False)
    logger.info(f"Saved Track A backtest report to {out_path}")
    
    # Merge with Track B backtest if available
    track_b_path = os.path.join(reports_dir, "trading_backtest_track_b.csv")
    if os.path.exists(track_b_path):
        b_df = pd.read_csv(track_b_path)
        # Exclude benchmark from Track B to avoid duplicate
        b_models = b_df[b_df["Model"] != "Buy & Hold (Benchmark)"]
        all_backtest = pd.concat([backtest_df, b_models], ignore_index=True)
        all_out = os.path.join(reports_dir, "trading_backtest_all.csv")
        all_backtest.to_csv(all_out, index=False)
        logger.info(f"Saved Unified Backtest report to {all_out}")
        return all_backtest
        
    return backtest_df


if __name__ == "__main__":
    df_results = run_track_a_backtesting()
    print("\n=======================================================")
    print("      TRADING STRATEGY BACKTEST RESULTS (TEST SET)     ")
    print("=======================================================")
    print(df_results.to_string(index=False))
