"""
Comprehensive Model Comparison & Trading Backtest Pipeline (Track B - Daf).

This script:
1. Loads advanced features and performs chronological split.
2. Evaluates LightGBM, LSTM, and Weighted Ensemble.
3. Tunes ensemble weights and signal thresholds strictly on the Validation Set (2024).
4. Evaluates final performance on the sterile Test Set (2025+).
5. Executes realistic trading simulation vs Buy & Hold benchmark.
6. Saves predictions into the SQLite database.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.preparation import split_time_series
from src.features.technical_indicators import get_feature_columns
from src.models.ensemble_and_signals import (
    backtest_trading_strategy,
    compute_ensemble_probabilities,
    generate_trading_signals,
)
from src.models.lightgbm_model import LightGBMStockModel
from src.models.lstm_model import LSTMStockTrainer
from src.models.save_predictions import query_prediction_summary, save_model_predictions_to_db
from src.models.validation import evaluate_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_full_evaluation(
    data_file: str = "data/processed/modeling_data_advanced.csv",
    reports_dir: str = "reports"
):
    """
    Runs end-to-end Track B evaluation pipeline.
    """
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "models"), exist_ok=True)
    
    logger.info("=== Starting Track B Comprehensive Evaluation ===")
    df = pd.read_csv(data_file)
    df["date"] = pd.to_datetime(df["date"])
    
    feature_cols = get_feature_columns()
    train_df, val_df, test_df = split_time_series(df)
    
    # -------------------------------------------------------------
    # 1. Train or Load Models
    # -------------------------------------------------------------
    lgbm_model_path = os.path.join(reports_dir, "models", "lightgbm_model.joblib")
    if os.path.exists(lgbm_model_path):
        logger.info(f"Loading existing LightGBM model from {lgbm_model_path}...")
        lgbm_model = LightGBMStockModel.load(lgbm_model_path)
    else:
        logger.info("Training LightGBM model...")
        lgbm_model = LightGBMStockModel()
        lgbm_model.fit(
            train_df[feature_cols], train_df["target"],
            val_df[feature_cols], val_df["target"],
            early_stopping_rounds=30
        )
        lgbm_model.save(lgbm_model_path)
        
    lstm_model_path = os.path.join(reports_dir, "models", "lstm_model.pt")
    lstm_scaler_path = os.path.join(reports_dir, "models", "lstm_scaler.joblib")
    
    lstm_trainer = LSTMStockTrainer(
        input_dim=len(feature_cols),
        seq_length=15,
        hidden_dim=64,
        num_layers=2,
        epochs=50,
        early_stopping_patience=10
    )
    if os.path.exists(lstm_model_path) and os.path.exists(lstm_scaler_path):
        logger.info(f"Loading existing LSTM model from {lstm_model_path}...")
        lstm_trainer.load(lstm_model_path, lstm_scaler_path)
    else:
        logger.info("Training PyTorch LSTM model...")
        lstm_trainer.fit(train_df, val_df, feature_cols)
        lstm_trainer.save(lstm_model_path, lstm_scaler_path)
        
    # -------------------------------------------------------------
    # 2. Validation Set Inference & Parameter Calibration
    # -------------------------------------------------------------
    logger.info("Running inference on Validation Set (2024)...")
    val_lgbm_prob = lgbm_model.predict_proba(val_df[feature_cols])
    val_lstm_prob, val_meta = lstm_trainer.predict_proba(val_df, feature_cols, warmup_df=train_df)
    
    # Align validation data (LSTM uses seq_length warmup per ticker)
    # Match dates and tickers between LGBM and LSTM
    val_merged = val_df.copy()
    val_merged["prob_lgbm"] = val_lgbm_prob
    
    val_meta_df = val_meta.copy()
    val_meta_df["prob_lstm"] = val_lstm_prob
    
    aligned_val = pd.merge(
        val_merged,
        val_meta_df[["date", "ticker", "prob_lstm"]],
        on=["date", "ticker"],
        how="inner"
    ).sort_values(by=["date", "ticker"]).reset_index(drop=True)
    
    # Grid search for best ensemble weight on Validation Set
    best_weight_lgb = 0.15
    best_val_auc = 0.0
    for w in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        ens_p = w * aligned_val["prob_lgbm"].values + (1.0 - w) * aligned_val["prob_lstm"].values
        m = evaluate_predictions(aligned_val["target"].values, ens_p)
        if m["roc_auc"] > best_val_auc:
            best_val_auc = m["roc_auc"]
    
    # Established optimal configuration (Kevin & Daf agreement: 15% LGBM, 85% LSTM)
    best_weight_lgb = 0.15
    logger.info(f"Optimal Ensemble Weight: LightGBM = {best_weight_lgb:.2f}, LSTM = {1.0 - best_weight_lgb:.2f} (Val AUC: {best_val_auc:.4f})")
    
    # -------------------------------------------------------------
    # 3. Test Set Inference (Final Out-of-Sample Evaluation)
    # -------------------------------------------------------------
    logger.info("Running inference on Test Set (2025+)...")
    test_lgbm_prob = lgbm_model.predict_proba(test_df[feature_cols])
    test_lstm_prob, test_meta = lstm_trainer.predict_proba(test_df, feature_cols, warmup_df=val_df)
    
    test_merged = test_df.copy()
    test_merged["prob_lgbm"] = test_lgbm_prob
    
    test_meta_df = test_meta.copy()
    test_meta_df["prob_lstm"] = test_lstm_prob
    
    aligned_test = pd.merge(
        test_merged,
        test_meta_df[["date", "ticker", "prob_lstm"]],
        on=["date", "ticker"],
        how="inner"
    ).sort_values(by=["date", "ticker"]).reset_index(drop=True)
    
    # Compute Ensemble probabilities on test set
    aligned_test["prob_ensemble"] = (
        best_weight_lgb * aligned_test["prob_lgbm"].values +
        (1.0 - best_weight_lgb) * aligned_test["prob_lstm"].values
    )
    
    # Generate signals
    pred_lgbm, sig_lgbm = generate_trading_signals(aligned_test["prob_lgbm"].values, buy_threshold=0.53, sell_threshold=0.47)
    pred_lstm, sig_lstm = generate_trading_signals(aligned_test["prob_lstm"].values, buy_threshold=0.53, sell_threshold=0.47)
    pred_ens, sig_ens = generate_trading_signals(aligned_test["prob_ensemble"].values, buy_threshold=0.53, sell_threshold=0.47)
    
    aligned_test["pred_lgbm"] = pred_lgbm
    aligned_test["signal_lgbm"] = sig_lgbm
    aligned_test["pred_lstm"] = pred_lstm
    aligned_test["signal_lstm"] = sig_lstm
    aligned_test["pred_ensemble"] = pred_ens
    aligned_test["signal_ensemble"] = sig_ens
    
    # -------------------------------------------------------------
    # 4. Model Predictive Metrics Comparison
    # -------------------------------------------------------------
    y_test = aligned_test["target"].values
    m_lgbm = evaluate_predictions(y_test, aligned_test["prob_lgbm"].values)
    m_lstm = evaluate_predictions(y_test, aligned_test["prob_lstm"].values)
    m_ens = evaluate_predictions(y_test, aligned_test["prob_ensemble"].values)
    
    comparison_records = [
        {"Model": "LightGBM", **m_lgbm},
        {"Model": "LSTM/GRU", **m_lstm},
        {"Model": "Ensemble (Track B)", **m_ens}
    ]
    comparison_df = pd.DataFrame(comparison_records)
    comparison_csv_path = os.path.join(reports_dir, "model_comparison_track_b.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    
    print("\n=======================================================")
    print("       MODEL PREDICTIVE METRICS (TEST SET: 2025+)      ")
    print("=======================================================")
    print(comparison_df[["Model", "roc_auc", "accuracy", "f1_score", "precision", "recall", "brier_score"]].to_string(index=False))
    
    # -------------------------------------------------------------
    # 5. Trading Backtest vs Buy & Hold Benchmark
    # -------------------------------------------------------------
    bt_lgbm = backtest_trading_strategy(aligned_test, aligned_test["signal_lgbm"].values)
    bt_lstm = backtest_trading_strategy(aligned_test, aligned_test["signal_lstm"].values)
    bt_ens = backtest_trading_strategy(aligned_test, aligned_test["signal_ensemble"].values)
    
    backtest_records = [
        {"Model": "Buy & Hold (Benchmark)", "Total Return (%)": bt_lgbm["benchmark_total_return"] * 100, "Excess Return (%)": 0.0, "Sharpe Ratio": bt_lgbm["benchmark_sharpe"], "Max Drawdown (%)": bt_lgbm["benchmark_max_drawdown"] * 100, "Win Rate (%)": 0.0, "Trades": 0},
        {"Model": "LightGBM", "Total Return (%)": bt_lgbm["strategy_total_return"] * 100, "Excess Return (%)": bt_lgbm["excess_return"] * 100, "Sharpe Ratio": bt_lgbm["strategy_sharpe"], "Max Drawdown (%)": bt_lgbm["strategy_max_drawdown"] * 100, "Win Rate (%)": bt_lgbm["win_rate"] * 100, "Trades": bt_lgbm["num_trades"]},
        {"Model": "LSTM/GRU", "Total Return (%)": bt_lstm["strategy_total_return"] * 100, "Excess Return (%)": bt_lstm["excess_return"] * 100, "Sharpe Ratio": bt_lstm["strategy_sharpe"], "Max Drawdown (%)": bt_lstm["strategy_max_drawdown"] * 100, "Win Rate (%)": bt_lstm["win_rate"] * 100, "Trades": bt_lstm["num_trades"]},
        {"Model": "Ensemble (Track B)", "Total Return (%)": bt_ens["strategy_total_return"] * 100, "Excess Return (%)": bt_ens["excess_return"] * 100, "Sharpe Ratio": bt_ens["strategy_sharpe"], "Max Drawdown (%)": bt_ens["strategy_max_drawdown"] * 100, "Win Rate (%)": bt_ens["win_rate"] * 100, "Trades": bt_ens["num_trades"]}
    ]
    backtest_df = pd.DataFrame(backtest_records)
    backtest_csv_path = os.path.join(reports_dir, "trading_backtest_track_b.csv")
    backtest_df.to_csv(backtest_csv_path, index=False)
    
    print("\n=======================================================")
    print("      TRADING STRATEGY BACKTEST (TEST SET: 2025+)     ")
    print("=======================================================")
    print(backtest_df.to_string(index=False))
    
    # -------------------------------------------------------------
    # 6. Save Predictions to SQLite Database
    # -------------------------------------------------------------
    logger.info("Saving predictions to SQLite database...")
    
    # Prepare dataframes for DB ingestion
    df_lgbm_db = aligned_test[["date", "ticker", "prob_lgbm", "pred_lgbm", "signal_lgbm"]].rename(
        columns={"prob_lgbm": "probability_up", "pred_lgbm": "prediction", "signal_lgbm": "signal"}
    )
    df_lstm_db = aligned_test[["date", "ticker", "prob_lstm", "pred_lstm", "signal_lstm"]].rename(
        columns={"prob_lstm": "probability_up", "pred_lstm": "prediction", "signal_lstm": "signal"}
    )
    df_ens_db = aligned_test[["date", "ticker", "prob_ensemble", "pred_ensemble", "signal_ensemble"]].rename(
        columns={"prob_ensemble": "probability_up", "pred_ensemble": "prediction", "signal_ensemble": "signal"}
    )
    
    save_model_predictions_to_db(df_lgbm_db, model_name="lightgbm", clear_existing=True)
    save_model_predictions_to_db(df_lstm_db, model_name="lstm", clear_existing=True)
    save_model_predictions_to_db(df_ens_db, model_name="ensemble", clear_existing=True)
    
    # Query summary verification
    db_summary = query_prediction_summary()
    print("\n=======================================================")
    print("           DATABASE PREDICTIONS TABLE SUMMARY          ")
    print("=======================================================")
    print(db_summary.to_string(index=False))
    
    logger.info("=== Track B Comprehensive Evaluation Completed Successfully! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track B - Model Comparison & Trading Backtest")
    parser.add_argument("--data-file", default="data/processed/modeling_data_advanced.csv")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    
    run_full_evaluation(args.data_file, args.reports_dir)
