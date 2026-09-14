"""
Unit Tests for Model Inference & Trading Signals (Track B - Daf).

Guarantees:
1. predict_proba() output is strictly bounded within [0.0, 1.0].
2. Discrete signal classification logic strictly matches:
   - P(Up) >= 0.53 -> BUY
   - P(Up) <= 0.47 -> SELL
   - 0.47 < P(Up) < 0.53 -> HOLD
3. Prediction serialization & artifact loading without re-training.
4. Multi-ticker backtesting position independence (no position bleeding).
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ensemble_and_signals import (
    generate_trading_signals,
    compute_ensemble_probabilities,
    backtest_trading_strategy
)
from src.models.predict_daily import classify_signal, DailyPredictor
from src.models.lightgbm_model import LightGBMStockModel
from src.models.lstm_model import LSTMStockTrainer, StockLSTM
from src.features.technical_indicators import get_feature_columns


def test_signal_classification_mapping():
    """Verify exact signal threshold rules: BUY >= 0.53, SELL <= 0.47, HOLD in between."""
    # Test cases: (prob, expected_pred, expected_signal)
    test_cases = [
        (0.60, 1, "BUY"),
        (0.53, 1, "BUY"),        # Boundary BUY
        (0.5299, 1, "HOLD"),     # Just below BUY threshold
        (0.50, 1, "HOLD"),       # Neutral midpoint, binary pred = 1
        (0.4999, 0, "HOLD"),     # Neutral midpoint, binary pred = 0
        (0.4701, 0, "HOLD"),     # Just above SELL threshold
        (0.47, 0, "SELL"),       # Boundary SELL
        (0.40, 0, "SELL"),
        (0.00, 0, "SELL"),
        (1.00, 1, "BUY"),
    ]
    
    for p, exp_pred, exp_sig in test_cases:
        pred, sig, advice = classify_signal(p, buy_th=0.53, sell_th=0.47)
        assert pred == exp_pred, f"P={p}: expected binary pred {exp_pred}, got {pred}"
        assert sig == exp_sig, f"P={p}: expected signal {exp_sig}, got {sig}"
        assert isinstance(advice, str) and len(advice) > 0


def test_generate_trading_signals_vectorized():
    """Verify batch vector signal generation matches scalar rules."""
    probs = np.array([0.70, 0.53, 0.51, 0.48, 0.47, 0.30])
    preds, signals = generate_trading_signals(probs, buy_threshold=0.53, sell_threshold=0.47)
    
    expected_preds = np.array([1, 1, 1, 0, 0, 0])
    expected_sigs = np.array(["BUY", "BUY", "HOLD", "HOLD", "SELL", "SELL"])
    
    np.testing.assert_array_equal(preds, expected_preds)
    np.testing.assert_array_equal(signals, expected_sigs)


def test_ensemble_probability_bounds():
    """Verify that ensemble probabilities are strictly float in [0.0, 1.0]."""
    p_lgb = np.array([0.1, 0.5, 0.9, -0.2, 1.5])  # with extreme bounds
    p_lstm = np.array([0.2, 0.6, 0.8, 0.0, 1.0])
    
    prob_dict = {"lightgbm": p_lgb, "lstm": p_lstm}
    weights = {"lightgbm": 0.15, "lstm": 0.85}
    
    ens_p = compute_ensemble_probabilities(prob_dict, weights)
    
    assert isinstance(ens_p, np.ndarray)
    assert np.all(ens_p >= 0.0), "Ensemble probabilities must be >= 0.0"
    assert np.all(ens_p <= 1.0), "Ensemble probabilities must be <= 1.0"


def test_model_artifacts_load_and_predict_bounds():
    """Verify pre-trained model artifacts load cleanly and output valid probabilities in [0.0, 1.0]."""
    lgb_path = "reports/models/lightgbm_model.joblib"
    lstm_path = "reports/models/lstm_model.pt"
    scaler_path = "reports/models/lstm_scaler.joblib"
    
    assert os.path.exists(lgb_path), f"Missing artifact: {lgb_path}"
    assert os.path.exists(lstm_path), f"Missing artifact: {lstm_path}"
    assert os.path.exists(scaler_path), f"Missing artifact: {scaler_path}"
    
    feature_cols = get_feature_columns()
    n_features = len(feature_cols)
    
    # 1. Test LightGBM
    lgb_model = LightGBMStockModel.load(lgb_path)
    synthetic_input = pd.DataFrame(np.random.normal(0, 1, (5, n_features)), columns=feature_cols)
    lgb_probs = lgb_model.predict_proba(synthetic_input)
    assert len(lgb_probs) == 5
    assert np.all(lgb_probs >= 0.0) and np.all(lgb_probs <= 1.0)
    
    # 2. Test LSTM
    predictor = DailyPredictor()
    predictor.load_lstm()
    
    dummy_seq = np.random.normal(0, 1, (1, 15, n_features)).astype(np.float32)
    seq_tensor = torch.tensor(dummy_seq, dtype=torch.float32)
    with torch.no_grad():
        lstm_prob = float(predictor.lstm_model(seq_tensor).numpy()[0])
    assert 0.0 <= lstm_prob <= 1.0, f"LSTM prob {lstm_prob} out of [0, 1] bounds!"


def test_backtesting_multi_ticker_isolation():
    """
    CRITICAL BUG FIX VERIFICATION:
    Verify that backtest_trading_strategy isolates positions per ticker.
    A BUY on Ticker A followed by rows for Ticker B must NOT bleed Long position into Ticker B.
    """
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    
    # Ticker A: BUY on day 1 -> Long
    df_a = pd.DataFrame({
        "date": dates,
        "ticker": "STOCK_A",
        "forward_return_1d": [0.01, 0.02, -0.01],
        "signal": ["BUY", "HOLD", "SELL"]
    })
    
    # Ticker B: SELL on all days -> Flat / Cash (position must stay 0)
    df_b = pd.DataFrame({
        "date": dates,
        "ticker": "STOCK_B",
        "forward_return_1d": [-0.05, -0.05, -0.05],
        "signal": ["SELL", "SELL", "SELL"]
    })
    
    combined = pd.concat([df_a, df_b], ignore_index=True)
    res = backtest_trading_strategy(combined, combined["signal"].values)
    
    # Stock B had massive drops (-5% each day), but since its signal was SELL,
    # its position should be 0 and not lose money.
    # Stock A had positive returns (+1%, +2%).
    # The combined portfolio return must be positive!
    assert res["strategy_total_return"] > 0.0, (
        f"Position bleeding detected! Strategy return is {res['strategy_total_return']} "
        f"despite Stock A having positive returns and Stock B being flat."
    )
