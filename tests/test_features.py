"""
Unit Tests for Technical Indicator Feature Engineering (Track B - Daf).

Guarantees:
1. Strictly backward-looking property (zero lookahead bias / no future leakage).
2. Absence of NaNs after standard 50-day warmup window.
3. Feature schema integrity: exactly 29 technical indicators.
4. Absence of target / forward return in feature columns.
5. Cross-ticker calculation isolation (no cross-contamination between stocks).
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.technical_indicators import (
    generate_single_stock_features,
    extract_features,
    get_feature_columns,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_stochastic
)


@pytest.fixture
def sample_market_data():
    """Generates synthetic multi-day OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    
    # Random walk prices
    returns = np.random.normal(0.001, 0.02, n)
    close = 5000.0 * np.cumprod(1.0 + returns)
    high = close * (1.0 + np.abs(np.random.normal(0.005, 0.005, n)))
    low = close * (1.0 - np.abs(np.random.normal(0.005, 0.005, n)))
    open_p = (high + low) / 2.0
    volume = np.random.randint(1_000_000, 50_000_000, n)
    
    df = pd.DataFrame({
        "date": dates,
        "ticker": "BBCA.JK",
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": volume,
        "adj_factor": 1.0
    })
    return df


def test_feature_column_schema():
    """Verify that exactly 29 technical indicators are defined and no target leakage columns exist."""
    cols = get_feature_columns()
    assert len(cols) == 29, f"Expected 29 feature columns, got {len(cols)}"
    
    # Critical Anti-Leakage Check
    assert "target" not in cols, "CRITICAL LEAKAGE: 'target' must not be in feature columns!"
    assert "forward_return_1d" not in cols, "CRITICAL LEAKAGE: 'forward_return_1d' must not be in feature columns!"
    assert "date" not in cols
    assert "ticker" not in cols


def test_no_rogue_nans_after_warmup(sample_market_data):
    """Verify that no NaN values exist in any of the 29 features after 50-day warmup."""
    featured = generate_single_stock_features(sample_market_data)
    feature_cols = get_feature_columns()
    
    # Rows after 50 days (index 50 onwards)
    post_warmup = featured.iloc[50:]
    assert len(post_warmup) == 50
    
    nan_counts = post_warmup[feature_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    assert len(nan_cols) == 0, f"Rogue NaNs detected post-warmup in columns: {nan_cols.to_dict()}"


def test_strictly_backward_looking_anti_leakage(sample_market_data):
    """
    CRITICAL ANTI-LEAKAGE TEST:
    Modifying future prices (at t+1, t+2, ... t+k) must have ZERO effect
    on any feature value calculated at time t.
    """
    feature_cols = get_feature_columns()
    
    # 1. Compute features on original data up to day 60
    t_idx = 60
    featured_original = generate_single_stock_features(sample_market_data)
    features_at_t_orig = featured_original.iloc[t_idx][feature_cols].values.astype(float)
    
    # 2. Perturb future data (days 61 to 100) drastically (e.g. 10x prices, negative shocks)
    perturbed_data = sample_market_data.copy()
    future_mask = perturbed_data.index > t_idx
    perturbed_data.loc[future_mask, "close"] *= 5.0
    perturbed_data.loc[future_mask, "adj_close"] *= 5.0
    perturbed_data.loc[future_mask, "high"] *= 10.0
    perturbed_data.loc[future_mask, "low"] *= 0.1
    perturbed_data.loc[future_mask, "volume"] *= 100
    
    featured_perturbed = generate_single_stock_features(perturbed_data)
    features_at_t_perturbed = featured_perturbed.iloc[t_idx][feature_cols].values.astype(float)
    
    # 3. Assert exact equality at time t
    diff = np.abs(features_at_t_orig - features_at_t_perturbed)
    max_diff = np.max(diff)
    assert max_diff < 1e-10, (
        f"DATA LEAKAGE DETECTED! Max feature difference at time t is {max_diff} "
        f"after modifying future data!"
    )


def test_standalone_indicator_math():
    """Verify mathematical range and constraints of core indicators."""
    prices = pd.Series([100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0, 110.0, 112.0, 111.0, 113.0, 115.0, 117.0, 116.0, 118.0, 120.0])
    rsi = compute_rsi(prices, period=14)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all(), "RSI must be bounded in [0, 100]"
    
    high = prices + 1.0
    low = prices - 1.0
    k, d = compute_stochastic(high, low, prices, k_period=5, d_period=3)
    valid_k = k.dropna()
    assert (valid_k >= 0.0).all() and (valid_k <= 100.0).all(), "Stochastic %K must be bounded in [0, 100]"


def test_cross_ticker_isolation():
    """Verify that feature calculation for one ticker is never influenced by another ticker."""
    np.random.seed(99)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    
    df1 = pd.DataFrame({
        "date": dates,
        "ticker": "BBCA.JK",
        "open": 9000.0,
        "high": 9100.0,
        "low": 8900.0,
        "close": 9050.0,
        "adj_close": 9050.0,
        "volume": 10_000_000,
        "adj_factor": 1.0
    })
    
    df2 = pd.DataFrame({
        "date": dates,
        "ticker": "TLKM.JK",
        "open": 3000.0,
        "high": 3100.0,
        "low": 2900.0,
        "close": 3050.0,
        "adj_close": 3050.0,
        "volume": 50_000_000,
        "adj_factor": 1.0
    })
    
    feat_bbca_solo = generate_single_stock_features(df1)
    combined = pd.concat([df1, df2], ignore_index=True)
    feat_combined = extract_features(combined, drop_warmup=False)
    
    feat_bbca_combined = feat_combined[feat_combined["ticker"] == "BBCA.JK"].reset_index(drop=True)
    
    feature_cols = get_feature_columns()
    diff = np.nan_to_num(feat_bbca_solo[feature_cols].values) - np.nan_to_num(feat_bbca_combined[feature_cols].values)
    assert np.max(np.abs(diff)) < 1e-10, "Cross-ticker leakage detected when combining dataframes!"
