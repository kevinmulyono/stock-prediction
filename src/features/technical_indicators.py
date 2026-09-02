"""
Technical Indicators & Advanced Feature Engineering (Track B - Daf).

This module computes backward-looking technical indicators strictly without
look-ahead bias (no future data leakage). All computations are executed per ticker.

Categories of features computed:
1. Momentum:
   - RSI (Relative Strength Index, 14d)
   - MACD (12, 26, 9) + MACD Signal + MACD Histogram
   - Stochastic Oscillator (%K, %D with 14-period window, 3-period smoothing)
2. Trend:
   - EMA 5, EMA 20, and EMA Crossover Spread: (EMA_5 - EMA_20) / EMA_20
   - Bollinger Bands (20d, 2 std): Upper, Middle, Lower, BandWidth, and %B
   - Price to Moving Average ratios (Close / SMA_20, Close / SMA_50)
3. Lag Returns:
   - Historical returns: 1-day, 2-day, 3-day, 5-day, 10-day
4. Volatility:
   - Rolling Standard Deviation of returns (10d, 20d)
   - Normalized Average True Range (NATR 14d)
   - High-Low Spread / Close
5. Volume:
   - Volume / 20-day Volume Moving Average
   - Volume Rate of Change (5d)
   - Volume-Price Interaction: return_1d * volume_ratio
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes Relative Strength Index (RSI) using exponential moving average (Wilder's method).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's smoothing / EMA
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes MACD line, Signal line, and MACD Histogram.
    """
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Computes Stochastic Oscillator %K and %D.
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    
    stoch_k = 100.0 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()
    return stoch_k, stoch_d


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Computes Bollinger Bands: Middle, Upper, Lower, BandWidth, and %B.
    """
    middle_band = series.rolling(window=period, min_periods=period).mean()
    rolling_std = series.rolling(window=period, min_periods=period).std()
    
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    bandwidth = (upper_band - lower_band) / (middle_band + 1e-10)
    pct_b = (series - lower_band) / (upper_band - lower_band + 1e-10)
    
    return middle_band, upper_band, lower_band, bandwidth, pct_b


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Computes Normalized Average True Range (NATR, as percentage of close).
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    natr = (atr / (close + 1e-10)) * 100.0
    return natr


def generate_single_stock_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all Track B technical features for a single stock's historical data.
    Ensures that strictly backward-looking data is used.
    All OHLC bars are properly adjusted using adj_factor to maintain price consistency.
    """
    df = group.copy()
    df = df.sort_values(by="date").reset_index(drop=True)
    
    # Calculate split/dividend-adjusted OHLC
    adj_factor = df["adj_factor"] if "adj_factor" in df.columns else (df["adj_close"] / df["close"])
    adj_open = df["open"] * adj_factor
    adj_high = df["high"] * adj_factor
    adj_low = df["low"] * adj_factor
    adj_close = df["adj_close"]
    volume = df["volume"]
    
    # -------------------------------------------------------------
    # 1. Momentum Features
    # -------------------------------------------------------------
    df["rsi_14"] = compute_rsi(adj_close, period=14)
    macd_line, macd_signal, macd_hist = compute_macd(adj_close, fast_period=12, slow_period=26, signal_period=9)
    # Normalize MACD by price to make it scale-invariant across tickers
    df["macd_norm"] = macd_line / (adj_close + 1e-10)
    df["macd_signal_norm"] = macd_signal / (adj_close + 1e-10)
    df["macd_hist_norm"] = macd_hist / (adj_close + 1e-10)
    
    stoch_k, stoch_d = compute_stochastic(adj_high, adj_low, adj_close, k_period=14, d_period=3)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d
    
    # -------------------------------------------------------------
    # 2. Trend Features
    # -------------------------------------------------------------
    ema_5 = adj_close.ewm(span=5, adjust=False).mean()
    ema_20 = adj_close.ewm(span=20, adjust=False).mean()
    ema_50 = adj_close.ewm(span=50, adjust=False).mean()
    
    df["ema_5_ratio"] = adj_close / (ema_5 + 1e-10) - 1.0
    df["ema_20_ratio"] = adj_close / (ema_20 + 1e-10) - 1.0
    df["ema_crossover_5_20"] = (ema_5 - ema_20) / (ema_20 + 1e-10)
    df["ema_crossover_20_50"] = (ema_20 - ema_50) / (ema_50 + 1e-10)
    
    _, _, _, bb_width, bb_pct = compute_bollinger_bands(adj_close, period=20, num_std=2.0)
    df["bb_bandwidth"] = bb_width
    df["bb_pct_b"] = bb_pct
    
    # -------------------------------------------------------------
    # 3. Historical / Lag Returns (Backward-looking only)
    # -------------------------------------------------------------
    # return_1d at row t = (P_t - P_{t-1}) / P_{t-1}
    df["return_1d"] = adj_close.pct_change(1)
    df["return_2d"] = adj_close.pct_change(2)
    df["return_3d"] = adj_close.pct_change(3)
    df["return_5d"] = adj_close.pct_change(5)
    df["return_10d"] = adj_close.pct_change(10)
    
    # Lagged daily returns (Return of yesterday, 2 days ago, etc.)
    df["return_lag_1"] = df["return_1d"].shift(1)
    df["return_lag_2"] = df["return_1d"].shift(2)
    df["return_lag_3"] = df["return_1d"].shift(3)
    df["return_lag_5"] = df["return_1d"].shift(5)
    
    # -------------------------------------------------------------
    # 4. Volatility Features
    # -------------------------------------------------------------
    df["volatility_10d"] = df["return_1d"].rolling(window=10, min_periods=10).std()
    df["volatility_20d"] = df["return_1d"].rolling(window=20, min_periods=20).std()
    df["natr_14"] = compute_atr(adj_high, adj_low, adj_close, period=14)
    df["hl_spread"] = (adj_high - adj_low) / (adj_close + 1e-10)
    df["co_spread"] = (adj_close - adj_open) / (adj_open + 1e-10)
    
    # -------------------------------------------------------------
    # 5. Volume Features
    # -------------------------------------------------------------
    vol_sma_20 = volume.rolling(window=20, min_periods=20).mean()
    df["volume_ratio_20d"] = volume / (vol_sma_20 + 1e-10)
    df["volume_roc_5d"] = volume.pct_change(5)
    df["price_volume_interaction"] = df["return_1d"] * df["volume_ratio_20d"]
    
    return df


def extract_features(df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """
    Takes a dataframe containing market data or baseline modeling data,
    computes all Track B technical indicators grouped by ticker,
    and optionally drops warmup rows with NaNs.
    """
    logger.info("Computing advanced technical features per ticker...")
    featured_groups = []
    
    # Group by ticker to ensure strictly isolated calculations per stock
    for ticker, group in df.groupby("ticker"):
        logger.info(f"Processing technical indicators for ticker: {ticker} ({len(group)} rows)...")
        fg = generate_single_stock_features(group)
        featured_groups.append(fg)
        
    full_df = pd.concat(featured_groups, ignore_index=True)
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = full_df.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    
    # Get list of feature columns
    feature_cols = get_feature_columns()
    
    if drop_warmup:
        initial_len = len(full_df)
        full_df = full_df.dropna(subset=feature_cols).reset_index(drop=True)
        dropped_rows = initial_len - len(full_df)
        logger.info(f"Dropped {dropped_rows} warmup rows with NaNs across all indicators.")
        
    logger.info(f"Generated {len(feature_cols)} technical features across {len(full_df)} rows.")
    return full_df


def get_feature_columns() -> List[str]:
    """
    Returns the standardized list of Track B feature column names.
    Note: 'forward_return_1d' and 'target' are strictly excluded from feature columns.
    """
    return [
        # Momentum
        "rsi_14",
        "macd_norm",
        "macd_signal_norm",
        "macd_hist_norm",
        "stoch_k",
        "stoch_d",
        # Trend
        "ema_5_ratio",
        "ema_20_ratio",
        "ema_crossover_5_20",
        "ema_crossover_20_50",
        "bb_bandwidth",
        "bb_pct_b",
        # Lag Returns
        "return_1d",
        "return_2d",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_lag_1",
        "return_lag_2",
        "return_lag_3",
        "return_lag_5",
        # Volatility
        "volatility_10d",
        "volatility_20d",
        "natr_14",
        "hl_spread",
        "co_spread",
        # Volume
        "volume_ratio_20d",
        "volume_roc_5d",
        "price_volume_interaction"
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track B - Technical Feature Engineering")
    parser.add_argument("--input-file", default="data/processed/modeling_data_baseline.csv", help="Input baseline data")
    parser.add_argument("--output-file", default="data/processed/modeling_data_advanced.csv", help="Output feature engineered dataset")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        logger.error(f"Input file {args.input_file} not found. Please run preparation.py first.")
        sys.exit(1)
        
    df_raw = pd.read_csv(args.input_file)
    featured_df = extract_features(df_raw, drop_warmup=True)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    featured_df.to_csv(args.output_file, index=False)
    logger.info(f"Advanced featured dataset saved successfully to {args.output_file} ({len(featured_df)} rows).")
    
    feature_cols = get_feature_columns()
    print(f"\n--- Feature Engineering Summary ---")
    print(f"Total Features: {len(feature_cols)}")
    print(f"Features list: {', '.join(feature_cols)}")
    print(f"Sample data head:")
    print(featured_df[["ticker", "date"] + feature_cols[:6] + ["target"]].head(3).to_string())
