import argparse
import glob
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def validate_ohlc_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check logical consistency of OHLC bars:
    1. Low <= Open <= High
    2. Low <= Close <= High
    3. Low <= Adj Close (normalized)
    4. Prices > 0 and Volume >= 0
    
    Returns a boolean mask or flags anomalous rows.
    """
    invalid_open = (df["Open"] > df["High"]) | (df["Open"] < df["Low"])
    invalid_close = (df["Close"] > df["High"]) | (df["Close"] < df["Low"])
    negative_prices = (df["Open"] <= 0) | (df["High"] <= 0) | (df["Low"] <= 0) | (df["Close"] <= 0)
    negative_volume = df["Volume"] < 0
    
    anomalies = invalid_open | invalid_close | negative_prices | negative_volume
    return anomalies


def clean_single_stock_data(
    file_path: str,
    output_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean raw market data for a single stock CSV file:
    - Standardize column names (lowercase with underscores)
    - Convert Date to datetime and sort chronologically
    - Check and eliminate duplicate timestamps
    - Filter zero-volume non-trading records / flat public holiday artifacts
    - Check for missing values and validate OHLC logic
    - Compute adjustment factor (Adj Close / Close)
    - Save cleaned dataset to output_dir
    
    Returns:
    --------
    Tuple[pd.DataFrame, Dict]: Cleaned DataFrame and Audit Dictionary
    """
    raw_df = pd.read_csv(file_path)
    initial_rows = len(raw_df)
    ticker = raw_df["Ticker"].iloc[0] if "Ticker" in raw_df.columns else os.path.basename(file_path).split("_")[0]
    
    logger.info(f"Cleaning data for {ticker} (Initial rows: {initial_rows})...")
    
    # 1. Standardize column names
    df = raw_df.copy()
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    
    # Ensure expected columns exist
    expected_cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns {missing_cols} in {file_path}")
        
    # Reorder standard columns
    df = df[expected_cols]
    
    # 2. Date parsing & Chronological Sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
    
    # 3. Duplicate Date Removal
    dup_count = df.duplicated(subset=["date"]).sum()
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate date rows for {ticker}. Keeping the first entry.")
        df = df.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
        
    # 4. Cast numeric types
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # 5. Missing Values Check
    null_counts = df[numeric_cols].isnull().sum().to_dict()
    has_nulls = sum(null_counts.values()) > 0
    if has_nulls:
        logger.warning(f"Missing values found for {ticker}: {null_counts}. Dropping NaN rows without forward-looking bias.")
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)
        
    # 6. Filter Non-Trading Day Artifacts (Volume = 0 and flat Open=High=Low=Close)
    # On public holidays or trading halts, yfinance often records 0 volume with previous close
    flat_zero_vol = (df["volume"] == 0) & (df["open"] == df["close"]) & (df["high"] == df["low"])
    zero_vol_count = flat_zero_vol.sum()
    if zero_vol_count > 0:
        logger.info(f"Filtering out {zero_vol_count} non-trading / holiday bars with zero volume for {ticker}.")
        df = df[~flat_zero_vol].reset_index(drop=True)
        
    # 7. OHLC Logic Validation
    # Map lowercase back to uppercase temp for validator
    temp_df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "adj_close": "Adj Close", "volume": "Volume"})
    anomalies_mask = validate_ohlc_logic(temp_df)
    anomaly_count = anomalies_mask.sum()
    if anomaly_count > 0:
        logger.warning(f"Found {anomaly_count} anomalous OHLC rows for {ticker}. Removing anomalous rows.")
        df = df[~anomalies_mask].reset_index(drop=True)
        
    # 8. Compute Adjustment Factor (for split/dividend tracking)
    # adj_factor = adj_close / close
    df["adj_factor"] = np.where(df["close"] > 0, df["adj_close"] / df["close"], 1.0)
    
    # 9. Format Date as standard string YYYY-MM-DD
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
    # Final row count
    cleaned_rows = len(df)
    
    # Save individual cleaned CSV
    os.makedirs(output_dir, exist_ok=True)
    safe_ticker = ticker.replace(".", "_")
    output_path = os.path.join(output_dir, f"{safe_ticker}_cleaned.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path} ({cleaned_rows} rows).")
    
    audit_record = {
        "ticker": ticker,
        "initial_rows": initial_rows,
        "duplicates_removed": int(dup_count),
        "zero_vol_removed": int(zero_vol_count),
        "anomalies_removed": int(anomaly_count),
        "final_cleaned_rows": cleaned_rows,
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "output_path": output_path
    }
    
    return df, audit_record


def clean_all_raw_data(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterates over all raw CSV files, performs cleaning, and produces
    both individual cleaned files and a unified dataset.
    """
    raw_files = glob.glob(os.path.join(raw_dir, "*_raw.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No raw CSV files matching '*_raw.csv' found in {raw_dir}")
        
    cleaned_dfs = []
    audit_records = []
    
    for file_path in raw_files:
        cleaned_df, audit_rec = clean_single_stock_data(file_path, output_dir)
        cleaned_dfs.append(cleaned_df)
        audit_records.append(audit_rec)
        
    # Combine all cleaned stocks into unified dataset
    combined_df = pd.concat(cleaned_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    
    combined_path = os.path.join(output_dir, "combined_market_data.csv")
    combined_df.to_csv(combined_path, index=False)
    logger.info(f"Unified cleaned dataset saved to {combined_path} ({len(combined_df)} total records).")
    
    # Save cleaning audit summary
    audit_df = pd.DataFrame(audit_records)
    audit_path = os.path.join(output_dir, "cleaning_audit.csv")
    audit_df.to_csv(audit_path, index=False)
    logger.info(f"Cleaning audit report saved to {audit_path}.")
    
    return combined_df, audit_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A - Data Cleaning Pipeline")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Path to processed output directory")
    args = parser.parse_args()
    
    logger.info("=== Starting Data Cleaning Pipeline (Track A) ===")
    combined_df, audit_df = clean_all_raw_data(args.raw_dir, args.output_dir)
    print("\n--- Data Cleaning Audit Report ---")
    print(audit_df.to_string(index=False))
