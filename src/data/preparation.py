import argparse
import logging
import os
import sys
from typing import Tuple
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def create_target_variable(df: pd.DataFrame, is_live: bool = False) -> pd.DataFrame:
    """
    Creates the Target Variable (Next-Day Price Direction).
    
    Formula:
    R_{t+1} = (AdjClose_{t+1} - AdjClose_t) / AdjClose_t
    Target = 1 if R_{t+1} > 0 else 0
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned market data (must contain 'ticker', 'date', 'adj_close')
    is_live : bool
        If True, keeps the last row (where target is NaN) for inference.
        If False, drops rows with NaN targets for clean model training.
        
    Returns:
    --------
    pd.DataFrame with 'target' column appended.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    
    # Calculate Forward Return R_{t+1}
    df["forward_return_1d"] = df.groupby("ticker")["adj_close"].shift(-1) / df["adj_close"] - 1
    
    # Create Binary Target: 1 (Up) or 0 (Down/Flat)
    df["target"] = (df["forward_return_1d"] > 0).astype(float)
    
    # Restore NaN for the last row since shift(-1) inherently produces a NaN
    df.loc[df["forward_return_1d"].isna(), "target"] = pd.NA
    
    # Handle the NaN records
    nan_count = df["target"].isna().sum()
    if not is_live:
        logger.info(f"Dropping {nan_count} rows with missing targets (safe for training).")
        df = df.dropna(subset=["target"]).reset_index(drop=True)
        # Convert target to int safely
        df["target"] = df["target"].astype(int)
    else:
        logger.info(f"Retaining {nan_count} rows with missing targets for Live Inference mode.")
        
    return df


def split_time_series(
    df: pd.DataFrame, 
    train_start: str = "2018-01-01", 
    train_end: str = "2023-12-31", 
    val_start: str = "2024-01-01",
    val_end: str = "2024-12-31",
    test_start: str = "2025-01-01"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a strict chronological time-series split.
    
    Returns:
    --------
    Tuple of DataFrames: (train_df, val_df, test_df)
    """
    df["date"] = pd.to_datetime(df["date"])
    
    train_mask = (df["date"] >= train_start) & (df["date"] <= train_end)
    val_mask = (df["date"] >= val_start) & (df["date"] <= val_end)
    test_mask = (df["date"] >= test_start)
    
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)
    
    logger.info(f"Time-Series Split Results:")
    logger.info(f"  Train : {len(train_df)} rows ({train_start} to {train_end})")
    logger.info(f"  Val   : {len(val_df)} rows ({val_start} to {val_end})")
    logger.info(f"  Test  : {len(test_df)} rows ({test_start} onwards)")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A - Data Preparation Pipeline (Phase 5 & 6)")
    parser.add_argument("--input-file", default="data/processed/combined_market_data.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    
    logger.info("=== Starting Data Preparation Pipeline (Phase 5 & 6) ===")
    
    if not os.path.exists(args.input_file):
        logger.error(f"Input file {args.input_file} not found!")
        sys.exit(1)
        
    raw_df = pd.read_csv(args.input_file)
    
    # Phase 5: Create Target Variable
    df_with_target = create_target_variable(raw_df, is_live=False)
    
    # Save baseline modeling data for Track A
    output_path = os.path.join(args.output_dir, "modeling_data_baseline.csv")
    df_with_target.to_csv(output_path, index=False)
    logger.info(f"Saved modeling baseline data to {output_path}")
    
    # Phase 6: Verify Time-Series Split
    train_df, val_df, test_df = split_time_series(df_with_target)
    
    # Audit class balances
    print("\n--- Target Class Balance (Train Set) ---")
    print(train_df["target"].value_counts(normalize=True).mul(100).round(2).to_string() + " %")
