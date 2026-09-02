"""
Database Ingestion for Predictions (Track B - Daf).

Saves model probabilistic predictions and trading signals to the
SQLite database (data/stock_prediction.db) under the 'predictions' table.
"""

import argparse
import logging
import os
import sys
from typing import List, Optional
import numpy as np
import pandas as pd
from sqlalchemy import text

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.connection import SessionLocal
from src.database.models import Prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def save_model_predictions_to_db(
    df: pd.DataFrame,
    model_name: str,
    probability_col: str = "probability_up",
    prediction_col: str = "prediction",
    signal_col: str = "signal",
    clear_existing: bool = True
) -> int:
    """
    Inserts or bulk-replaces predictions for a specific model in the database.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'date', 'ticker', probability_col, prediction_col, signal_col.
    model_name : str
        Model identifier ('lightgbm', 'lstm', 'ensemble').
    clear_existing : bool
        If True, clears previous records for this model before inserting.
        
    Returns:
    --------
    Number of records successfully inserted.
    """
    session = SessionLocal()
    try:
        if clear_existing:
            deleted = session.query(Prediction).filter(Prediction.model == model_name).delete()
            session.commit()
            logger.info(f"Cleared {deleted} previous records for model '{model_name}'.")
            
        records = []
        for _, row in df.iterrows():
            rec_date = pd.to_datetime(row["date"]).date()
            prob = float(np.clip(row[probability_col], 0.0, 1.0))
            pred = int(row[prediction_col])
            sig = str(row[signal_col])
            
            record = Prediction(
                date=rec_date,
                ticker=str(row["ticker"]),
                model=model_name,
                probability_up=round(prob, 5),
                prediction=pred,
                signal=sig
            )
            records.append(record)
            
        session.bulk_save_objects(records)
        session.commit()
        logger.info(f"Successfully saved {len(records)} prediction records for model '{model_name}'.")
        return len(records)
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving predictions for model '{model_name}': {str(e)}")
        raise e
    finally:
        session.close()


def query_prediction_summary() -> pd.DataFrame:
    """
    Queries and prints summary statistics of the predictions table.
    """
    session = SessionLocal()
    try:
        query = text("""
            SELECT model, 
                   COUNT(*) as total_predictions,
                   MIN(date) as min_date,
                   MAX(date) as max_date,
                   AVG(probability_up) as avg_p_up,
                   SUM(CASE WHEN signal = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                   SUM(CASE WHEN signal = 'HOLD' THEN 1 ELSE 0 END) as hold_count,
                   SUM(CASE WHEN signal = 'SELL' THEN 1 ELSE 0 END) as sell_count
            FROM predictions
            GROUP BY model
        """)
        result = session.execute(query)
        df_summary = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df_summary
    finally:
        session.close()


if __name__ == "__main__":
    print("Database Prediction Ingestion Helper Loaded.")
    summary = query_prediction_summary()
    print("\nCurrent Database Predictions Summary:")
    print(summary.to_string(index=False) if not summary.empty else "No predictions in database yet.")
