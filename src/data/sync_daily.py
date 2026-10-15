"""
Real-Time Market Data Synchronization Module (Track C - Ali).

Safely fetches missing daily market candles from Yahoo Finance (yfinance)
and upserts them into the SQLite 'market_data' table without modifying
trained model artifacts or historical backtesting data.
"""

from datetime import date, datetime, timedelta
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import func

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database.connection import get_db_session
from src.database.models import MarketData, Stock

logger = logging.getLogger(__name__)

TARGET_TICKERS = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]


def sync_single_ticker_data(ticker: str, session) -> int:
    """
    Checks the latest date in DB for a single ticker, fetches missing days
    up to today from Yahoo Finance, and persists new rows into market_data.
    Returns the count of newly inserted rows.
    """
    max_date = session.query(func.max(MarketData.date)).filter(MarketData.ticker == ticker).scalar()
    today = datetime.today().date()

    if max_date is not None and max_date >= today:
        logger.info(f"[{ticker}] Market data is already up to date ({max_date}).")
        return 0

    # Start date is the day after max_date
    if max_date is not None:
        start_date = max_date + timedelta(days=1)
    else:
        start_date = date(2018, 1, 1)

    # End date in yfinance is exclusive, so add 1 day to today
    end_date = today + timedelta(days=1)

    if start_date >= end_date:
        return 0

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"[{ticker}] Fetching missing candles from {start_str} to {end_str} via yfinance...")

    try:
        df = yf.download(
            tickers=ticker,
            start=start_str,
            end=end_str,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if df is None or df.empty:
            logger.info(f"[{ticker}] No new candles returned (market may be closed / holiday).")
            return 0

        # Handle yfinance MultiIndex column headers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Normalize column names to lowercase
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

        if "date" not in df.columns:
            logger.warning(f"[{ticker}] 'date' column missing in yfinance response.")
            return 0

        # Convert date to standard python date
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Filter strictly for dates greater than max_date (prevents duplicate key errors)
        if max_date is not None:
            df = df[df["date"] > max_date]

        if df.empty:
            return 0

        # Basic data validation and cleaning
        # Ensure prices > 0 and volume >= 0
        valid_mask = (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
        df = df[valid_mask].copy()

        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        # Fill any missing volume with 0
        df["volume"] = df["volume"].fillna(0).astype(int)

        # Compute adj_factor (Adj Close / Close)
        df["adj_factor"] = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], 1.0).fillna(1.0).round(6)

        inserted = 0
        for _, row in df.iterrows():
            m_obj = MarketData(
                date=row["date"],
                ticker=ticker,
                open=round(float(row["open"]), 4),
                high=round(float(row["high"]), 4),
                low=round(float(row["low"]), 4),
                close=round(float(row["close"]), 4),
                adj_close=round(float(row["adj_close"]), 4),
                volume=int(row["volume"]),
                adj_factor=round(float(row["adj_factor"]), 6)
            )
            session.add(m_obj)
            inserted += 1

        session.commit()
        logger.info(f"[{ticker}] Successfully inserted {inserted} new live candles up to {df['date'].iloc[-1]}.")
        return inserted

    except Exception as e:
        session.rollback()
        logger.error(f"[{ticker}] Failed to sync live market data: {e}")
        return 0


def sync_market_data(tickers: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Synchronizes real-time market data for the target IDX bluechip tickers.
    Returns a dictionary of ticker -> inserted rows count.
    """
    tickers = tickers or TARGET_TICKERS
    summary = {}

    with get_db_session() as session:
        for ticker in tickers:
            count = sync_single_ticker_data(ticker, session)
            summary[ticker] = count

    total_inserted = sum(summary.values())
    logger.info(f"=== Live Market Data Sync Complete: {total_inserted} total candles inserted ===")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sync_market_data()
