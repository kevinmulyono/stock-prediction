import logging
import os
import sys
from datetime import datetime
import pandas as pd
from sqlalchemy import text

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.connection import engine, init_db, SessionLocal
from src.database.models import Stock, MarketData, Prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Stock Metadata Universe
STOCK_METADATA = [
    {
        "ticker": "BBCA.JK",
        "company_name": "Bank Central Asia Tbk PT",
        "sector": "Financials"
    },
    {
        "ticker": "BBRI.JK",
        "company_name": "Bank Rakyat Indonesia (Persero) Tbk PT",
        "sector": "Financials"
    },
    {
        "ticker": "BMRI.JK",
        "company_name": "Bank Mandiri (Persero) Tbk PT",
        "sector": "Financials"
    },
    {
        "ticker": "TLKM.JK",
        "company_name": "Telkom Indonesia (Persero) Tbk PT",
        "sector": "Telecommunication & Infrastructure"
    },
    {
        "ticker": "ASII.JK",
        "company_name": "Astra International Tbk PT",
        "sector": "Industrials & Conglomerate"
    }
]


def seed_stocks(session):
    """Seed stock metadata into the stocks table."""
    logger.info("Seeding stocks metadata...")
    for item in STOCK_METADATA:
        existing = session.query(Stock).filter(Stock.ticker == item["ticker"]).first()
        if not existing:
            stock = Stock(
                ticker=item["ticker"],
                company_name=item["company_name"],
                sector=item["sector"]
            )
            session.add(stock)
    session.commit()
    logger.info(f"Seeded {len(STOCK_METADATA)} stock records.")


def seed_market_data(session, cleaned_data_path="data/processed/combined_market_data.csv"):
    """Seed cleaned market data into the market_data table."""
    if not os.path.exists(cleaned_data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_data_path}")

    logger.info(f"Loading cleaned market data from {cleaned_data_path}...")
    df = pd.read_csv(cleaned_data_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Bulk insert or upsert
    total_records = len(df)
    logger.info(f"Inserting {total_records} rows into market_data table...")
    
    # Process in batches of 1000 for optimal memory & database performance
    batch_size = 1000
    for i in range(0, total_records, batch_size):
        batch = df.iloc[i:i + batch_size]
        records = []
        for _, row in batch.iterrows():
            record = {
                "date": row["date"],
                "ticker": row["ticker"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "adj_close": float(row["adj_close"]),
                "volume": int(row["volume"]),
                "adj_factor": float(row["adj_factor"]),
            }
            records.append(record)
        
        # Use bulk insert mappings
        session.bulk_insert_mappings(MarketData, records)
        session.commit()
        logger.info(f"Inserted batch {min(i + batch_size, total_records)} / {total_records}...")

    logger.info("All market data successfully seeded into database.")


def run_database_setup():
    """Initializes tables and seeds data."""
    logger.info("=== Starting Database Migration & Seeding Pipeline ===")
    init_db()
    session = SessionLocal()
    try:
        seed_stocks(session)
        
        # Check if market_data is already populated
        existing_count = session.query(MarketData).count()
        if existing_count == 0:
            seed_market_data(session)
        else:
            logger.info(f"Market data already contains {existing_count} records. Skipping market data seeding.")
            
        # Summary verification
        stocks_count = session.query(Stock).count()
        market_count = session.query(MarketData).count()
        pred_count = session.query(Prediction).count()
        
        print("\n=== Database Table Verification ===")
        print(f"- stocks: {stocks_count} records")
        print(f"- market_data: {market_count} records")
        print(f"- predictions: {pred_count} records (ready for Daf and Track A)")
        
    finally:
        session.close()


if __name__ == "__main__":
    run_database_setup()
