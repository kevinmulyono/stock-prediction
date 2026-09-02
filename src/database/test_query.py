import os
import sys
import pandas as pd
from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.database.connection import engine, SessionLocal
from src.database.models import Stock, MarketData, Prediction


def test_db_queries():
    session = SessionLocal()
    try:
        print("1. Querying Stocks Metadata:")
        stocks = session.query(Stock).all()
        for s in stocks:
            print(f"   - [{s.ticker}] {s.company_name} | Sector: {s.sector}")
            
        print("\n2. Querying Market Data Sample (BBCA.JK Recent 5 Days):")
        query = text("""
            SELECT date, ticker, open, high, low, close, volume 
            FROM market_data 
            WHERE ticker = 'BBCA.JK' 
            ORDER BY date DESC 
            LIMIT 5
        """)
        df = pd.read_sql(query, con=engine)
        print(df.to_string(index=False))
        
        print("\n3. Testing Predictions Table Insertion (Mock Test):")
        mock_pred = Prediction(
            date=pd.to_datetime("2026-09-01").date(),
            ticker="BBCA.JK",
            model="xgboost",
            probability_up=0.61250,
            prediction=1,
            signal="BUY"
        )
        session.add(mock_pred)
        session.commit()
        
        # Verify prediction record
        pred = session.query(Prediction).filter_by(ticker="BBCA.JK", model="xgboost").first()
        print(f"   Successfully inserted & queried prediction: {pred}")
        
        # Clean up mock test record
        session.delete(pred)
        session.commit()
        print("   Mock test record cleaned up successfully.")
        
    finally:
        session.close()


if __name__ == "__main__":
    test_db_queries()
