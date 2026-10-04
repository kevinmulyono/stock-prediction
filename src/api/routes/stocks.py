"""
Routes for Stocks & Market Data Endpoints (Track C - Ali).
"""

from datetime import date, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, desc

from src.database.connection import get_db_session
from src.database.models import Stock, MarketData
from src.api.schemas import (
    StockItem,
    StocksListResponse,
    MarketDataPoint,
    MarketDataHistoryResponse
)

router = APIRouter(prefix="/api/stocks", tags=["Stocks & Market Data"])


@router.get("", response_model=StocksListResponse)
def get_all_stocks():
    """
    Returns list of 5 IDX target bluechip stocks with their latest available
    close price and 1-day return change percentage.
    """
    with get_db_session() as session:
        stocks = session.query(Stock).order_by(Stock.ticker).all()
        stock_items = []

        for stock in stocks:
            # Query last 2 market data entries to compute daily change
            latest_rows = (
                session.query(MarketData)
                .filter(MarketData.ticker == stock.ticker)
                .order_by(desc(MarketData.date))
                .limit(2)
                .all()
            )

            latest_close = None
            latest_date_str = None
            change_pct = 0.0

            if latest_rows:
                latest_close = float(latest_rows[0].close)
                latest_date_str = str(latest_rows[0].date)
                if len(latest_rows) > 1 and float(latest_rows[1].close) > 0:
                    prev_close = float(latest_rows[1].close)
                    change_pct = round(((latest_close - prev_close) / prev_close) * 100, 2)

            stock_items.append(
                StockItem(
                    ticker=stock.ticker,
                    company_name=stock.company_name,
                    sector=stock.sector,
                    latest_date=latest_date_str,
                    latest_close=latest_close,
                    change_pct=change_pct
                )
            )

        return StocksListResponse(total=len(stock_items), data=stock_items)


@router.get("/{ticker}/history", response_model=MarketDataHistoryResponse)
def get_stock_history(
    ticker: str,
    timeframe: str = Query("1Y", pattern="^(1M|3M|6M|1Y|ALL)$", description="Filter range: 1M, 3M, 6M, 1Y, ALL"),
    start_date: Optional[str] = Query(None, description="Start date format YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date format YYYY-MM-DD")
):
    """
    Returns historical OHLCV data for the specified ticker, optimized for Candlestick and Volume charts.
    """
    normalized_ticker = ticker.upper()
    if not normalized_ticker.endswith(".JK"):
        normalized_ticker = f"{normalized_ticker}.JK"

    with get_db_session() as session:
        # Check if stock exists
        stock_exists = session.query(Stock).filter(Stock.ticker == normalized_ticker).first()
        if not stock_exists:
            raise HTTPException(status_code=404, detail=f"Stock ticker '{normalized_ticker}' not found in database.")

        query = session.query(MarketData).filter(MarketData.ticker == normalized_ticker)

        if start_date:
            query = query.filter(MarketData.date >= start_date)
        if end_date:
            query = query.filter(MarketData.date <= end_date)

        # Apply timeframe filter if start_date wasn't manually specified
        if not start_date and timeframe != "ALL":
            # Get latest available date for this ticker first
            max_date = session.query(func.max(MarketData.date)).filter(MarketData.ticker == normalized_ticker).scalar()
            if max_date:
                days_map = {
                    "1M": 30,
                    "3M": 90,
                    "6M": 180,
                    "1Y": 365
                }
                delta_days = days_map.get(timeframe, 365)
                cutoff_date = max_date - timedelta(days=delta_days)
                query = query.filter(MarketData.date >= cutoff_date)

        records = query.order_by(MarketData.date.asc()).all()

        points = [
            MarketDataPoint(
                date=str(row.date),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=int(row.volume),
                adj_close=float(row.adj_close) if row.adj_close is not None else None
            )
            for row in records
        ]

        return MarketDataHistoryResponse(
            ticker=normalized_ticker,
            timeframe=timeframe,
            count=len(points),
            data=points
        )
