from datetime import datetime, date
from typing import Optional
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
    Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Stock(Base):
    __tablename__ = "stocks"

    stock_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    company_name = Column(String(255), nullable=False)
    sector = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    market_data = relationship("MarketData", back_populates="stock", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="stock", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Stock(ticker='{self.ticker}', company='{self.company_name}')>"


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer().with_variant(BigInteger, "postgresql"), primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    ticker = Column(String(20), ForeignKey("stocks.ticker", ondelete="CASCADE"), nullable=False, index=True)
    open = Column(Numeric(14, 4), nullable=False)
    high = Column(Numeric(14, 4), nullable=False)
    low = Column(Numeric(14, 4), nullable=False)
    close = Column(Numeric(14, 4), nullable=False)
    adj_close = Column(Numeric(14, 4), nullable=False)
    volume = Column(BigInteger, nullable=False)
    adj_factor = Column(Numeric(10, 6), default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    stock = relationship("Stock", back_populates="market_data")

    __table_args__ = (
        UniqueConstraint("date", "ticker", name="uq_market_data_date_ticker"),
        Index("idx_market_data_ticker_date", "ticker", "date"),
    )

    def __repr__(self) -> str:
        return f"<MarketData(ticker='{self.ticker}', date='{self.date}', close={self.close})>"


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer().with_variant(BigInteger, "postgresql"), primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    ticker = Column(String(20), ForeignKey("stocks.ticker", ondelete="CASCADE"), nullable=False, index=True)
    model = Column(String(50), nullable=False, index=True)
    probability_up = Column(Numeric(6, 5), nullable=False)  # 0.00000 to 1.00000
    prediction = Column(SmallInteger, nullable=False)        # 0 or 1
    signal = Column(String(10), nullable=False)             # 'BUY', 'HOLD', 'SELL'
    created_at = Column(DateTime, default=datetime.utcnow)

    stock = relationship("Stock", back_populates="predictions")

    __table_args__ = (
        UniqueConstraint("date", "ticker", "model", name="uq_predictions_date_ticker_model"),
        Index("idx_predictions_ticker_date_model", "ticker", "date", "model"),
    )

    def __repr__(self) -> str:
        return f"<Prediction(ticker='{self.ticker}', date='{self.date}', model='{self.model}', P(Up)={self.probability_up}, signal='{self.signal}')>"
