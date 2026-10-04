"""
Pydantic Schemas for API Requests & Responses (Track C - Ali).
"""

from typing import List, Optional, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field


# --- Stocks & Market Data ---
class StockItem(BaseModel):
    ticker: str
    company_name: str
    sector: str
    latest_date: Optional[str] = None
    latest_close: Optional[float] = None
    change_pct: Optional[float] = None


class StocksListResponse(BaseModel):
    total: int
    data: List[StockItem]


class MarketDataPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None


class MarketDataHistoryResponse(BaseModel):
    ticker: str
    timeframe: str
    count: int
    data: List[MarketDataPoint]


# --- Predictions ---
class PredictionItem(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    target_date: str
    model: str
    probability_up: float = Field(..., description="P(Up) in percentage, e.g. 64.25")
    raw_probability: float = Field(..., description="P(Up) between 0 and 1")
    signal: str = Field(..., description="BUY, HOLD, SELL")
    prediction_label: str = Field(..., description="NAIK or TURUN/FLAT")
    latest_close: Optional[float] = None
    recommendation: Optional[str] = None


class LatestPredictionsResponse(BaseModel):
    latest_target_date: str
    model: str
    count: int
    data: List[PredictionItem]


class SignalOverlayItem(BaseModel):
    date: str
    signal: str
    probability_up: float
    price: float


class SignalOverlayResponse(BaseModel):
    ticker: str
    model: str
    count: int
    data: List[SignalOverlayItem]


class TriggerPredictionRequest(BaseModel):
    model: Optional[str] = "ensemble"
    tickers: Optional[List[str]] = None


class TriggerPredictionResponse(BaseModel):
    status: str
    message: str
    model: str
    inserted_count: int
    predictions: List[Dict[str, Any]]


# --- Analytics & Models ---
class BacktestRow(BaseModel):
    model: str
    total_return_pct: float
    excess_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades: int


class BacktestSummaryResponse(BaseModel):
    total_models: int
    benchmark: BacktestRow
    winner: BacktestRow
    data: List[BacktestRow]


class ModelComparisonRow(BaseModel):
    model: str
    roc_auc: float
    brier_score: float
    log_loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    threshold: float


class ModelComparisonResponse(BaseModel):
    total_models: int
    data: List[ModelComparisonRow]


class SystemHealthResponse(BaseModel):
    status: str
    database_connected: bool
    total_stocks: int
    total_market_data: int
    total_predictions: int
    latest_market_date: Optional[str]
    latest_prediction_date: Optional[str]
    active_models: List[str]
    timestamp: str
