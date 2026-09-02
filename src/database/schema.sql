-- ============================================================================
-- AI Stock Movement Prediction & Backtesting Platform
-- PostgreSQL Database Schema Definition
-- Track A: Data Analyst / Data Scientist
-- ============================================================================

-- 1. STOCKS TABLE (Metadata)
CREATE TABLE IF NOT EXISTS stocks (
    stock_id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) UNIQUE NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    sector VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. MARKET DATA TABLE (Historical OHLCV + Cleaned Metrics)
CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    open NUMERIC(14, 4) NOT NULL,
    high NUMERIC(14, 4) NOT NULL,
    low NUMERIC(14, 4) NOT NULL,
    close NUMERIC(14, 4) NOT NULL,
    adj_close NUMERIC(14, 4) NOT NULL,
    volume BIGINT NOT NULL,
    adj_factor NUMERIC(10, 6) DEFAULT 1.000000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_market_data_date_ticker UNIQUE (date, ticker)
);

-- Indices for rapid querying by date range and ticker (used heavily by FastAPI & Backtest engine)
CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date ON market_data (ticker, date ASC);
CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data (date);

-- 3. PREDICTIONS TABLE (Model Inferences & Trading Signals)
-- Shared interface: Daf writes predictions, Ali reads for FastAPI dashboard
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    model VARCHAR(50) NOT NULL,
    probability_up NUMERIC(6, 5) NOT NULL,  -- e.g. 0.65432 (range 0.00000 to 1.00000)
    prediction SMALLINT NOT NULL,           -- 1 (Up) or 0 (Down)
    signal VARCHAR(10) NOT NULL,            -- 'BUY', 'HOLD', 'SELL'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_predictions_date_ticker_model UNIQUE (date, ticker, model)
);

-- Indices for querying model predictions
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date_model ON predictions (ticker, date ASC, model);
CREATE INDEX IF NOT EXISTS idx_predictions_model_date ON predictions (model, date);
