"""
Automated Historical Backfill Engine for Stock Predictions (Track B & C).

Solves the missing signal indicators on newly synchronized candles:
1. Detects any dates in the SQLite 'market_data' table (>= 2025-01-01) that do not have
   corresponding predictions in the 'predictions' table.
2. Computes the 29 technical indicators without lookahead bias for the missing periods.
3. Evaluates all models (Ensemble, LSTM, LightGBM, plus Track A models) using
   strictly frozen pre-trained weights (NO re-training).
4. Stores predictions with date = candle_date, ensuring seamless JOIN with market_data
   so BUY/SELL marker indicators render perfectly on the financial charts.
5. Automatically exports updated static JSON data to frontend/data/ for Netlify.
"""

import argparse
from datetime import date, datetime, timedelta
import logging
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
import torch
from sqlalchemy import func

from src.database.connection import get_db_session, SessionLocal
from src.database.models import MarketData, Prediction, Stock
from src.features.technical_indicators import generate_single_stock_features, get_feature_columns
from src.models.lightgbm_model import LightGBMStockModel
from src.models.lstm_model import StockLSTM
from src.models.predict_daily import classify_signal, get_next_business_day, BUY_THRESHOLD, SELL_THRESHOLD, ENSEMBLE_WEIGHT_LGB, ENSEMBLE_WEIGHT_LSTM, SEQ_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

TARGET_TICKERS = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_PREDICTION_DATE = date(2025, 1, 1)


class ModelArtifactsManager:
    """Loads and caches all pre-trained model artifacts."""
    def __init__(self):
        self.feature_cols = get_feature_columns()
        self.lgbm_model = None
        self.lstm_model = None
        self.lstm_scaler = None
        self.lr_model = None
        self.rf_model = None
        self.xgb_model = None
        self.track_a_scaler = None
        self._load_all()

    def _load_all(self):
        # 1. LightGBM
        lgb_path = os.path.join(PROJECT_ROOT, "reports/models/lightgbm_model.joblib")
        if os.path.exists(lgb_path):
            self.lgbm_model = LightGBMStockModel.load(lgb_path)

        # 2. PyTorch LSTM
        lstm_pt_path = os.path.join(PROJECT_ROOT, "reports/models/lstm_model.pt")
        lstm_scaler_path = os.path.join(PROJECT_ROOT, "reports/models/lstm_scaler.joblib")
        if os.path.exists(lstm_pt_path) and os.path.exists(lstm_scaler_path):
            self.lstm_scaler = joblib.load(lstm_scaler_path)
            self.lstm_model = StockLSTM(input_dim=len(self.feature_cols)).to(DEVICE)
            self.lstm_model.load_state_dict(torch.load(lstm_pt_path, map_location=DEVICE))
            self.lstm_model.eval()

        # 3. Track A Models
        lr_path = os.path.join(PROJECT_ROOT, "reports/models/logistic_regression.joblib")
        rf_path = os.path.join(PROJECT_ROOT, "reports/models/random_forest.joblib")
        xgb_path = os.path.join(PROJECT_ROOT, "reports/models/xgboost.joblib")
        track_a_scaler_path = os.path.join(PROJECT_ROOT, "reports/models/track_a_scaler.joblib")

        if os.path.exists(lr_path):
            self.lr_model = joblib.load(lr_path)
        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)
        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
        if os.path.exists(track_a_scaler_path):
            self.track_a_scaler = joblib.load(track_a_scaler_path)

        logger.info("All model artifacts initialized successfully.")


def clean_orphan_predictions(session):
    """
    Deletes any prediction records whose date is strictly greater than the maximum
    available market_data date (e.g. future target dates that have no candle yet).
    """
    max_market_date = session.query(func.max(MarketData.date)).scalar()
    if max_market_date is not None:
        orphans = session.query(Prediction).filter(Prediction.date > max_market_date).delete()
        if orphans > 0:
            session.commit()
            logger.info(f"Cleaned {orphans} orphan future predictions (date > {max_market_date}).")


def get_missing_dates_for_ticker(session, ticker: str, model: str = "ensemble") -> List[date]:
    """
    Returns sorted list of dates present in market_data for this ticker (>= MIN_PREDICTION_DATE)
    that do not yet have a record in the predictions table.
    """
    market_dates = set(
        d[0] for d in session.query(MarketData.date)
        .filter(MarketData.ticker == ticker, MarketData.date >= MIN_PREDICTION_DATE)
        .order_by(MarketData.date.asc())
        .all()
    )

    pred_dates = set(
        d[0] for d in session.query(Prediction.date)
        .filter(Prediction.ticker == ticker, Prediction.model == model, Prediction.date >= MIN_PREDICTION_DATE)
        .all()
    )

    missing = sorted(list(market_dates - pred_dates))
    return missing


def backfill_ticker_predictions(
    ticker: str,
    missing_dates: List[date],
    artifacts: ModelArtifactsManager,
    session
) -> int:
    """
    Calculates features and generates predictions for the specified missing dates
    for a single ticker.
    """
    if not missing_dates:
        return 0

    first_missing = missing_dates[0]
    # We load market data starting at least 100 days prior to the first missing date
    # to guarantee full rolling indicator warmup (e.g. 50-day SMA, 26-day MACD)
    warmup_start = first_missing - timedelta(days=160)

    rows = (
        session.query(MarketData)
        .filter(MarketData.ticker == ticker, MarketData.date >= warmup_start)
        .order_by(MarketData.date.asc())
        .all()
    )

    if len(rows) < 50:
        logger.warning(f"[{ticker}] Insufficient history ({len(rows)} rows) for indicator warmup.")
        return 0

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "date": r.date,
            "ticker": r.ticker,
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
            "adj_close": float(r.adj_close),
            "volume": int(r.volume),
            "adj_factor": float(r.adj_factor)
        }
        for r in rows
    ])

    # Generate 29 technical indicators strictly backward-looking
    featured_df = generate_single_stock_features(df)
    feature_cols = artifacts.feature_cols

    # Ensure no NaN rows in our missing target dates
    missing_set = set(missing_dates)
    eval_df = featured_df[featured_df["date"].isin(missing_set)].copy()

    if eval_df.empty:
        return 0

    records_to_insert = []
    models_to_run = ["ensemble", "lightgbm", "lstm"]
    if artifacts.lr_model is not None:
        models_to_run.extend(["logistic_regression", "random_forest", "xgboost"])

    for _, row in eval_df.iterrows():
        candle_date = row["date"]
        # Find index in featured_df to extract sequence window
        idx = featured_df.index[featured_df["date"] == candle_date][0]

        # 1. LightGBM
        f_vec = featured_df.loc[[idx], feature_cols].fillna(0.0)
        p_lgb = float(artifacts.lgbm_model.predict_proba(f_vec)[0])

        # 2. LSTM Sequence
        seq_start = max(0, idx - SEQ_LENGTH + 1)
        seq_slice = featured_df.loc[seq_start:idx, feature_cols].fillna(0.0)
        if len(seq_slice) < SEQ_LENGTH:
            # Pad with first available row if sequence warmup is too short
            pad_needed = SEQ_LENGTH - len(seq_slice)
            pad_block = pd.concat([seq_slice.iloc[[0]]] * pad_needed, ignore_index=True)
            seq_slice = pd.concat([pad_block, seq_slice], ignore_index=True)

        seq_scaled = artifacts.lstm_scaler.transform(seq_slice.values)
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p_lstm = float(artifacts.lstm_model(seq_tensor).cpu().numpy()[0])

        # 3. Weighted Ensemble (Track B best)
        p_ens = float(np.clip(
            (ENSEMBLE_WEIGHT_LGB * p_lgb) + (ENSEMBLE_WEIGHT_LSTM * p_lstm),
            0.0,
            1.0
        ))

        probs_map = {
            "ensemble": p_ens,
            "lightgbm": p_lgb,
            "lstm": p_lstm
        }

        # 4. Track A Models
        if artifacts.track_a_scaler is not None:
            f_vec_scaled = artifacts.track_a_scaler.transform(f_vec)
            if artifacts.lr_model:
                probs_map["logistic_regression"] = float(artifacts.lr_model.predict_proba(f_vec_scaled)[0, 1])
            if artifacts.rf_model:
                probs_map["random_forest"] = float(artifacts.rf_model.predict_proba(f_vec_scaled)[0, 1])
            if artifacts.xgb_model:
                probs_map["xgboost"] = float(artifacts.xgb_model.predict_proba(f_vec_scaled)[0, 1])

        for m_name in models_to_run:
            if m_name in probs_map:
                p_val = probs_map[m_name]
                binary_pred, signal, _ = classify_signal(p_val)
                records_to_insert.append(
                    Prediction(
                        date=candle_date,
                        ticker=ticker,
                        model=m_name,
                        probability_up=round(float(p_val), 5),
                        prediction=binary_pred,
                        signal=signal
                    )
                )

    if records_to_insert:
        session.bulk_save_objects(records_to_insert)
        session.commit()
        logger.info(f"[{ticker}] Backfilled {len(records_to_insert)} model predictions across {len(eval_df)} candle dates.")

    return len(eval_df)


def run_auto_backfill(
    tickers: Optional[List[str]] = None,
    sync_live: bool = False,
    auto_export: bool = True
) -> Dict[str, int]:
    """
    Main entry point for automated historical backfilling.
    """
    tickers = tickers or TARGET_TICKERS
    logger.info("=== Starting Automated Historical Prediction Backfill ===")

    # 1. Optionally sync live candles from Yahoo Finance if online
    if sync_live:
        try:
            from src.data.sync_daily import sync_market_data
            sync_market_data(tickers)
        except Exception as e:
            logger.warning(f"Yahoo Finance sync skipped/warning: {e}")

    # 2. Clean any orphan records
    with get_db_session() as session:
        clean_orphan_predictions(session)

    # 3. Load artifacts once
    artifacts = ModelArtifactsManager()
    summary = {}

    with get_db_session() as session:
        for ticker in tickers:
            missing = get_missing_dates_for_ticker(session, ticker, model="ensemble")
            if missing:
                logger.info(f"[{ticker}] Found {len(missing)} candle dates missing predictions (from {missing[0]} to {missing[-1]}).")
                count = backfill_ticker_predictions(ticker, missing, artifacts, session)
                summary[ticker] = count
            else:
                logger.info(f"[{ticker}] Predictions are already fully up to date with market data.")
                summary[ticker] = 0

    total_backfilled = sum(summary.values())
    logger.info(f"=== Auto-Backfill Complete: {total_backfilled} total candle dates scored ===")

    # 4. Automatically re-export static data for Netlify & Dashboard
    if auto_export and total_backfilled > 0:
        logger.info("Updating static JSON artifacts in frontend/data/...")
        try:
            from src.export_static_data import run_full_export
            run_full_export()
        except Exception as export_err:
            logger.error(f"Failed to export static data: {export_err}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Backfill Model Predictions on New Candles")
    parser.add_argument("--sync-live", action="store_true", help="Sync latest candles from Yahoo Finance before backfill")
    parser.add_argument("--no-export", action="store_true", help="Skip static JSON export after backfill")
    args = parser.parse_args()

    run_auto_backfill(sync_live=args.sync_live, auto_export=not args.no_export)
