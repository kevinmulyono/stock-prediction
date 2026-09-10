"""
Daily Live Prediction Pipeline (Track B - Daf).

Operational live daily scoring (T+1):
1. Loads the latest market data (60+ days) for the 5 target IDX stocks:
   - BBCA.JK (Bank Central Asia)
   - BBRI.JK (Bank Rakyat Indonesia)
   - BMRI.JK (Bank Mandiri)
   - TLKM.JK (Telkom Indonesia)
   - ASII.JK (Astra International)
2. Computes 29 backward-looking technical indicators strictly without lookahead bias.
3. Loads pre-trained model artifacts from reports/models/ without re-training:
   - LightGBM (reports/models/lightgbm_model.joblib)
   - PyTorch LSTM (reports/models/lstm_model.pt & reports/models/lstm_scaler.joblib)
   - Weighted Ensemble (0.15 LightGBM + 0.85 LSTM)
4. Evaluates probability P(Up) and maps to discrete trading signals:
   - P(Up) >= 0.53 -> BUY
   - P(Up) <= 0.47 -> SELL
   - 0.47 < P(Up) < 0.53 -> HOLD
5. Automatically ingests predictions into the SQLite 'predictions' table
   with date = next_trading_day (T+1).

Usage:
  python src/models/predict_daily.py --model ensemble
  python src/models/predict_daily.py --model lightgbm
  python src/models/predict_daily.py --model lstm
  python src/models/predict_daily.py --model all
"""

import argparse
from datetime import date, datetime, timedelta
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import numpy as np
import pandas as pd
import torch

from src.database.connection import SessionLocal
from src.database.models import MarketData, Prediction, Stock
from src.features.technical_indicators import generate_single_stock_features, get_feature_columns
from src.models.lightgbm_model import LightGBMStockModel
from src.models.lstm_model import StockLSTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

TARGET_TICKERS = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]
BUY_THRESHOLD = 0.53
SELL_THRESHOLD = 0.47
ENSEMBLE_WEIGHT_LGB = 0.15
ENSEMBLE_WEIGHT_LSTM = 0.85
SEQ_LENGTH = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_next_business_day(as_of_date: date) -> date:
    """
    Computes the next trading business day (T+1), skipping weekends.
    """
    next_day = as_of_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    return next_day


def load_recent_market_data(
    tickers: List[str],
    as_of_date: Optional[date] = None,
    lookback_days: int = 80
) -> Dict[str, pd.DataFrame]:
    """
    Fetches the last N trading days of market data from SQLite database for each ticker.
    """
    session = SessionLocal()
    data_by_ticker = {}
    try:
        for ticker in tickers:
            query = session.query(MarketData).filter(MarketData.ticker == ticker)
            if as_of_date is not None:
                query = query.filter(MarketData.date <= as_of_date)
            query = query.order_by(MarketData.date.desc()).limit(lookback_days)
            
            df = pd.read_sql(query.statement, session.bind)
            if df.empty:
                logger.warning(f"No market data found for {ticker}")
                continue
                
            df = df.sort_values(by="date").reset_index(drop=True)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            data_by_ticker[ticker] = df
            logger.info(f"Loaded {len(df)} rows for {ticker} (latest: {df['date'].iloc[-1]})")
            
        return data_by_ticker
    finally:
        session.close()


class DailyPredictor:
    """
    Inference engine that loads trained models and scores live market features.
    """
    def __init__(
        self,
        lgbm_model_path: str = "reports/models/lightgbm_model.joblib",
        lstm_model_path: str = "reports/models/lstm_model.pt",
        lstm_scaler_path: str = "reports/models/lstm_scaler.joblib",
        feature_cols: Optional[List[str]] = None
    ):
        self.feature_cols = feature_cols or get_feature_columns()
        self.lgbm_model_path = lgbm_model_path
        self.lstm_model_path = lstm_model_path
        self.lstm_scaler_path = lstm_scaler_path
        
        self.lgbm_model: Optional[LightGBMStockModel] = None
        self.lstm_model: Optional[StockLSTM] = None
        self.lstm_scaler = None

    def load_lightgbm(self):
        if self.lgbm_model is None:
            if not os.path.exists(self.lgbm_model_path):
                raise FileNotFoundError(f"LightGBM artifact not found at {self.lgbm_model_path}")
            logger.info(f"Loading LightGBM model from {self.lgbm_model_path}...")
            self.lgbm_model = LightGBMStockModel.load(self.lgbm_model_path)

    def load_lstm(self):
        if self.lstm_model is None:
            if not os.path.exists(self.lstm_model_path):
                raise FileNotFoundError(f"LSTM artifact not found at {self.lstm_model_path}")
            if not os.path.exists(self.lstm_scaler_path):
                raise FileNotFoundError(f"LSTM scaler artifact not found at {self.lstm_scaler_path}")
                
            logger.info(f"Loading LSTM model from {self.lstm_model_path}...")
            self.lstm_scaler = joblib.load(self.lstm_scaler_path)
            self.lstm_model = StockLSTM(input_dim=len(self.feature_cols)).to(DEVICE)
            self.lstm_model.load_state_dict(torch.load(self.lstm_model_path, map_location=DEVICE))
            self.lstm_model.eval()

    def predict_ticker(
        self,
        ticker_df: pd.DataFrame,
        model_type: str = "ensemble"
    ) -> Dict[str, float]:
        """
        Computes technical features and runs model prediction for a single ticker.
        Returns dict with probabilities and latest market info.
        """
        if len(ticker_df) < 50:
            raise ValueError(f"Insufficient history: {len(ticker_df)} rows. Need at least 50 days for indicators.")
            
        featured_df = generate_single_stock_features(ticker_df)
        last_row = featured_df.iloc[-1]
        
        # Verify no NaN in the 29 features of the latest row
        nan_cols = [c for c in self.feature_cols if pd.isna(last_row[c])]
        if nan_cols:
            raise ValueError(f"NaN values found in features for {ticker_df['ticker'].iloc[0]}: {nan_cols}")
            
        probs = {}
        
        # LightGBM
        if model_type in ("lightgbm", "ensemble", "all"):
            self.load_lightgbm()
            feature_vector = featured_df[self.feature_cols].iloc[[-1]]
            p_lgb = float(self.lgbm_model.predict_proba(feature_vector)[0])
            probs["lightgbm"] = p_lgb
            
        # LSTM
        if model_type in ("lstm", "ensemble", "all"):
            self.load_lstm()
            if len(featured_df) < SEQ_LENGTH:
                raise ValueError(f"Insufficient data for sequence window: {len(featured_df)} < {SEQ_LENGTH}")
            seq_raw = featured_df[self.feature_cols].iloc[-SEQ_LENGTH:].values
            seq_scaled = self.lstm_scaler.transform(seq_raw)
            seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p_lstm = float(self.lstm_model(seq_tensor).cpu().numpy()[0])
            probs["lstm"] = p_lstm
            
        # Ensemble
        if model_type in ("ensemble", "all"):
            p_ens = (ENSEMBLE_WEIGHT_LGB * probs["lightgbm"]) + (ENSEMBLE_WEIGHT_LSTM * probs["lstm"])
            probs["ensemble"] = float(np.clip(p_ens, 0.0, 1.0))
            
        return {
            "ticker": str(last_row["ticker"]),
            "as_of_date": last_row["date"],
            "close_price": float(last_row["close"]),
            "probabilities": probs
        }


def classify_signal(prob: float, buy_th: float = BUY_THRESHOLD, sell_th: float = SELL_THRESHOLD) -> Tuple[int, str, str]:
    """
    Converts P(Up) into binary prediction, trading signal, and action advice.
    """
    prediction = 1 if prob >= 0.50 else 0
    if prob >= buy_th:
        signal = "BUY"
        advice = "Bullish momentum detected. Take Long position (+1.0)."
    elif prob <= sell_th:
        signal = "SELL"
        advice = "Bearish/Downside risk. Close Long or stay in Cash (0.0)."
    else:
        signal = "HOLD"
        advice = "Neutral / low conviction zone. Maintain previous position."
    return prediction, signal, advice


def save_prediction_to_database(records: List[Dict]) -> int:
    """
    Saves or updates daily prediction records into the database.
    """
    session = SessionLocal()
    inserted_count = 0
    try:
        for rec in records:
            pred_date = rec["date"]
            ticker = rec["ticker"]
            model = rec["model"]
            
            # Upsert: delete existing prediction for this (date, ticker, model) if any
            session.query(Prediction).filter(
                Prediction.date == pred_date,
                Prediction.ticker == ticker,
                Prediction.model == model
            ).delete()
            
            p_obj = Prediction(
                date=pred_date,
                ticker=ticker,
                model=model,
                probability_up=round(float(rec["probability_up"]), 5),
                prediction=int(rec["prediction"]),
                signal=str(rec["signal"])
            )
            session.add(p_obj)
            inserted_count += 1
            
        session.commit()
        logger.info(f"Successfully committed {inserted_count} predictions to 'predictions' table.")
        return inserted_count
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to commit predictions to database: {e}")
        raise e
    finally:
        session.close()


def run_daily_prediction(
    model: str = "ensemble",
    tickers: Optional[List[str]] = None,
    as_of_date_str: Optional[str] = None,
    target_date_str: Optional[str] = None,
    save_to_db: bool = True
) -> pd.DataFrame:
    """
    Main orchestration routine for daily inference and T+1 scoring.
    """
    tickers = tickers or TARGET_TICKERS
    as_of_date = datetime.strptime(as_of_date_str, "%Y-%m-%d").date() if as_of_date_str else None
    
    logger.info(f"=== Starting Daily Prediction Pipeline (Model: {model}) ===")
    
    # 1. Load recent market data from DB
    data_by_ticker = load_recent_market_data(tickers, as_of_date=as_of_date, lookback_days=80)
    if not data_by_ticker:
        raise RuntimeError("No market data retrieved. Cannot proceed with prediction.")
        
    predictor = DailyPredictor()
    results = []
    db_records = []
    
    # Identify models to score
    model_list = ["ensemble"] if model == "ensemble" else (
        ["lightgbm"] if model == "lightgbm" else (
            ["lstm"] if model == "lstm" else ["lightgbm", "lstm", "ensemble"]
        )
    )
    
    for ticker in tickers:
        if ticker not in data_by_ticker:
            logger.warning(f"Skipping {ticker}: No data available.")
            continue
            
        ticker_data = data_by_ticker[ticker]
        pred_output = predictor.predict_ticker(ticker_data, model_type=model)
        
        ticker_as_of = pred_output["as_of_date"]
        if target_date_str:
            t_plus_1 = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        else:
            t_plus_1 = get_next_business_day(ticker_as_of)
            
        close_price = pred_output["close_price"]
        
        for m_name in model_list:
            p_up = pred_output["probabilities"][m_name]
            binary_pred, signal, advice = classify_signal(p_up)
            
            results.append({
                "Ticker": ticker,
                "Model": m_name,
                "As Of Date (T)": ticker_as_of.strftime("%Y-%m-%d"),
                "Close Price": f"Rp {close_price:,.0f}",
                "Target Date (T+1)": t_plus_1.strftime("%Y-%m-%d"),
                "P(Up)": round(p_up, 4),
                "Signal": signal,
                "Recommendation": advice
            })
            
            db_records.append({
                "date": t_plus_1,
                "ticker": ticker,
                "model": m_name,
                "probability_up": p_up,
                "prediction": binary_pred,
                "signal": signal
            })
            
    summary_df = pd.DataFrame(results)
    
    # 2. Database Ingestion
    if save_to_db and db_records:
        save_prediction_to_database(db_records)
        
    # 3. Print CLI Display Table
    print("\n" + "=" * 92)
    print("           LIVE DAILY STOCK MOVEMENT PREDICTIONS (T+1 FORECAST)           ")
    print("=" * 92)
    print(summary_df[["Ticker", "Model", "As Of Date (T)", "Close Price", "Target Date (T+1)", "P(Up)", "Signal"]].to_string(index=False))
    print("-" * 92)
    print("Trading Recommendation Summary:")
    for _, row in summary_df[summary_df["Model"] == (model if model != "all" else "ensemble")].iterrows():
        print(f"  * {row['Ticker']:<8} | P(Up): {row['P(Up)']:.4f} -> [{row['Signal']}] : {row['Recommendation']}")
    print("=" * 92 + "\n")
    
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Daily Stock Movement Prediction (T+1 Scoring)")
    parser.add_argument(
        "--model",
        default="ensemble",
        choices=["ensemble", "lightgbm", "lstm", "all"],
        help="Model architecture to use (default: ensemble)"
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated list of tickers (e.g. BBCA.JK,BBRI.JK)"
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Historical as-of date (YYYY-MM-DD). Defaults to latest available in database."
    )
    parser.add_argument(
        "--target-date",
        default=None,
        help="Prediction target date (YYYY-MM-DD). Defaults to next business day."
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="If set, skips saving predictions to database."
    )
    
    args = parser.parse_args()
    tickers_list = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    
    run_daily_prediction(
        model=args.model,
        tickers=tickers_list,
        as_of_date_str=args.as_of_date,
        target_date_str=args.target_date,
        save_to_db=not args.no_db
    )
