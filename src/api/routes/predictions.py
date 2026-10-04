"""
Routes for Prediction & Trading Signal Endpoints (Track C - Ali).
"""

from datetime import date, timedelta
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from sqlalchemy import func, desc

from src.database.connection import get_db_session
from src.database.models import Stock, MarketData, Prediction
from src.models.predict_daily import run_daily_prediction
from src.api.schemas import (
    PredictionItem,
    LatestPredictionsResponse,
    SignalOverlayItem,
    SignalOverlayResponse,
    TriggerPredictionRequest,
    TriggerPredictionResponse
)

router = APIRouter(prefix="/api/predictions", tags=["Predictions & Signals"])


def get_recommendation_text(signal: str, prob_up: float) -> str:
    if signal == "BUY":
        return f"Model merekomendasikan Buka Posisi Beli (P(Naik) = {prob_up:.1f}% ≥ 53%)."
    elif signal == "SELL":
        return f"Model merekomendasikan Jual / Amankan Kas (P(Naik) = {prob_up:.1f}% ≤ 47%)."
    else:
        return f"Model netral / Tidak ada sinyal kuat ({prob_up:.1f}%). Pertahankan posisi."


@router.get("/latest", response_model=LatestPredictionsResponse)
def get_latest_predictions(
    model: str = Query("ensemble", description="Model filter: ensemble, lstm, lightgbm, logistic_regression, xgboost, random_forest")
):
    """
    Returns latest T+1 stock movement predictions and trading signals for all 5 bluechip stocks.
    """
    with get_db_session() as session:
        # Find maximum prediction date for this model
        max_date = session.query(func.max(Prediction.date)).filter(Prediction.model == model).scalar()
        if not max_date:
            raise HTTPException(status_code=404, detail=f"No predictions found for model '{model}'.")

        records = (
            session.query(Prediction, Stock.company_name)
            .join(Stock, Prediction.ticker == Stock.ticker)
            .filter(Prediction.model == model, Prediction.date == max_date)
            .order_by(Prediction.ticker)
            .all()
        )

        results = []
        for pred, company_name in records:
            # Get latest market close price for context
            latest_md = (
                session.query(MarketData.close)
                .filter(MarketData.ticker == pred.ticker)
                .order_by(desc(MarketData.date))
                .first()
            )
            latest_close = float(latest_md[0]) if latest_md else None

            raw_prob = float(pred.probability_up)
            prob_pct = round(raw_prob * 100, 2)

            results.append(
                PredictionItem(
                    ticker=pred.ticker,
                    company_name=company_name,
                    target_date=str(pred.date),
                    model=pred.model,
                    probability_up=prob_pct,
                    raw_probability=raw_prob,
                    signal=pred.signal,
                    prediction_label="NAIK" if pred.prediction == 1 else "TURUN/FLAT",
                    latest_close=latest_close,
                    recommendation=get_recommendation_text(pred.signal, prob_pct)
                )
            )

        return LatestPredictionsResponse(
            latest_target_date=str(max_date),
            model=model,
            count=len(results),
            data=results
        )


@router.get("/signals-overlay", response_model=SignalOverlayResponse)
def get_signals_overlay(
    ticker: str = Query(..., description="Stock ticker e.g. BBCA.JK"),
    model: str = Query("ensemble", description="Model filter: ensemble, lstm, lightgbm, etc."),
    timeframe: str = Query("1Y", pattern="^(1M|3M|6M|1Y|ALL)$")
):
    """
    Returns historical signals matched with close prices to render BUY/SELL markers
    on top of the financial charts.
    """
    normalized_ticker = ticker.upper()
    if not normalized_ticker.endswith(".JK"):
        normalized_ticker = f"{normalized_ticker}.JK"

    with get_db_session() as session:
        # Determine cutoff date
        max_date = session.query(func.max(MarketData.date)).filter(MarketData.ticker == normalized_ticker).scalar()
        if not max_date:
            raise HTTPException(status_code=404, detail=f"No data found for {normalized_ticker}")

        query = (
            session.query(
                Prediction.date,
                Prediction.signal,
                Prediction.probability_up,
                MarketData.close
            )
            .join(MarketData, (MarketData.ticker == Prediction.ticker) & (MarketData.date == Prediction.date))
            .filter(
                Prediction.ticker == normalized_ticker,
                Prediction.model == model,
                Prediction.signal.in_(["BUY", "SELL"])  # We only mark actionable BUY / SELL signals
            )
        )

        if timeframe != "ALL":
            days_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
            cutoff = max_date - timedelta(days=days_map.get(timeframe, 365))
            query = query.filter(Prediction.date >= cutoff)

        records = query.order_by(Prediction.date.asc()).all()

        overlays = [
            SignalOverlayItem(
                date=str(row.date),
                signal=row.signal,
                probability_up=round(float(row.probability_up) * 100, 2),
                price=float(row.close)
            )
            for row in records
        ]

        return SignalOverlayResponse(
            ticker=normalized_ticker,
            model=model,
            count=len(overlays),
            data=overlays
        )


@router.get("/history")
def get_prediction_history(
    ticker: str = Query(..., description="Stock ticker e.g. BBCA.JK"),
    model: str = Query("ensemble"),
    limit: int = Query(100, ge=1, le=500)
):
    """
    Returns historical predictions for a given ticker and model.
    """
    normalized_ticker = ticker.upper()
    if not normalized_ticker.endswith(".JK"):
        normalized_ticker = f"{normalized_ticker}.JK"

    with get_db_session() as session:
        records = (
            session.query(Prediction)
            .filter(Prediction.ticker == normalized_ticker, Prediction.model == model)
            .order_by(desc(Prediction.date))
            .limit(limit)
            .all()
        )

        return {
            "ticker": normalized_ticker,
            "model": model,
            "count": len(records),
            "data": [
                {
                    "date": str(r.date),
                    "probability_up": round(float(r.probability_up) * 100, 2),
                    "prediction": r.prediction,
                    "signal": r.signal,
                    "created_at": str(r.created_at)
                }
                for r in records
            ]
        }


@router.post("/trigger-daily", response_model=TriggerPredictionResponse)
def trigger_daily_prediction(request: TriggerPredictionRequest):
    """
    Runs the live daily scoring pipeline (predict_daily.py) for the selected model
    and automatically persists T+1 predictions into the SQLite database.
    """
    try:
        model_type = request.model or "ensemble"
        result_df = run_daily_prediction(
            model=model_type,
            tickers=request.tickers,
            save_to_db=True
        )

        return TriggerPredictionResponse(
            status="success",
            message=f"Successfully generated and committed daily {model_type} predictions to database.",
            model=model_type,
            inserted_count=len(result_df),
            predictions=result_df.to_dict(orient="records")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute daily prediction pipeline: {str(e)}")
