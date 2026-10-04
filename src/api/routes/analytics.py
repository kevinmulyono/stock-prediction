"""
Routes for Analytics, Backtest Leaderboard, and Model Comparisons (Track C - Ali).
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
import pandas as pd
from sqlalchemy import func

from src.database.connection import get_db_session
from src.database.models import Stock, MarketData, Prediction
from src.api.schemas import (
    BacktestRow,
    BacktestSummaryResponse,
    ModelComparisonRow,
    ModelComparisonResponse,
    SystemHealthResponse
)

router = APIRouter(prefix="/api", tags=["Analytics & Evaluation"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
BACKTEST_CSV = os.path.join(PROJECT_ROOT, "reports", "trading_backtest_all.csv")
MODEL_COMP_CSV = os.path.join(PROJECT_ROOT, "reports", "model_comparison_all.csv")
MODEL_REGISTRY_JSON = os.path.join(PROJECT_ROOT, "reports", "models", "model_registry.json")


@router.get("/backtest/summary", response_model=BacktestSummaryResponse)
def get_backtest_summary():
    """
    Returns realistic backtest simulation performance metrics (2025-2026 Test Set)
    including IDX trading costs (0.15%), showing how the AI strategy outperforms
    the Buy & Hold benchmark (+22.22% vs -0.40%).
    """
    if not os.path.exists(BACKTEST_CSV):
        raise HTTPException(status_code=404, detail="trading_backtest_all.csv not found.")

    try:
        df = pd.read_csv(BACKTEST_CSV)
        # Expected cols: Model,Total Return (%),Excess Return (%),Sharpe Ratio,Max Drawdown (%),Win Rate (%),Trades
        items = []
        benchmark_item = None
        winner_item = None

        for _, row in df.iterrows():
            item = BacktestRow(
                model=str(row["Model"]),
                total_return_pct=round(float(row["Total Return (%)"]), 2),
                excess_return_pct=round(float(row["Excess Return (%)"]), 2),
                sharpe_ratio=round(float(row["Sharpe Ratio"]), 2),
                max_drawdown_pct=round(float(row["Max Drawdown (%)"]), 2),
                win_rate_pct=round(float(row["Win Rate (%)"]), 2),
                trades=int(row["Trades"])
            )
            items.append(item)
            if "Benchmark" in item.model or "Buy & Hold" in item.model:
                benchmark_item = item
            if "Ensemble" in item.model:
                winner_item = item

        if not winner_item and items:
            winner_item = max(items, key=lambda x: x.total_return_pct)
        if not benchmark_item and items:
            benchmark_item = items[0]

        return BacktestSummaryResponse(
            total_models=len(items),
            benchmark=benchmark_item,
            winner=winner_item,
            data=items
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading backtest data: {str(e)}")


@router.get("/models/comparison", response_model=ModelComparisonResponse)
def get_model_comparison():
    """
    Returns machine learning evaluation metrics (ROC-AUC, Brier score, accuracy, precision, recall, F1).
    """
    if not os.path.exists(MODEL_COMP_CSV):
        raise HTTPException(status_code=404, detail="model_comparison_all.csv not found.")

    try:
        df = pd.read_csv(MODEL_COMP_CSV)
        items = []
        for _, row in df.iterrows():
            items.append(
                ModelComparisonRow(
                    model=str(row["model"]),
                    roc_auc=round(float(row["roc_auc"]), 4),
                    brier_score=round(float(row["brier_score"]), 4),
                    log_loss=round(float(row["log_loss"]), 4),
                    accuracy=round(float(row["accuracy"]) * 100, 2),
                    precision=round(float(row["precision"]) * 100, 2),
                    recall=round(float(row["recall"]) * 100, 2),
                    f1_score=round(float(row["f1_score"]) * 100, 2),
                    threshold=float(row.get("threshold", 0.5))
                )
            )

        return ModelComparisonResponse(
            total_models=len(items),
            data=items
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading model comparison: {str(e)}")


@router.get("/models/registry")
def get_model_registry() -> Dict[str, Any]:
    """
    Returns model registry configurations, optimal ensemble weights (0.15 LGB + 0.85 LSTM),
    and 29 engineered technical indicators list.
    """
    if not os.path.exists(MODEL_REGISTRY_JSON):
        raise HTTPException(status_code=404, detail="model_registry.json not found.")

    with open(MODEL_REGISTRY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@router.get("/health", response_model=SystemHealthResponse)
def get_system_health():
    """
    Health check endpoint reporting database integrity and data freshness.
    """
    try:
        with get_db_session() as session:
            stock_count = session.query(Stock).count()
            market_count = session.query(MarketData).count()
            pred_count = session.query(Prediction).count()

            max_market_date = session.query(func.max(MarketData.date)).scalar()
            max_pred_date = session.query(func.max(Prediction.date)).scalar()

            models = [
                m[0] for m in session.query(Prediction.model).distinct().all()
            ]

            return SystemHealthResponse(
                status="healthy",
                database_connected=True,
                total_stocks=stock_count,
                total_market_data=market_count,
                total_predictions=pred_count,
                latest_market_date=str(max_market_date) if max_market_date else None,
                latest_prediction_date=str(max_pred_date) if max_pred_date else None,
                active_models=models,
                timestamp=datetime.utcnow().isoformat()
            )
    except Exception as e:
        return SystemHealthResponse(
            status="degraded",
            database_connected=False,
            total_stocks=0,
            total_market_data=0,
            total_predictions=0,
            latest_market_date=None,
            latest_prediction_date=None,
            active_models=[],
            timestamp=datetime.utcnow().isoformat()
        )
