"""
Static Data Exporter for Netlify JAMstack Hosting (Track C - Ali).

Exports all necessary dashboard data from the SQLite database and CSV reports
into lightweight static JSON files under frontend/data/.
This enables 100% free hosting on Netlify without requiring a persistent Python backend.
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
from sqlalchemy import desc, func

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database.connection import get_db_session
from src.database.models import Stock, MarketData, Prediction

EXPORT_DIR = os.path.join(PROJECT_ROOT, "frontend", "data")
MODELS = ["ensemble", "lstm", "lightgbm", "logistic_regression", "xgboost", "random_forest"]
TICKERS = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]


def get_recommendation_text(signal: str, prob_up: float) -> str:
    if signal == "BUY":
        return f"Model merekomendasikan Buka Posisi Beli (P(Naik) = {prob_up:.1f}% ≥ 53%)."
    elif signal == "SELL":
        return f"Model merekomendasikan Jual / Amankan Kas (P(Naik) = {prob_up:.1f}% ≤ 47%)."
    else:
        return f"Model netral / Tidak ada sinyal kuat ({prob_up:.1f}%). Pertahankan posisi."


def export_stocks(session, base_dir):
    stocks = session.query(Stock).order_by(Stock.ticker).all()
    stock_items = []

    for stock in stocks:
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

        stock_items.append({
            "ticker": stock.ticker,
            "company_name": stock.company_name,
            "sector": stock.sector,
            "latest_date": latest_date_str,
            "latest_close": latest_close,
            "change_pct": change_pct
        })

    out_file = os.path.join(base_dir, "stocks.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"total": len(stock_items), "data": stock_items}, f, indent=2)
    print(f"Exported {len(stock_items)} stocks -> {out_file}")


def export_history(session, base_dir):
    hist_dir = os.path.join(base_dir, "history")
    os.makedirs(hist_dir, exist_ok=True)

    for ticker in TICKERS:
        rows = (
            session.query(MarketData)
            .filter(MarketData.ticker == ticker)
            .order_by(MarketData.date.asc())
            .all()
        )

        candle_items = [
            {
                "date": str(r.date),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "adj_close": float(r.adj_close),
                "volume": int(r.volume)
            }
            for r in rows
        ]

        out_file = os.path.join(hist_dir, f"{ticker}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "ticker": ticker,
                "timeframe": "ALL",
                "count": len(candle_items),
                "data": candle_items
            }, f)
        print(f"Exported {len(candle_items)} history candles for {ticker} -> {out_file}")


def export_predictions(session, base_dir):
    pred_dir = os.path.join(base_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for model in MODELS:
        max_date = session.query(func.max(Prediction.date)).filter(Prediction.model == model).scalar()
        if not max_date:
            continue

        records = (
            session.query(Prediction, Stock.company_name)
            .join(Stock, Prediction.ticker == Stock.ticker)
            .filter(Prediction.model == model, Prediction.date == max_date)
            .order_by(Prediction.ticker)
            .all()
        )

        items = []
        for pred, company_name in records:
            latest_md = (
                session.query(MarketData.close)
                .filter(MarketData.ticker == pred.ticker)
                .order_by(desc(MarketData.date))
                .first()
            )
            latest_close = float(latest_md[0]) if latest_md else None
            raw_prob = float(pred.probability_up)
            prob_pct = round(raw_prob * 100, 2)

            items.append({
                "ticker": pred.ticker,
                "company_name": company_name,
                "target_date": str(pred.date),
                "model": pred.model,
                "probability_up": prob_pct,
                "raw_probability": raw_prob,
                "signal": pred.signal,
                "prediction_label": "NAIK" if pred.prediction == 1 else "TURUN/FLAT",
                "latest_close": latest_close,
                "recommendation": get_recommendation_text(pred.signal, prob_pct)
            })

        out_file = os.path.join(pred_dir, f"{model}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "latest_target_date": str(max_date),
                "model": model,
                "count": len(items),
                "data": items
            }, f, indent=2)
        print(f"Exported predictions for model {model} -> {out_file}")


def export_signals(session, base_dir):
    sig_dir = os.path.join(base_dir, "signals")
    os.makedirs(sig_dir, exist_ok=True)

    for ticker in TICKERS:
        for model in MODELS:
            query = (
                session.query(
                    Prediction.date,
                    Prediction.signal,
                    Prediction.probability_up,
                    MarketData.close
                )
                .join(MarketData, (MarketData.ticker == Prediction.ticker) & (MarketData.date == Prediction.date))
                .filter(
                    Prediction.ticker == ticker,
                    Prediction.model == model,
                    Prediction.signal.in_(["BUY", "SELL"])
                )
                .order_by(Prediction.date.asc())
            )

            records = query.all()
            overlays = [
                {
                    "date": str(r.date),
                    "signal": r.signal,
                    "probability_up": round(float(r.probability_up) * 100, 2),
                    "price": float(r.close)
                }
                for r in records
            ]

            out_file = os.path.join(sig_dir, f"{ticker}_{model}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({
                    "ticker": ticker,
                    "model": model,
                    "count": len(overlays),
                    "data": overlays
                }, f)
            print(f"Exported {len(overlays)} signals for {ticker} ({model}) -> {out_file}")


def export_analytics(session, base_dir):
    # Backtest summary
    backtest_csv = os.path.join(PROJECT_ROOT, "reports", "trading_backtest_all.csv")
    if os.path.exists(backtest_csv):
        df = pd.read_csv(backtest_csv)
        items = []
        benchmark_item = None
        winner_item = None

        for _, row in df.iterrows():
            item = {
                "model": str(row["Model"]),
                "total_return_pct": round(float(row["Total Return (%)"]), 2),
                "excess_return_pct": round(float(row["Excess Return (%)"]), 2),
                "sharpe_ratio": round(float(row["Sharpe Ratio"]), 2),
                "max_drawdown_pct": round(float(row["Max Drawdown (%)"]), 2),
                "win_rate_pct": round(float(row["Win Rate (%)"]), 2),
                "trades": int(row["Trades"])
            }
            items.append(item)
            if "Benchmark" in item["model"] or "Buy & Hold" in item["model"]:
                benchmark_item = item
            if "Ensemble" in item["model"]:
                winner_item = item

        if not winner_item and items:
            winner_item = max(items, key=lambda x: x["total_return_pct"])
        if not benchmark_item and items:
            benchmark_item = items[0]

        out_file = os.path.join(base_dir, "backtest_summary.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_models": len(items),
                "benchmark": benchmark_item,
                "winner": winner_item,
                "data": items
            }, f, indent=2)
        print(f"Exported backtest summary -> {out_file}")

    # Model comparison
    model_comp_csv = os.path.join(PROJECT_ROOT, "reports", "model_comparison_all.csv")
    if os.path.exists(model_comp_csv):
        df_comp = pd.read_csv(model_comp_csv)
        items = []
        for _, row in df_comp.iterrows():
            items.append({
                "model": str(row["model"]),
                "roc_auc": round(float(row["roc_auc"]), 4),
                "brier_score": round(float(row["brier_score"]), 4),
                "log_loss": round(float(row["log_loss"]), 4),
                "accuracy": round(float(row["accuracy"]) * 100, 2),
                "precision": round(float(row["precision"]) * 100, 2),
                "recall": round(float(row["recall"]) * 100, 2),
                "f1_score": round(float(row["f1_score"]) * 100, 2),
                "threshold": float(row.get("threshold", 0.5))
            })

        out_file = os.path.join(base_dir, "model_comparison.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_models": len(items),
                "data": items
            }, f, indent=2)
        print(f"Exported model comparison -> {out_file}")

    # Model registry
    registry_file = os.path.join(PROJECT_ROOT, "reports", "models", "model_registry.json")
    if os.path.exists(registry_file):
        with open(registry_file, "r", encoding="utf-8") as f:
            reg_data = json.load(f)
        out_file = os.path.join(base_dir, "model_registry.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(reg_data, f, indent=2)
        print(f"Exported model registry -> {out_file}")

    # Health
    stock_count = session.query(Stock).count()
    market_count = session.query(MarketData).count()
    pred_count = session.query(Prediction).count()
    max_market_date = session.query(func.max(MarketData.date)).scalar()
    max_pred_date = session.query(func.max(Prediction.date)).scalar()

    health_data = {
        "status": "healthy",
        "mode": "static_netlify",
        "database_connected": True,
        "total_stocks": stock_count,
        "total_market_data": market_count,
        "total_predictions": pred_count,
        "latest_market_date": str(max_market_date) if max_market_date else None,
        "latest_prediction_date": str(max_pred_date) if max_pred_date else None,
        "active_models": MODELS,
        "timestamp": datetime.utcnow().isoformat()
    }
    out_file = os.path.join(base_dir, "health.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(health_data, f, indent=2)
    print(f"Exported health status -> {out_file}")


def run_full_export():
    print(f"=== Starting Static Data Export to {EXPORT_DIR} ===")
    os.makedirs(EXPORT_DIR, exist_ok=True)

    with get_db_session() as session:
        export_stocks(session, EXPORT_DIR)
        export_history(session, EXPORT_DIR)
        export_predictions(session, EXPORT_DIR)
        export_signals(session, EXPORT_DIR)
        export_analytics(session, EXPORT_DIR)

    print("=== Static Data Export Completed Successfully! ===")


if __name__ == "__main__":
    run_full_export()
