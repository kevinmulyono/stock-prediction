"""
Unit Tests for Automated Historical Backfill (Track B & C).

Validates:
1. ModelArtifactsManager loads without errors.
2. clean_orphan_predictions removes records strictly beyond max market date.
3. get_missing_dates_for_ticker correctly identifies missing date sequences.
"""

from datetime import date, timedelta
import pytest
from src.database.connection import get_db_session
from src.database.models import MarketData, Prediction
from src.models.auto_backfill import (
    ModelArtifactsManager,
    clean_orphan_predictions,
    get_missing_dates_for_ticker
)


def test_model_artifacts_manager_init():
    """Verify all pre-trained models and scalers load into memory without re-training."""
    manager = ModelArtifactsManager()
    assert manager.lgbm_model is not None, "LightGBM model artifact should be loaded."
    assert manager.lstm_model is not None, "PyTorch LSTM model artifact should be loaded."
    assert manager.lstm_scaler is not None, "LSTM scaler artifact should be loaded."
    assert len(manager.feature_cols) == 29, f"Expected 29 feature columns, got {len(manager.feature_cols)}"


def test_clean_orphan_predictions():
    """Verify orphan future predictions beyond latest candle are safely removed."""
    with get_db_session() as session:
        # Create a mock future prediction
        future_date = date(2099, 1, 1)
        test_pred = Prediction(
            date=future_date,
            ticker="BBCA.JK",
            model="ensemble",
            probability_up=0.55,
            prediction=1,
            signal="BUY"
        )
        session.add(test_pred)
        session.commit()

        # Run cleaner
        clean_orphan_predictions(session)

        # Verify it was removed
        remaining = session.query(Prediction).filter(Prediction.date == future_date).count()
        assert remaining == 0, "Orphan prediction with future date should have been removed."


def test_get_missing_dates_returns_empty_when_synced():
    """Verify that after backfill, missing dates for primary target stocks is 0."""
    with get_db_session() as session:
        missing = get_missing_dates_for_ticker(session, "BBCA.JK", model="ensemble")
        assert len(missing) == 0, f"Expected 0 missing dates for BBCA.JK, got {len(missing)}"

