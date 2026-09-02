"""
Walk-Forward Validation Module for Time-Series Cross-Validation (Track B - Daf).

This module implements expanding-window walk-forward cross-validation strictly on
training data (2018-01-01 to 2023-12-31) to prevent time-series data leakage.
"""

import logging
import sys
from typing import Any, Callable, Dict, Generator, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def generate_walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_ratio: float = 0.5
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generates expanding-window Walk-Forward chronological splits.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training dataframe containing 'date' column.
    n_splits : int
        Number of walk-forward validation folds (e.g. 5).
    min_train_ratio : float
        Minimum initial training portion (e.g. 0.5 = 50% of the timeline).
        
    Yields:
    -------
    (train_fold_df, val_fold_df) for each fold.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    unique_dates = np.sort(df["date"].unique())
    total_dates = len(unique_dates)
    
    min_train_dates = int(total_dates * min_train_ratio)
    remaining_dates = total_dates - min_train_dates
    val_fold_size = remaining_dates // n_splits
    
    for fold in range(n_splits):
        train_end_idx = min_train_dates + (fold * val_fold_size)
        val_end_idx = train_end_idx + val_fold_size if fold < n_splits - 1 else total_dates
        
        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[train_end_idx:val_end_idx]
        
        train_fold = df[df["date"].isin(train_dates)].reset_index(drop=True)
        val_fold = df[df["date"].isin(val_dates)].reset_index(drop=True)
        
        yield train_fold, val_fold


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluates probabilistic predictions against ground truth binary targets.
    
    Calculates:
    - ROC-AUC
    - Brier Score (lower is better, measuring calibration & precision)
    - Log-Loss
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Safely handle single-class edge cases in small subsets
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
        
    try:
        loss = log_loss(y_true, y_prob, labels=[0, 1])
    except Exception:
        loss = 1.0
        
    brier = brier_score_loss(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "roc_auc": float(auc),
        "brier_score": float(brier),
        "log_loss": float(loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "threshold": float(threshold)
    }


def run_walk_forward_cv(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    train_and_predict_fn: Callable[[pd.DataFrame, pd.DataFrame, List[str]], Tuple[np.ndarray, Any]],
    n_splits: int = 5,
    model_name: str = "Model"
) -> Tuple[pd.DataFrame, List[Dict[str, float]]]:
    """
    Executes Walk-Forward CV across folds using the provided training callback.
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training set (e.g. 2018-2023).
    feature_cols : List[str]
        List of feature column names.
    train_and_predict_fn : Callable
        Function with signature (train_fold, val_fold, feature_cols) -> (y_val_prob, model_obj).
    n_splits : int
        Number of walk-forward folds.
    model_name : str
        Name of model for logging.
        
    Returns:
    --------
    Tuple: (summary_dataframe, list of fold metrics dictionaries)
    """
    logger.info(f"=== Starting Walk-Forward Cross-Validation for {model_name} ({n_splits} folds) ===")
    fold_metrics = []
    
    for fold_idx, (train_fold, val_fold) in enumerate(generate_walk_forward_splits(df_train, n_splits=n_splits), 1):
        train_start = train_fold["date"].min().strftime("%Y-%m-%d")
        train_end = train_fold["date"].max().strftime("%Y-%m-%d")
        val_start = val_fold["date"].min().strftime("%Y-%m-%d")
        val_end = val_fold["date"].max().strftime("%Y-%m-%d")
        
        logger.info(f"Fold {fold_idx}/{n_splits} | Train: {train_start} -> {train_end} ({len(train_fold)} rows) | Val: {val_start} -> {val_end} ({len(val_fold)} rows)")
        
        y_val_prob, _ = train_and_predict_fn(train_fold, val_fold, feature_cols)
        metrics = evaluate_predictions(val_fold["target"].values, y_val_prob)
        metrics["fold"] = fold_idx
        metrics["train_rows"] = len(train_fold)
        metrics["val_rows"] = len(val_fold)
        metrics["val_start"] = val_start
        metrics["val_end"] = val_end
        
        logger.info(
            f"Fold {fold_idx} Results: AUC={metrics['roc_auc']:.4f} | "
            f"Acc={metrics['accuracy']:.4f} | F1={metrics['f1_score']:.4f} | "
            f"Brier={metrics['brier_score']:.4f}"
        )
        fold_metrics.append(metrics)
        
    summary_df = pd.DataFrame(fold_metrics)
    avg_auc = summary_df["roc_auc"].mean()
    avg_acc = summary_df["accuracy"].mean()
    avg_f1 = summary_df["f1_score"].mean()
    avg_brier = summary_df["brier_score"].mean()
    
    logger.info(
        f"=== Walk-Forward CV Mean: AUC={avg_auc:.4f} (+/- {summary_df['roc_auc'].std():.4f}) | "
        f"Acc={avg_acc:.4f} | F1={avg_f1:.4f} | Brier={avg_brier:.4f} ==="
    )
    
    return summary_df, fold_metrics
