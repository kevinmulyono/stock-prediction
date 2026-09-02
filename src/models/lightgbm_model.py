"""
LightGBM Classifier Model Module (Track B - Daf).

Features:
- Binary classification with probability prediction P(Up) in [0, 1]
- Walk-forward cross validation callback support
- Early stopping against validation fold
- Feature importance analysis (Gain and Split metrics)
- Model persistence
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.technical_indicators import extract_features, get_feature_columns
from src.models.validation import evaluate_predictions, run_walk_forward_cv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Default optimized hyperparameters for tabular financial time series
DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}


class LightGBMStockModel:
    """
    LightGBM model wrapper for Next-Day Stock Price Direction Prediction.
    """
    def __init__(self, params: Optional[Dict] = None):
        self.params = params or DEFAULT_LGBM_PARAMS.copy()
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: List[str] = []
        self.feature_importances_: Optional[pd.DataFrame] = None
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 30
    ) -> "LightGBMStockModel":
        """
        Fits LightGBM model with optional early stopping on validation fold.
        """
        self.feature_names = list(X_train.columns)
        self.model = lgb.LGBMClassifier(**self.params)
        
        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )
        
        # Calculate feature importances
        gain_imp = self.model.booster_.feature_importance(importance_type="gain")
        split_imp = self.model.booster_.feature_importance(importance_type="split")
        
        self.feature_importances_ = pd.DataFrame({
            "feature": self.feature_names,
            "importance_gain": gain_imp,
            "importance_split": split_imp
        }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)
        
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns probability of Up movement: P(target = 1) in [0, 1].
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        # Ensure column order matches training
        X_aligned = X[self.feature_names]
        # Return probability of class 1 (Up)
        return self.model.predict_proba(X_aligned)[:, 1]
        
    def save(self, filepath: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_importances_": self.feature_importances_
        }, filepath)
        logger.info(f"Saved LightGBM model to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> "LightGBMStockModel":
        """Load model from disk."""
        data = joblib.load(filepath)
        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.feature_importances_ = data["feature_importances_"]
        return instance


def lightgbm_fold_trainer(
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, LightGBMStockModel]:
    """
    Callback function for Walk-Forward cross validation.
    """
    X_train, y_train = train_fold[feature_cols], train_fold["target"]
    X_val, y_val = val_fold[feature_cols], val_fold["target"]
    
    model = LightGBMStockModel()
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=25)
    y_val_prob = model.predict_proba(X_val)
    return y_val_prob, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM Model (Track B)")
    parser.add_argument("--data-file", default="data/processed/modeling_data_advanced.csv")
    parser.add_argument("--model-output", default="reports/models/lightgbm_model.joblib")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        logger.error(f"Dataset {args.data_file} not found. Generate it first!")
        sys.exit(1)
        
    df = pd.read_csv(args.data_file)
    feature_cols = get_feature_columns()
    
    from src.data.preparation import split_time_series
    train_df, val_df, test_df = split_time_series(df)
    
    # 1. Walk-Forward CV on Train Set
    cv_summary, fold_metrics = run_walk_forward_cv(
        train_df,
        feature_cols,
        lightgbm_fold_trainer,
        n_splits=5,
        model_name="LightGBM"
    )
    print("\n--- LightGBM Walk-Forward CV Results ---")
    print(cv_summary[["fold", "roc_auc", "accuracy", "f1_score", "brier_score"]].to_string(index=False))
    
    # 2. Train on full train_df, validate on val_df
    final_model = LightGBMStockModel()
    final_model.fit(
        train_df[feature_cols], train_df["target"],
        val_df[feature_cols], val_df["target"],
        early_stopping_rounds=30
    )
    
    # Evaluate on Validation Set
    val_prob = final_model.predict_proba(val_df[feature_cols])
    val_metrics = evaluate_predictions(val_df["target"].values, val_prob)
    
    print("\n--- LightGBM Validation Set (2024) Performance ---")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n--- Top 10 Feature Importances (Gain) ---")
    print(final_model.feature_importances_.head(10).to_string(index=False))
    
    # Save Model
    final_model.save(args.model_output)
