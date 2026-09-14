"""
PyTorch LSTM / GRU Deep Learning Sequence Model (Track B - Daf).

Features:
- Sliding window sequence dataset construction isolated per ticker
- Multi-layer LSTM / GRU with Dropout and LayerNorm
- Fit scaler ONLY on train set to strictly prevent data leakage
- Early stopping based on validation loss / ROC-AUC
- Outputs probability P(Up) in [0, 1]
- Walk-forward cross validation support
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.features.technical_indicators import get_feature_columns
from src.models.validation import evaluate_predictions, run_walk_forward_cv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Device configuration (CPU or CUDA)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockSequenceDataset(Dataset):
    """
    PyTorch Dataset for multi-ticker time series sliding windows.
    Constructed per ticker to guarantee zero sequence bleed between stocks.
    """
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_sequences_by_ticker(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = 15,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
    warmup_df: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, StandardScaler]:
    """
    Builds sliding window sequences per ticker.
    Supports warmup_df to provide sequence history for out-of-sample sets without dropping initial rows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Market dataframe containing features and optionally 'target'
    feature_cols : List[str]
        Feature columns to scale and sequence
    seq_length : int
        Number of historical timesteps per sequence (e.g. 15 days)
    scaler : StandardScaler
        Optional pre-fitted scaler
    fit_scaler : bool
        If True, fits the scaler on df (must only be True for training set)
    warmup_df : Optional[pd.DataFrame]
        Historical dataframe (e.g. last rows of validation set) used as sequence history.
        
    Returns:
    --------
    Tuple: (sequences, targets, aligned_dataframe_metadata, scaler)
    """
    df = df.copy().sort_values(by=["ticker", "date"]).reset_index(drop=True)
    
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(df[feature_cols].values)
    elif scaler is None:
        raise ValueError("Scaler must be provided when fit_scaler is False.")
        
    all_seqs = []
    all_targets = []
    meta_rows = []
    
    if warmup_df is not None:
        warmup_df = warmup_df.copy().sort_values(by=["ticker", "date"]).reset_index(drop=True)
    
    for ticker, group in df.groupby("ticker"):
        group_df = group.reset_index(drop=True)
        has_target = "target" in group_df.columns
        targets = group_df["target"].values if has_target else np.zeros(len(group_df), dtype=np.float32)
        
        if warmup_df is not None and ticker in warmup_df["ticker"].values:
            ticker_warmup = warmup_df[warmup_df["ticker"] == ticker].tail(seq_length - 1).reset_index(drop=True)
            combined = pd.concat([ticker_warmup, group_df], ignore_index=True)
            scaled_combined = scaler.transform(combined[feature_cols].values)
            n_warmup = len(ticker_warmup)
            
            for j in range(len(group_df)):
                idx = n_warmup + j
                if idx - seq_length + 1 >= 0:
                    seq = scaled_combined[idx - seq_length + 1 : idx + 1]
                    all_seqs.append(seq)
                    all_targets.append(targets[j])
                    meta_rows.append(group_df.iloc[j])
        else:
            scaled_features = scaler.transform(group_df[feature_cols].values)
            n_rows = len(group_df)
            if n_rows < seq_length:
                continue
                
            for i in range(seq_length - 1, n_rows):
                seq = scaled_features[i - seq_length + 1 : i + 1]
                all_seqs.append(seq)
                all_targets.append(targets[i])
                meta_rows.append(group_df.iloc[i])
            
    seq_array = np.array(all_seqs, dtype=np.float32)
    target_array = np.array(all_targets, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)
    
    return seq_array, target_array, meta_df, scaler


class StockLSTM(nn.Module):
    """
    LSTM Architecture for Stock Movement Prediction.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Take hidden state of last timestep
        last_timestep = out[:, -1, :]
        prob = self.fc(last_timestep)
        return prob.squeeze(1)


class LSTMStockTrainer:
    """
    Trainer wrapper for LSTM model with early stopping, scaling, and sequence inference.
    """
    def __init__(
        self,
        input_dim: int,
        seq_length: int = 15,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 50,
        early_stopping_patience: int = 10
    ):
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.model = StockLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(DEVICE)
        
        self.scaler: Optional[StandardScaler] = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_auc": []}

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> "LSTMStockTrainer":
        """
        Fits LSTM model using training sequences and early stops on validation sequences.
        """
        # Fit scaler on train_df ONLY
        X_train_seq, y_train_seq, _, self.scaler = build_sequences_by_ticker(
            train_df, feature_cols, seq_length=self.seq_length, fit_scaler=True
        )
        
        # Transform val_df using fitted scaler
        X_val_seq, y_val_seq, _, _ = build_sequences_by_ticker(
            val_df, feature_cols, seq_length=self.seq_length, scaler=self.scaler, fit_scaler=False
        )
        
        train_dataset = StockSequenceDataset(X_train_seq, y_train_seq)
        val_dataset = StockSequenceDataset(X_val_seq, y_val_seq)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(y_batch)
                
            avg_train_loss = total_train_loss / len(train_dataset)
            
            # Validation evaluation
            self.model.eval()
            total_val_loss = 0.0
            val_preds_list = []
            val_targets_list = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    preds = self.model(X_batch)
                    loss = criterion(preds, y_batch)
                    total_val_loss += loss.item() * len(y_batch)
                    val_preds_list.extend(preds.cpu().numpy())
                    val_targets_list.extend(y_batch.cpu().numpy())
                    
            avg_val_loss = total_val_loss / len(val_dataset)
            val_metrics = evaluate_predictions(np.array(val_targets_list), np.array(val_preds_list))
            
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_auc"].append(val_metrics["roc_auc"])
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                    break
                    
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            
        return self

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        warmup_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generates probability predictions P(Up) for sequences in the dataframe.
        Supports warmup_df to provide sequence history without dropping rows.
        Returns (probabilities, aligned_dataframe_metadata).
        """
        self.model.eval()
        X_seq, _, meta_df, _ = build_sequences_by_ticker(
            df, feature_cols, seq_length=self.seq_length, scaler=self.scaler, fit_scaler=False, warmup_df=warmup_df
        )
        
        dataset = StockSequenceDataset(X_seq, np.zeros(len(X_seq)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        preds_list = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(DEVICE)
                probs = self.model(X_batch)
                preds_list.extend(probs.cpu().numpy())
                
        return np.array(preds_list), meta_df

    def save(self, model_path: str, scaler_path: str):
        """Saves model weights and scaler."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved LSTM model to {model_path} and Scaler to {scaler_path}")

    def load(self, model_path: str, scaler_path: str):
        """Loads model weights and scaler."""
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Loaded LSTM model from {model_path}")


def lstm_fold_trainer(
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, LSTMStockTrainer]:
    """
    Callback function for Walk-Forward cross validation with LSTM.
    """
    trainer = LSTMStockTrainer(
        input_dim=len(feature_cols),
        seq_length=15,
        hidden_dim=32,
        num_layers=1,
        epochs=30,
        early_stopping_patience=8,
        batch_size=64
    )
    trainer.fit(train_fold, val_fold, feature_cols)
    probs, meta_df = trainer.predict_proba(val_fold, feature_cols)
    return probs, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM/GRU Sequence Model (Track B)")
    parser.add_argument("--data-file", default="data/processed/modeling_data_advanced.csv")
    parser.add_argument("--model-output", default="reports/models/lstm_model.pt")
    parser.add_argument("--scaler-output", default="reports/models/lstm_scaler.joblib")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        logger.error(f"Dataset {args.data_file} not found. Generate it first!")
        sys.exit(1)
        
    df = pd.read_csv(args.data_file)
    feature_cols = get_feature_columns()
    
    from src.data.preparation import split_time_series
    train_df, val_df, test_df = split_time_series(df)
    
    # Train full LSTM on Train set, early stopping on Val set
    logger.info("Training full LSTM model on training set...")
    trainer = LSTMStockTrainer(
        input_dim=len(feature_cols),
        seq_length=15,
        hidden_dim=64,
        num_layers=2,
        epochs=50,
        early_stopping_patience=10
    )
    trainer.fit(train_df, val_df, feature_cols)
    
    val_probs, val_meta = trainer.predict_proba(val_df, feature_cols)
    val_metrics = evaluate_predictions(val_meta["target"].values, val_probs)
    
    print("\n--- LSTM Validation Set (2024) Performance ---")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    trainer.save(args.model_output, args.scaler_output)
