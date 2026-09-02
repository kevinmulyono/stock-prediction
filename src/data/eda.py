import json
import logging
import os
import sys
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def calculate_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily returns, log returns, rolling volatility, and volume features.
    
    CRITICAL TIME-SERIES RULE:
    Returns are strictly backward-looking: (P[t] - P[t-1]) / P[t-1]
    No forward-looking indicators are computed here.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    
    # 1. Simple Daily Return & Log Return on Adjusted Close
    df["simple_return"] = df.groupby("ticker")["adj_close"].pct_change()
    df["log_return"] = df.groupby("ticker")["adj_close"].apply(lambda s: np.log(s / s.shift(1))).reset_index(level=0, drop=True)
    
    # 2. Cumulative Return
    df["cumulative_return"] = df.groupby("ticker")["simple_return"].apply(lambda s: (1 + s.fillna(0)).cumprod() - 1).reset_index(level=0, drop=True)
    
    # 3. Rolling Annualized Volatility (252 trading days/year)
    # 20-day window (~1 trading month) and 60-day window (~1 trading quarter)
    df["volatility_20d"] = df.groupby("ticker")["simple_return"].transform(
        lambda s: s.rolling(window=20, min_periods=20).std() * np.sqrt(252)
    )
    df["volatility_60d"] = df.groupby("ticker")["simple_return"].transform(
        lambda s: s.rolling(window=60, min_periods=60).std() * np.sqrt(252)
    )
    
    # 4. Volume Moving Average & Volume Spike Detection (> 2.5 std over 20-day mean)
    df["vol_ma20"] = df.groupby("ticker")["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["vol_std20"] = df.groupby("ticker")["volume"].transform(lambda s: s.rolling(20, min_periods=20).std())
    df["volume_spike"] = (df["volume"] > (df["vol_ma20"] + 2.5 * df["vol_std20"])).astype(int)
    
    return df


def generate_eda_figures(df: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    """
    Generates and saves visual figures for Track A EDA.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Generating EDA visual artifacts...")
    
    tickers = df["ticker"].unique()
    
    # --- 1. Cumulative Returns Comparison ---
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        sub = df[df["ticker"] == ticker]
        plt.plot(sub["date"], sub["cumulative_return"] * 100, label=ticker, linewidth=1.8)
    plt.title("Cumulative Returns Comparison (2018–2026)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Cumulative Return (%)", fontsize=11)
    plt.axhline(0, color="grey", linestyle="--", alpha=0.7)
    plt.legend(title="Ticker", loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_cumulative_returns.png"), dpi=300)
    plt.close()
    
    # --- 2. Return Distribution & Fat-Tails (KDE vs Normal) ---
    fig, axes = plt.subplots(len(tickers), 1, figsize=(12, 3 * len(tickers)), sharex=True)
    if len(tickers) == 1:
        axes = [axes]
    for ax, ticker in zip(axes, tickers):
        sub = df[df["ticker"] == ticker]["simple_return"].dropna()
        sns.histplot(sub, kde=True, ax=ax, stat="density", color="royalblue", bins=80, alpha=0.4, label="Actual Return KDE")
        
        # Plot theoretical normal distribution for comparison
        mu, sigma = sub.mean(), sub.std()
        x = np.linspace(sub.min(), sub.max(), 200)
        p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, p, 'r--', linewidth=1.5, label="Normal Dist Fit")
        
        skew = sub.skew()
        kurt = sub.kurtosis()
        ax.set_title(f"{ticker} Daily Return Distribution (Skew: {skew:.2f}, Excess Kurtosis: {kurt:.2f})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
    plt.xlabel("Daily Return")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_return_distributions.png"), dpi=300)
    plt.close()
    
    # --- 3. Rolling Volatility Regimes (20d & 60d) ---
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        sub = df[df["ticker"] == ticker]
        plt.plot(sub["date"], sub["volatility_20d"] * 100, label=f"{ticker} (20d)", alpha=0.85, linewidth=1.2)
    plt.title("Rolling 20-Day Annualized Volatility Regimes (%)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Annualized Volatility (%)", fontsize=11)
    plt.legend(title="Ticker", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_rolling_volatility.png"), dpi=300)
    plt.close()
    
    # --- 4. Cross-Asset Return Correlation Heatmap ---
    pivot_returns = df.pivot(index="date", columns="ticker", values="simple_return").dropna()
    corr_matrix = pivot_returns.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation"}
    )
    plt.title("Cross-Asset Daily Return Correlation Heatmap (2018–2026)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # --- 5. Volume Trends & Spike Detection ---
    fig, axes = plt.subplots(len(tickers), 1, figsize=(14, 3 * len(tickers)), sharex=True)
    if len(tickers) == 1:
        axes = [axes]
    for ax, ticker in zip(axes, tickers):
        sub = df[df["ticker"] == ticker]
        ax.plot(sub["date"], sub["volume"] / 1e6, color="slategray", alpha=0.6, label="Volume (Millions)")
        ax.plot(sub["date"], sub["vol_ma20"] / 1e6, color="navy", linewidth=1.5, label="20D MA")
        
        spikes = sub[sub["volume_spike"] == 1]
        ax.scatter(spikes["date"], spikes["volume"] / 1e6, color="crimson", s=25, zorder=5, label="Volume Spike (>2.5σ)")
        
        ax.set_title(f"{ticker} Trading Volume & Spikes", fontsize=11, fontweight="bold")
        ax.set_ylabel("Vol (M)")
        ax.legend(loc="upper right")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_volume_spikes.png"), dpi=300)
    plt.close()
    
    logger.info(f"All EDA figures successfully saved to {output_dir}")


def compute_statistical_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Computes key descriptive statistics per ticker:
    - Annualized Return, Annualized Volatility, Sharpe Ratio (rf=0)
    - Skewness, Excess Kurtosis
    - Max Drawdown
    - Total Volume Spikes
    """
    summary = []
    
    for ticker, sub in df.groupby("ticker"):
        returns = sub["simple_return"].dropna()
        n_days = len(returns)
        
        # Annualized metrics
        mean_daily_ret = returns.mean()
        ann_return = mean_daily_ret * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        
        # Distribution shape
        skew = returns.skew()
        kurt = returns.kurtosis()  # Excess kurtosis (Fisher)
        
        # Max Drawdown calculation
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min()
        
        # Up-day vs Down-day count
        up_days = (returns > 0).sum()
        down_days = (returns < 0).sum()
        flat_days = (returns == 0).sum()
        up_ratio = up_days / (up_days + down_days) if (up_days + down_days) > 0 else 0.5
        
        summary.append({
            "ticker": ticker,
            "observations": n_days,
            "ann_return_pct": round(ann_return * 100, 2),
            "ann_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio_zero_rf": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "skewness": round(skew, 3),
            "excess_kurtosis": round(kurt, 3),
            "up_days_ratio": round(up_ratio, 3),
            "volume_spikes_count": int(sub["volume_spike"].sum())
        })
        
    summary_df = pd.DataFrame(summary)
    summary_dict = summary_df.to_dict(orient="records")
    return summary_df, summary_dict


def run_eda_pipeline(
    data_path: str = "data/processed/combined_market_data.csv",
    report_dir: str = "reports"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main driver for Phase 3 EDA.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Please run Phase 2 first.")
        
    df = pd.read_csv(data_path)
    df_with_metrics = calculate_financial_metrics(df)
    
    os.makedirs(report_dir, exist_ok=True)
    figures_dir = os.path.join(report_dir, "figures")
    generate_eda_figures(df_with_metrics, figures_dir)
    
    stats_df, stats_dict = compute_statistical_summary(df_with_metrics)
    
    # Save statistical artifacts
    stats_csv_path = os.path.join(report_dir, "eda_statistics.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    
    stats_json_path = os.path.join(report_dir, "eda_summary.json")
    with open(stats_json_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
        
    logger.info(f"EDA statistical summary saved to {stats_csv_path} and {stats_json_path}")
    return df_with_metrics, stats_df


if __name__ == "__main__":
    logger.info("=== Starting Exploratory Data Analysis (EDA) Pipeline (Track A) ===")
    _, stats_df = run_eda_pipeline()
    print("\n=== Track A: Stock Market Statistical Summary (2018–2026) ===")
    print(stats_df.to_string(index=False))
