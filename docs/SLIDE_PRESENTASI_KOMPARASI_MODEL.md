# 📊 SLIDE PRESENTASI & MATERI KOMPARASI MODEL
**Proyek:** AI Stock Movement Prediction (IDX Bluechips: BBCA, BBRI, BMRI, TLKM, ASII)  
**Lead Presenter:** Kevin (Track A Lead / ML Modeler)  
**Kolaborator:** Daf (Track B - Advanced ML & DL), Ali (Track C - Fullstack API & Dashboard)  
**Dataset Split:** Train (2018–2023), Validation (2024), Test Set Steril (2025–2026, 1.950 baris)

---

## 🖥️ SLIDE 1: JUDUL & EXECUTIVE SUMMARY
### **AI-Driven Stock Movement Prediction & Systematic Trading on IDX Bluechips**
*Optimal Model Architecture: Hybrid Weighted Ensemble (LightGBM + PyTorch LSTM)*

#### **Highlight Pencapaian Kunci:**
* **Pasar (Buy & Hold Benchmark):** Return **-0.40%**, Sharpe **0.13**, Max Drawdown **-31.09%**.
* **Strategi AI Terbaik (Ensemble Track B):** Return **+22.22%**, Sharpe **1.37**, Max Drawdown **-7.86%**, Win Rate **54.46%**.
* **Alpha / Excess Return Terhadap Pasar:** **+22.62%** (Bersih setelah memperhitungkan fee transaksi bursa IDX 0.15% per trade).

---

## 🖥️ SLIDE 2: DESAIN EKSPERIMEN & INTEGRITAS DATA (ANTI-LEAKAGE)
*Filosofi: Tanpa Lookahead Bias, Tanpa Data Snooping*

1. **Strict 3-Way Chronological Split:**
   * **Train Set (2018–2023 | 7.153 baris):** Fitting model & standardisasi fitur scaler.
   * **Validation Set (2024 | 1.185 baris):** Kalibrasi probabilitas (Isotonic) & penentuan threshold sinyal (BUY $\ge 0.53$, SELL $\le 0.47$).
   * **Test Set (2025–2026 | 1.950 baris):** 100% steril dan *out-of-sample*, hanya dibuka saat evaluasi final.
2. **29 Fitur Teknikal Backward-Looking:**
   * Momentum (RSI, MACD, Stochastic), Tren (EMA Cross, Bollinger Bands), Lagged Historical Returns (1d s.d. 10d), dan Rolling Volatility.
   * Semua fitur menggunakan `shift(+n)` — tidak pernah mengintip masa depan.
3. **Multi-Ticker Position & Portfolio Compounding:**
   * Posisi dan fee bursa (0.15%) diisolasi secara independen per emiten.
   * Compounding portofolio dihitung secara *equal-weight* harian (bebas dari distorsi akumulasi intra-day multi-saham).

---

## 🖥️ SLIDE 3: TABEL KOMPARASI MENYELURUH (LEADERBOARD)
*Evaluasi Out-of-Sample pada Test Set (2025+ | 1.950 Baris)*

### A. Metrik Simulasi Trading Riil (Fee Drag 0.15%)
| Peringkat | Model | Total Return (%) | Excess Return (%) | Sharpe Ratio | Max Drawdown (%) | Win Rate (%) | Total Trades | Kategori |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Ensemble (Track B)** | **+22.22%** | **+22.62%** | **1.37** | **-7.86%** | **54.46%** | 180 | **Hybrid ML+DL** |
| 🥈 | **LSTM / GRU** | **+20.56%** | **+20.96%** | **1.22** | **-8.55%** | **52.74%** | 200 | Deep Learning |
| 🥉 | **Logistic Regression** | **+18.25%** | **+18.65%** | **1.27** | **-6.40%** | **55.76%** | 94 | Klasikal (L2) |
| 4 | **XGBoost** | **+14.70%** | **+15.10%** | **1.37** | **-5.01%** | **51.40%** | 122 | Gradient Boosted Trees |
| 5 | **Random Forest** | **+11.89%** | **+12.29%** | **1.14** | **-5.20%** | **48.80%** | 116 | Bagging Trees |
| 6 | **LightGBM** | **+1.63%** | **+2.03%** | **0.26** | **-3.72%** | **61.54%** | 72 | Gradient Boosted Trees |
| 🔴 | *Buy & Hold (Benchmark)* | *-0.40%* | *0.00%* | *0.13* | *-31.09%* | *0.00%* | 0 | *Pasif / Pasar* |

### B. Metrik Prediktif Statistik
| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score | Brier Score | Log Loss |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **0.5538** | **57.18%** | 55.21% | 18.38% | 0.2758 | 0.2447 | 0.6826 |
| **Random Forest** | 0.5524 | 56.77% | 53.11% | 21.73% | 0.3085 | 0.2452 | 0.6837 |
| **Ensemble (Track B)** | 0.5468 | 55.74% | 50.36% | 16.18% | 0.2450 | 0.2452 | 0.6836 |
| **LSTM / GRU** | 0.5453 | 55.85% | 50.64% | 18.27% | 0.2685 | 0.2454 | 0.6840 |
| **LightGBM** | 0.5393 | 56.31% | 53.48% | 11.56% | 0.1901 | 0.2454 | 0.6839 |
| **XGBoost** | 0.5278 | 56.46% | 52.31% | 20.92% | 0.2989 | 0.2464 | 0.6860 |

---

## 🖥️ SLIDE 4: ANALISIS MENDALAM — MENGAPA TIAP MODEL BERPERILAKU BERBEDA?

### 1. Rahasia Performa Tinggi Logistic Regression (Track A - Kevin):
* **Linear Regularization vs Market Noise:** Regularisasi $L_2$ (Ridge) mencegah model menangkap *overfitted noise* intra-day.
* **Turnover Sangat Efisien (Hanya 94 Trades):** Model ini sangat selektif. Saat sinyal tidak meyakinkan, model diam di posisi kas (HOLD/SELL). Hasilnya: *Fee drag* minimal dan volatilitas rendah $\to$ **Sharpe Ratio 1.27** & **Win Rate 55.76%**.

### 2. Mengapa Weighted Ensemble (Track B) Menjadi Juara Tertinggi (+22.22% Return)?
* **Sinergi Dua Paradigma Berbeda:**
  * **LightGBM (15% Weight):** Memetakan interaksi diskrit tabular antar 29 indikator teknikal.
  * **PyTorch LSTM (85% Weight):** Membaca dependensi sekuensial tren sepanjang 15 hari bursa (*sliding window*).
* **Decorrelation of Errors:** False signals dari LightGBM diredam oleh LSTM, dan sebaliknya. Sinyal BUY hanya aktif ketika kedua model sepakat pada probabilitas $P(\text{Up}) \ge 0.53$.

---

## 🖥️ SLIDE 5: MITIGASI RISIKO & ANALISIS DRAWDOWN
*Perlindungan Modal di Tengah Koreksi Pasar IDX*

* **Kelemahan Buy & Hold:** Saat 5 saham bluechip mengalami tekanan jual masif di tahun 2025–2026, investor pasif menderita penurunan modal drastis (**Max Drawdown -31.09%**).
* **Keunggulan Strategi AI:**
  * Algoritma mendeteksi pembalikan tren dan segera beralih ke posisi **CASH (SELL)**.
  * Max Drawdown Ensemble tertahan hanya di **-7.86%** (reduksi risiko hingga 75% dibanding pasar!).
  * Menghasilkan kurva ekuitas yang bertumbuh stabil (*smooth equity curve*).

---

## 🖥️ SLIDE 6: KESIMPULAN & REKOMENDASI UNTUK DEPLOYMENT (TRACK C)

### 📌 Rekomendasi Utama:
1. **Model Produksi Terpilih:** **Weighted Ensemble (15% LightGBM + 85% LSTM)** direkomendasikan sebagai *core engine* sinyal trading harian.
2. **Model Alternatif Berbiaya Rendah:** **Logistic Regression** direkomendasikan jika infrastruktur membutuhkan *ultra-fast low-latency inference* dengan konsumsi memori minimal.
3. **Status Handover ke Ali (Track C):**
   * SQLite Database (`stock_prediction.db`) telah terisi penuh 11.490 rekam prediksi historis.
   * Modul live daily scoring (`src/models/predict_daily.py`) sudah siap diintegrasikan ke endpoint FastAPI (`GET /api/predictions/latest`).
