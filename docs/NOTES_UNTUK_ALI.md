# 🌐 DOKUMENTASI LENGKAP & PANDUAN PENGEMBANGAN UNTUK ALI (TRACK C)
**Proyek:** AI Stock Movement Prediction & Trading Dashboard (IDX Bluechips)  
**Role Ali:** Track C — Fullstack API & Interactive Web Dashboard  
**Status Pipeline Data & Model:** 100% READY, TESTED & INGESTED (Track A - Kevin & Track B - Daf)  
**Target Output:** FastAPI Backend (`src/api/`) + Interactive Web Dashboard (`frontend/`)

---

## 📌 DAFTAR ISI
1. [Ringkasan Peran & Tanggung Jawab Ali](#1-ringkasan-peran--tanggung-jawab-ali)
2. [Peta Lengkap Lokasi & Deskripsi Data yang Dibutuhkan](#2-peta-lengkap-lokasi--deskripsi-data-yang-dibutuhkan)
   - [A. Database SQLite (`data/stock_prediction.db`)](#a-database-sqlite-datastock_predictiondb)
   - [B. Laporan Backtest Finansial (`reports/trading_backtest_all.csv`)](#b-laporan-backtest-finansial-reportstrading_backtest_allcsv)
   - [C. Laporan Evaluasi Statistik Model (`reports/model_comparison_all.csv`)](#c-laporan-evaluasi-statistik-model-reportsmodel_comparison_allcsv)
   - [D. Metadata Model Registry (`reports/models/model_registry.json`)](#d-metadata-model-registry-reportsmodelsmodel_registryjson)
   - [E. Pipeline Live Daily Prediction (`src/models/predict_daily.py`)](#e-pipeline-live-daily-prediction-srcmodelspredict_dailypy)
3. [Spesifikasi REST API FastAPI (`src/api/`)](#3-spesifikasi-rest-api-fastapi-srcapi)
4. [Blueprint Desain Frontend Web Dashboard (`frontend/`)](#4-blueprint-desain-frontend-web-dashboard-frontend)
5. [Panduan Langkah Demi Langkah (Step-by-Step Execution)](#5-panduan-langkah-demi-langkah-step-by-step-execution)

---

## 1. Ringkasan Peran & Tanggung Jawab Ali
Halo Ali! 👋  
Tim kita telah menyelesaikan dua fondasi utama:
- **Track A (Kevin):** Data Pipeline, Pembersihan Data, Database Schema, Model Klasikal (Logistic Regression, Random Forest, XGBoost), serta Backtesting Terpadu.
- **Track B (Daf):** 29 Indikator Teknikal Lanjutan, Model Deep Learning LSTM/GRU, LightGBM, Weighted Ensemble, dan Live Daily Scoring.

**Tugas Ali (Track C):**
Membangun antarmuka pengguna interaktif (*FinTech Web Dashboard*) dan backend API penghubungnya, sehingga trader/investor dapat melihat:
1. Rekomendasi sinyal harian (**BUY / HOLD / SELL**) untuk 5 saham IDX Bluechip (`BBCA`, `BBRI`, `BMRI`, `TLKM`, `ASII`).
2. Grafik Candlestick harga interaktif yang dilengkapi penanda titik eksekusi sinyal beli & jual.
3. *Leaderboard* pembuktian performa: bagaimana model Ensemble kita mengalahkan pasar (**+22.22% vs -0.40% Benchmark**).

---

## 2. Peta Lengkap Lokasi & Deskripsi Data yang Dibutuhkan

Berikut adalah rincian katalog data yang sudah siap di repositori:

```
stock-prediction/
├── data/
│   └── stock_prediction.db               ← DATABASE UTAMA (SQLite)
├── reports/
│   ├── trading_backtest_all.csv          ← DATA LEADERBOARD BACKTEST (ROI, Sharpe, MDD)
│   ├── model_comparison_all.csv          ← DATA AKURASI & ROC-AUC STATISTIK
│   └── models/
│       └── model_registry.json           ← METADATA & BOBOT ENSEMBLE
└── src/
    ├── database/
    │   ├── models.py                     ← SQLAlchemy ORM Models
    │   └── connection.py                 ← get_db_session Helper
    └── models/
        └── predict_daily.py              ← Live Scoring Pipeline (T+1)
```

---

### A. Database SQLite (`data/stock_prediction.db`)
Database SQLite ini bersifat *self-contained* lokal, tidak memerlukan instalasi server database eksternal.

#### Cara Koneksi di Kode Python:
```python
from src.database.connection import get_db_session
from src.database.models import Stock, MarketData, Prediction

with get_db_session() as session:
    # Query SQLAlchemy di sini
    pass
```

Database ini memiliki **3 tabel utama**:

#### 1. Tabel `stocks` (Informasi Profil Emiten)
* **Deskripsi:** Berisi master data 5 saham likuid IDX yang menjadi universe analisis.
* **Jumlah Record:** 5 baris
* **Skema Kolom:**
  | Nama Kolom | Tipe Data | Deskripsi | Contoh Data |
  |---|---|---|---|
  | `stock_id` | Integer (PK) | Auto-increment ID | `1` |
  | `ticker` | String(20) | Kode saham IDX (Unique) | `'BBCA.JK'` |
  | `company_name`| String(255) | Nama resmi perusahaan | `'Bank Central Asia Tbk'` |
  | `sector` | String(100) | Sektor industri | `'Financials'` |
  | `created_at` | DateTime | Timestamp dibuat | `2026-09-02 08:00:00` |
* **Daftar 5 Saham:**
  - `BBCA.JK` (Bank Central Asia - Financials)
  - `BBRI.JK` (Bank Rakyat Indonesia - Financials)
  - `BMRI.JK` (Bank Mandiri - Financials)
  - `TLKM.JK` (Telkom Indonesia - Telecommunication/Infrastructure)
  - `ASII.JK` (Astra International - Industrials/Conglomerate)
* **Kegunaan di Web:** Dropdown pemilih saham, kartu navigasi saham di sidebar/header.

---

#### 2. Tabel `market_data` (Histori Harga Saham OHLCV)
* **Deskripsi:** Rekam historis harga harian lengkap untuk pembuatan grafik candlestick dan volume.
* **Rentang Waktu:** 2018-01-01 s.d. 2026-08-31
* **Jumlah Record:** 10.393 baris (~2.080 baris per saham)
* **Skema Kolom:**
  | Nama Kolom | Tipe Data | Deskripsi | Contoh Data |
  |---|---|---|---|
  | `id` | Integer (PK) | Auto-increment ID | `1024` |
  | `date` | Date | Tanggal perdagangan bursa | `'2025-06-15'` |
  | `ticker` | String(20) (FK) | Relasi ke `stocks.ticker` | `'BBCA.JK'` |
  | `open` | Numeric(14,4) | Harga pembukaan hari itu | `9850.0000` |
  | `high` | Numeric(14,4) | Harga tertinggi hari itu | `10000.0000` |
  | `low` | Numeric(14,4) | Harga terendah hari itu | `9800.0000` |
  | `close` | Numeric(14,4) | Harga penutupan hari itu | `9975.0000` |
  | `adj_close` | Numeric(14,4) | Harga penutupan adjusted | `9975.0000` |
  | `volume` | BigInteger | Volume lembar saham ditransaksikan | `85420300` |
  | `adj_factor` | Numeric(10,6)| Faktor penyesuaian split/dividen | `1.000000` |
* **Kegunaan di Web:** Dikonsumsi oleh library grafik seperti **TradingView Lightweight Charts** atau **Chart.js** untuk menampilkan grafik Candlestick interaktif + Volume bar di bawahnya.

---

#### 3. Tabel `predictions` (Prediksi Probabilitas & Sinyal Trading AI)
* **Deskripsi:** Rekam hasil inferensi probabilitas naik/turun dan sinyal trading dari seluruh model AI pada Test Set (2025–2026) dan live harian.
* **Jumlah Record:** **11.490 baris** (1.950 hari bursa $\times$ 6 model)
* **Skema Kolom:**
  | Nama Kolom | Tipe Data | Deskripsi | Contoh Data |
  |---|---|---|---|
  | `id` | Integer (PK) | Auto-increment ID | `501` |
  | `date` | Date | Tanggal prediksi berlaku ($T+1$) | `'2025-08-20'` |
  | `ticker` | String(20) (FK) | Kode saham | `'BMRI.JK'` |
  | `model` | String(50) | Nama model yang memprediksi | `'ensemble'` |
  | `probability_up`| Float | Probabilitas harga naik ($P(\text{Up}) \in [0, 1]$) | `0.64250` |
  | `prediction` | SmallInteger | Klasifikasi biner (`1` = naik, `0` = turun/flat)| `1` |
  | `signal` | String(10) | Rekomendasi tindakan bursa | `'BUY'` (`'BUY'`, `'HOLD'`, `'SELL'`) |
  | `created_at` | DateTime | Waktu komputasi prediksi | `2026-09-08 18:00:00` |
* **Daftar Model yang Tersedia di Kolom `model`:**
  - `ensemble` *(Model Utama - Rekomendasi Default)*
  - `lstm` *(PyTorch Deep Learning)*
  - `lightgbm` *(Gradient Boosting)*
  - `logistic_regression` *(Klasikal Track A)*
  - `xgboost` *(Klasikal Track A)*
  - `random_forest` *(Klasikal Track A)*
* **Aturan Sinyal (Trading Rules):**
  - $P(\text{Up}) \ge 0.53 \implies$ **`BUY`** 🟢 (Buka Posisi Long)
  - $P(\text{Up}) \le 0.47 \implies$ **`SELL`** 🔴 (Tutup Posisi / Amankan ke Kas)
  - $0.47 < P(\text{Up}) < 0.53 \implies$ **`HOLD`** 🟡 (Pertahankan Posisi Sebelumnya)
* **Kegunaan di Web:**
  1. Menampilkan **Badge Rekomendasi Hari Ini** (BUY / HOLD / SELL) di kartu saham.
  2. Menampilkan **Gauge Meter Keyakinan Model** ($P(\text{Up})$ dalam %).
  3. Menampilkan **Marker Titik Sinyal** di atas grafik candlestick harga (Panah hijau untuk BUY, panah merah untuk SELL).

---

### B. Laporan Backtest Finansial (`reports/trading_backtest_all.csv`)
* **Lokasi File:** [`reports/trading_backtest_all.csv`](file:///d:/project%20mandiri/stock%20prediction/reports/trading_backtest_all.csv)
* **Format:** CSV UTF-8
* **Deskripsi:** Hasil simulasi perdagangan harian realistis pada periode out-of-sample (Test Set: 2025–2026) dengan memperhitungkan biaya transaksi bursa IDX (0.15% per trade).
* **Isi Data Persis:**
  ```csv
  Model,Total Return (%),Excess Return (%),Sharpe Ratio,Max Drawdown (%),Win Rate (%),Trades
  Buy & Hold (Benchmark),-0.3994,0.0000,0.1300,-31.0873,0.0000,0
  Ensemble (Track B),22.2165,22.6159,1.3662,-7.8615,54.4601,180
  LSTM/GRU,20.5577,20.9571,1.2189,-8.5482,52.7426,200
  Logistic Regression,18.2488,18.6482,1.2672,-6.4035,55.7576,94
  XGBoost,14.7029,15.1023,1.3734,-5.0061,51.3966,122
  Random Forest,11.8865,12.2859,1.1400,-5.2029,48.8038,116
  LightGBM,1.6349,2.0343,0.2585,-3.7216,61.5385,72
  ```
* **Kegunaan di Web:** Ditampilkan pada tabel **Leaderboard Performa Strategi Trading**, membuktikan bahwa Ensemble memberikan keuntungan **+22.22%** sementara pasar pasif merugi **-0.40%** dan menderita drawdown parah **-31.09%**.

---

### C. Laporan Evaluasi Statistik Model (`reports/model_comparison_all.csv`)
* **Lokasi File:** [`reports/model_comparison_all.csv`](file:///d:/project%20mandiri/stock%20prediction/reports/model_comparison_all.csv)
* **Format:** CSV UTF-8
* **Deskripsi:** Metrik evaluasi Machine Learning standar untuk menguji ketajaman probabilitas model.
* **Isi Kolom:**
  - `model`: Nama model
  - `roc_auc`: Area Under ROC Curve (kemampuan membedakan saham naik vs turun)
  - `accuracy`: Persentase ketepatan tebakan arah
  - `precision` & `recall`: Rasio ketepatan sinyal positif
  - `brier_score`: Rata-rata error kalibrasi probabilitas (semakin kecil mendekati 0 semakin baik)
* **Kegunaan di Web:** Tab "Model Evaluation" atau radar chart perbandingan metrik teknis.

---

### D. Metadata Model Registry (`reports/models/model_registry.json`)
* **Lokasi File:** [`reports/models/model_registry.json`](file:///d:/project%20mandiri/stock%20prediction/reports/models/model_registry.json)
* **Format:** JSON
* **Deskripsi:** Berisi konfigurasi produksi model aktif:
  ```json
  {
    "active_model": "ensemble",
    "ensemble_weights": {
      "lightgbm": 0.15,
      "lstm": 0.85
    },
    "signal_thresholds": {
      "buy_threshold": 0.53,
      "sell_threshold": 0.47
    },
    "test_performance": {
      "ensemble_return_pct": 22.22,
      "benchmark_return_pct": -0.40,
      "sharpe_ratio": 1.37,
      "max_drawdown_pct": -7.86
    }
  }
  ```
* **Kegunaan di Web:** Menampilkan *Model Info Modal* atau *Hyperparameter Badge* di dashboard.

---

### E. Pipeline Live Daily Prediction (`src/models/predict_daily.py`)
* **Lokasi File:** [`src/models/predict_daily.py`](file:///d:/project%20mandiri/stock%20prediction/src/models/predict_daily.py)
* **Deskripsi:** Modul otomasi untuk mengambil data bursa terkini, menghitung 29 fitur teknikal, mengeksekusi inferensi model Ensemble, dan otomatis meng-insert sinyal $T+1$ (besok) ke tabel `predictions`.
* **Cara Eksekusi via Python di API Ali:**
  ```python
  from src.models.predict_daily import run_daily_prediction
  
  # Menghasilkan prediksi besok dan otomatis menyimpan ke database
  daily_results = run_daily_prediction(model_type="ensemble", persist_db=True)
  ```
* **Kegunaan di Web:** Tombol "🔄 Refresh Live Signals" di dashboard untuk men-trigger prediksi hari bursa berikutnya secara real-time.

---

## 3. Spesifikasi REST API FastAPI (`src/api/`)

Ali disarankan membuat struktur backend di folder `src/api/`:
```
src/api/
├── main.py             # FastAPI App, CORS, Exception Handlers
├── schemas.py          # Pydantic Response DTOs
└── routes/
    ├── stocks.py       # GET /api/stocks, GET /api/stocks/{ticker}/history
    ├── predictions.py  # GET /api/predictions/latest, GET /api/predictions/history
    └── analytics.py    # GET /api/backtest/summary, GET /api/models/comparison
```

### Daftar Endpoint yang Wajib Dibuat:

| Method | URL Endpoint | Parameter | Deskripsi & Return Value |
|---|---|---|---|
| `GET` | `/api/stocks` | - | Daftar 5 emiten (`ticker`, `company_name`, `sector`). |
| `GET` | `/api/stocks/{ticker}/history` | `timeframe` (`1M`, `3M`, `6M`, `1Y`, `ALL`) | Array data OHLCV (`date`, `open`, `high`, `low`, `close`, `volume`) untuk Candlestick chart. |
| `GET` | `/api/predictions/latest` | `model` (default: `'ensemble'`) | Prediksi terbaru 5 saham: harga terakhir, $P(\text{Up})$, sinyal (`BUY`/`HOLD`/`SELL`), dan tanggal berlaku. |
| `GET` | `/api/predictions/signals-overlay` | `ticker`, `model` | Array sinyal historis (`date`, `signal`, `probability_up`) untuk marker panah pada chart harga. |
| `GET` | `/api/backtest/summary` | - | Data JSON dari `trading_backtest_all.csv` (leaderboard ROI, Sharpe, Drawdown). |
| `GET` | `/api/models/comparison` | - | Data JSON dari `model_comparison_all.csv` (metrik statistik akurasi). |
| `POST` | `/api/predictions/trigger-daily` | - | Memanggil `run_daily_prediction()` untuk update sinyal $T+1$ terbaru. |

#### Contoh Implementasi Route Sinyal Terbaru (`src/api/routes/predictions.py`):
```python
from fastapi import APIRouter
from sqlalchemy import func
from src.database.connection import get_db_session
from src.database.models import Prediction, MarketData, Stock

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])

@router.get("/latest")
def get_latest_predictions(model: str = "ensemble"):
    with get_db_session() as session:
        # Cari tanggal prediksi paling akhir
        latest_date = session.query(func.max(Prediction.date)).filter(Prediction.model == model).scalar()
        
        records = (
            session.query(Prediction, Stock.company_name)
            .join(Stock, Prediction.ticker == Stock.ticker)
            .filter(Prediction.model == model, Prediction.date == latest_date)
            .all()
        )
        
        response = []
        for pred, company_name in records:
            response.append({
                "ticker": pred.ticker,
                "company_name": company_name,
                "target_date": str(pred.date),
                "model": pred.model,
                "probability_up": round(pred.probability_up * 100, 2),  # Misal: 64.25%
                "signal": pred.signal,                                  # "BUY", "HOLD", "SELL"
                "prediction_label": "NAIK" if pred.prediction == 1 else "TURUN/FLAT"
            })
        return {"latest_date": str(latest_date), "data": response}
```

---

## 4. Blueprint Desain Frontend Web Dashboard (`frontend/`)

Ali disarankan menggunakan **React + Vite** (atau Next.js) dengan TailwindCSS / Vanilla CSS bernuansa *Dark FinTech Mode* (latar belakang `#0f172a`, card `#1e293b`).

### 3 Komponen Utama Dashboard:

```
+---------------------------------------------------------------------------------------+
|  📊 IDX AI STOCK TRADER DASHBOARD              [🔄 Refresh Live] [Model: Ensemble ▾]  |
+---------------------------------------------------------------------------------------+
|  [BBCA.JK]             [BBRI.JK]             [BMRI.JK]         [TLKM.JK]    [ASII.JK] |
|  Rp 9.975              Rp 5.150              Rp 7.200          Rp 3.420     Rp 5.050  |
|  Badge: 🟢 BUY         Badge: 🟡 HOLD        Badge: 🟢 BUY     Badge: 🔴 SELL ...     |
|  Prob: 65.4%           Prob: 50.2%           Prob: 58.7%       Prob: 42.1%            |
+---------------------------------------------------------------------------------------+
|                                                                                       |
|   📈 INTERACTIVE CANDLESTICK CHART (BBCA.JK)                 [1M] [3M] [6M] [1Y] [ALL]|
|   +-------------------------------------------------------------------------------+   |
|   |          ▲ (BUY Sinyal Model)                                                 |   |
|   |         / \     /\                                                            |   |
|   |        /   \   /  \    ▼ (SELL Sinyal Model)                                  |   |
|   |   +---+     \-/    \--+                                                       |   |
|   +-------------------------------------------------------------------------------+   |
|   |   Volume Bar Chart: ▇ ▅ █ ▃ ▆ █ ▅ ▃ ▇                                         |   |
+---------------------------------------------------------------------------------------+
|  🏆 STRATEGY LEADERBOARD (Backtest 2025-2026 Out-of-Sample)                           |
|  Model                   Total Return     Excess Return   Sharpe Ratio   Max Drawdown |
|  🥇 Ensemble (Track B)   +22.22%          +22.62%         1.37           -7.86%       |
|  🥈 LSTM / GRU           +20.56%          +20.96%         1.22           -8.55%       |
|  🥉 Logistic Regression  +18.25%          +18.65%         1.27           -6.40%       |
|  🔴 Buy & Hold (Pasar)   -0.40%           0.00%           0.13           -31.09%      |
+---------------------------------------------------------------------------------------+
```

---

## 5. Panduan Langkah Demi Langkah (Step-by-Step Execution)

1. **Langkah 1: Tarik Kode Terbaru & Aktifkan Virtual Environment**
   ```bash
   git checkout main
   git pull origin main
   .\venv\Scripts\activate
   pip install fastapi uvicorn pydantic requests
   ```

2. **Langkah 2: Uji Baca Database & Data CSV**
   Jalankan verifikasi cepat di terminal Python untuk memastikan database terbaca:
   ```bash
   python -c "from src.database.connection import get_db_session; from src.database.models import Prediction; session=get_db_session(); print('Total Prediksi di DB:', session.query(Prediction).count()); session.close()"
   ```
   *Output harus menunjukkan:* `Total Prediksi di DB: 11490`

3. **Langkah 3: Bangun Backend API (`src/api/`)**
   - Buat `src/api/main.py`.
   - Jalankan server dev:
     ```bash
     uvicorn src.api.main:app --reload --port 8000
     ```
   - Buka dokumentasi Swagger interaktif di browser: `http://127.0.0.1:8000/docs`.

4. **Langkah 4: Bangun Frontend Dashboard (`frontend/`)**
   - Buat project frontend (misal: `npm create vite@latest frontend -- --template react`).
   - Pasang library charting: `npm install lightweight-charts axios lucide-react`.
   - Sambungkan API call `axios.get("http://localhost:8000/api/...")` ke komponen UI.

5. **Langkah 5: Koordinasi Tim**
   - Jika butuh custom query SQL / filter tambahan: hubungi **Kevin**.
   - Jika butuh penyesuaian parameter live inference / model registry: hubungi **Daf**.
