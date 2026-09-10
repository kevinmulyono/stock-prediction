# 🌐 BRIEFING & ACTION ITEMS UNTUK ALI (TRACK C)
**Proyek:** AI Stock Movement Prediction (IDX Bluechips)  
**Role Ali:** Track C — Fullstack API & Interactive Trading Dashboard  
**Update Terakhir:** 8 September 2026 (Data & Model Layer 100% Ready)

---

## 📢 Ringkasan Situasi Terkini
Halo Ali! 👋  
Seluruh pipeline **Data Engineering (Track A - Kevin)** dan **Machine Learning & Deep Learning (Track B - Daf & Kevin)** sudah **100% SELESAI, DIUJI, dan TERISI LENGKAP** ke database SQLite lokal dan file laporan CSV.

Sekarang adalah **giliran kamu (Track C)** untuk mengubah seluruh hasil prediksi dan analisis data ini menjadi **aplikasi siap pakai untuk pengguna/trader**, yang terdiri dari:
1. **REST API Backend** menggunakan **FastAPI** (`src/api/`).
2. **Interactive Web Dashboard** untuk visualisasi sinyal saham, chart harga, dan performa model AI.

---

## 📦 Apa Saja yang Sudah Siap & Bisa Kamu Konsumsi?

### 1. Database SQLite (`data/stock_prediction.db`)
Database sudah aktif dan berisi 3 tabel relasional (ORM SQLAlchemy di [`src/database/models.py`](file:///d:/project%20mandiri/stock%20prediction/src/database/models.py)):

| Nama Tabel | Jumlah Data | Deskripsi Kolom Penting |
|---|:---:|---|
| `stocks` | 5 emiten | `ticker`, `company_name`, `sector` (BBCA, BBRI, BMRI, TLKM, ASII) |
| `market_data` | 10.393 baris | `date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume` |
| `predictions` | **11.490 baris** | `date`, `ticker`, `model`, `probability_up`, `prediction`, `signal` |

> 💡 **Info Model:** Tabel `predictions` memiliki data prediksi lengkap dari 6 model: `logistic_regression`, `random_forest`, `xgboost`, `lightgbm`, `lstm`, dan `ensemble`. Sinyal bernilai: `"BUY"`, `"HOLD"`, atau `"SELL"`.

### 2. File Laporan Backtest & Perbandingan Model
- [`reports/trading_backtest_all.csv`](file:///d:/project%20mandiri/stock%20prediction/reports/trading_backtest_all.csv): Metrik simulasi trading (Total Return, Sharpe Ratio, Max Drawdown, Win Rate, Total Trades) semua model vs Buy & Hold Benchmark.
- [`reports/model_comparison_all.csv`](file:///d:/project%20mandiri/stock%20prediction/reports/model_comparison_all.csv): Metrik statistik (ROC-AUC, Accuracy, Precision, Recall, F1, Brier Score, Log Loss).

---

## 🚀 Langkah Pertama: Setup Lingkungan Kerja
Tarik kode terbaru dan aktifkan virtual environment:
```bash
git pull origin main
.\venv\Scripts\activate          # Windows
pip install fastapi uvicorn pydantic
```

---

## 🛠️ Panduan Pengerjaan Track C

### Bagian 1 — Backend REST API (`src/api/`)

Buat modul API di direktori `src/api/`:
```
src/api/
├── __init__.py
├── main.py             # Entry point FastAPI & CORS middleware
├── schemas.py          # Pydantic schemas (request/response DTO)
└── routes/
    ├── __init__.py
    ├── stocks.py       # Endpoints emiten & data harga
    ├── predictions.py  # Endpoints sinyal trading & probabilitas
    └── metrics.py      # Endpoints backtest & evaluasi model
```

#### Endpoint yang Wajib Disediakan:

1. `GET /api/stocks`
   - Mengembalikan daftar 5 emiten saham beserta sektornya.
2. `GET /api/stocks/{ticker}/history?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`
   - Mengambil data OHLCV historis dari tabel `market_data` untuk charting.
3. `GET /api/predictions/latest?model=ensemble`
   - Mengambil prediksi paling mutakhir per emiten saham (probabilitas $P(\text{Up})$, sinyal rekomendasi `BUY`/`HOLD`/`SELL`, dan tanggal prediksi).
4. `GET /api/predictions/history?ticker={ticker}&model=ensemble`
   - Mengambil histori prediksi untuk memplot sinyal BUY/SELL di atas candlestick chart.
5. `GET /api/backtest/summary`
   - Mengembalikan data JSON dari `reports/trading_backtest_all.csv` (leaderboard performa strategi trading).
6. `GET /api/models/comparison`
   - Mengembalikan data JSON dari `reports/model_comparison_all.csv`.

#### Cara Akses Database di FastAPI:
Gunakan helper `get_db_session` yang sudah disiapkan di `src.database.connection`:
```python
from fastapi import FastAPI, Depends
from src.database.connection import get_db_session
from src.database.models import Stock, Prediction, MarketData

app = FastAPI(title="IDX Stock Prediction & Trading Signals API")

@app.get("/api/stocks")
def get_stocks():
    with get_db_session() as session:
        stocks = session.query(Stock).all()
        return [{"ticker": s.ticker, "name": s.company_name, "sector": s.sector} for s in stocks]
```

---

### Bagian 2 — Interactive Frontend Dashboard (`frontend/`)

Bangun UI modern, responsif, dan bernuansa *FinTech* (boleh menggunakan React + Vite, Next.js, atau vanilla frontend).

#### 3 Komponen Utama yang Wajib Ada di UI:

1. **📊 Live Trading Signals & Recommendation Cards:**
   - Menampilkan 5 kartu saham (BBCA, BBRI, BMRI, TLKM, ASII).
   - Badge warna sinyal rekomendasi yang mencolok:
     - 🟢 **BUY** (misal jika Probabilitas Naik $\ge 53\%$)
     - 🟡 **HOLD** (jika Probabilitas Netral)
     - 🔴 **SELL** (jika Probabilitas Naik $\le 47\%$)
   - Confidence Gauge: Menampilkan nilai probabilitas model (contoh: `P(Up) = 62.5%`).

2. **📈 Interactive Price & Signals Candlestick Chart:**
   - Grafik interaktif (menggunakan TradingView *Lightweight Charts* atau *Chart.js*).
   - Menampilkan candlestick OHLCV saham yang dipilih.
   - **Tandai marker panah hijau (BUY) dan merah (SELL)** pada tanggal saat model mengeksekusi posisi trading.

3. **🏆 Model Performance & Backtest Leaderboard:**
   - Tabel perbandingan performa trading nyata (menampilkan Total Return, Sharpe Ratio, Max Drawdown).
   - Highlight pencapaian utama tim kita:
     - **Ensemble Model (Track B): +22.22% Return** (Sharpe 1.37, Max Drawdown hanya -7.86%)
     - **Buy & Hold Benchmark: -0.40% Return** (Max Drawdown parah -31.09%)
     - *Alpha / Excess Return:* **+22.62% di atas pasar!**

---

## 📞 Koordinasi dengan Tim
- **Kevin:** Jika butuh klarifikasi field database, query SQL, atau validasi struktur backtest.
- **Daf:** Koordinasikan skema output untuk *live daily prediction* agar API bisa langsung menyajikan sinyal harian terbaru secara otomatis.
