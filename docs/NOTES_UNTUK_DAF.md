# 📘 BRIEFING & ACTION ITEMS UNTUK DAF (TRACK B)
**Proyek:** AI Stock Movement Prediction (IDX Bluechips)  
**Role Daf:** Track B — Advanced ML, Deep Learning & Production Inference  
**Update Terakhir:** 8 September 2026 (Pasca Code Review & Bug Fixes oleh Kevin)

---

## 📢 Ringkasan Situasi Terkini
Halo Daf! 👋  
Pengerjaan Track B kamu (Feature Engineering 29 indikator teknikal, LightGBM, PyTorch LSTM, Ensemble, dan database ingestion) sudah di-review secara mendalam. Semua fondasi yang kamu buat sangat solid! 

Kevin baru saja melakukan perbaikan atas **6 bug & konsistensi logika** di codebase agar hasil eksperimen kita valid dan siap dipresentasikan. 

### 🔧 Apa Saja yang Baru Diperbaiki oleh Kevin:
1. **Multi-Ticker Position Bleed di Backtesting (`src/models/ensemble_and_signals.py`):**
   - Sebelumnya, variabel `position` tercampur antar ticker saat loop berurutan, dan fungsi melakukan `.cumprod()` langsung pada dataframe multi-saham yang menyebabkan compounding harian 5x lipat.
   - **Solusi baru:** Posisi dan fee dihitung independen per ticker, lalu return portofolio harian dihitung secara *equal-weight* antar saham aktif.
2. **LSTM Sequence Warmup & Apple-to-Apple Benchmark (`src/models/lstm_model.py` & `src/models/compare_models.py`):**
   - Sebelumnya, LSTM memotong 14 hari pertama test set (70 baris) karena sliding window tidak memiliki riwayat sebelum tanggal 2 Jan 2025.
   - **Solusi baru:** Fungsi `lstm_trainer.predict_proba()` sekarang menerima `warmup_df` (mengambil 14 baris terakhir dari `val_df` Desember 2024 sebagai sequence history tanpa membocorkan target).
   - **Hasil:** Test set Track B kini utuh **1.950 baris** (persis sama dengan Track A) dan Buy & Hold benchmark kedua track identik 100% di **-0.40%**.
3. **Hasil Backtest Terkini (Test Set: 2025+):**
   - **Ensemble (Track B) menjadi model TERBAIK di seluruh proyek:**
     * **Total Return:** **+22.22%** (vs Buy & Hold: -0.40% → **Excess Return: +22.62%**)
     * **Sharpe Ratio:** **1.37** (Benchmark: 0.13)
     * **Max Drawdown:** **-7.86%** (Benchmark: -31.09%)
     * **Win Rate:** **54.46%** (180 total trades)
   - Disusul oleh **LSTM/GRU:** Return **+20.56%**, Sharpe 1.22.

---

## 🚀 Langkah Pertama: Sinkronisasi Repo
Sebelum mulai coding, tarik update terbaru dari remote:
```bash
git pull origin main
```
Pastikan kamu membaca perubahan pada:
- `src/models/ensemble_and_signals.py`
- `src/models/lstm_model.py`
- `reports/trading_backtest_all.csv`

---

## 🛠️ Tugas & Tanggung Jawab Daf Selanjutnya

### 1️⃣ Buat Pipeline Live Daily Prediction (`src/models/predict_daily.py`)
Sekarang saatnya membuat sistem prediksi operasional harian (*T+1 live scoring*):
- **Tujuan:** Mengambil data penutupan hari ini untuk 5 saham (`BBCA.JK`, `BBRI.JK`, `BMRI.JK`, `TLKM.JK`, `ASII.JK`), menghitung 29 indikator teknikal dari data 60 hari terakhir, lalu memprediksi arah pergerakan harga **besok**.
- **Output:**
  - Probabilitas naik $P(\text{Up})$.
  - Rekomendasi Sinyal Trading:
    * $P(\text{Up}) \ge 0.53 \rightarrow$ **BUY**
    * $P(\text{Up}) \le 0.47 \rightarrow$ **SELL**
    * $0.47 < P(\text{Up}) < 0.53 \rightarrow$ **HOLD**
  - Insert otomatis ke tabel `predictions` di database dengan `date = tanggal_besok`.
- **Bisa dijalankan via CLI:**
  ```bash
  python src/models/predict_daily.py --model ensemble
  ```

### 2️⃣ Model Registry & Serialization Check
- Pastikan bobot model tersimpan rapi dan dapat di-load secara cepat tanpa re-training:
  - `reports/models/lightgbm_model.joblib`
  - `reports/models/lstm_model.pt`
  - `reports/models/lstm_scaler.joblib`
- Tambahkan metadata versi model (misal format JSON: nama model, tanggal training, bobot ensemble optimal: `LightGBM: 0.15, LSTM: 0.85`).

### 3️⃣ Unit Testing & Validasi Anti-Leakage (`tests/`)
Buat unit test otomatis dengan `pytest` untuk menjamin kualitas kode:
- `tests/test_features.py`:
  - Verifikasi bahwa semua fitur teknikal strictly backward-looking (tidak menghasilkan sinyal dari masa depan).
  - Verifikasi tidak ada NaN liar setelah index *warmup* (50 hari).
- `tests/test_inference.py`:
  - Verifikasi `predict_proba()` selalu mengembalikan nilai float dalam rentang $[0.0, 1.0]$.
  - Verifikasi konsistensi mapping sinyal (BUY / HOLD / SELL).

---

## 📞 Koordinasi dengan Tim
- **Kevin:** Jika butuh diskusi terkait format database, backtesting, atau validasi ensemble.
- **Ali:** Informasikan skema tabel `predictions` dan sediakan helper function untuk Ali mengambil sinyal prediksi terbaru agar bisa ditampilkan di Frontend Dashboard.
