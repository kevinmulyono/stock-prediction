# 🚀 Panduan Deploy: 100% Gratis di Netlify (Tanpa Kartu Kredit)

Selamat! Proyek dashboard prediksi saham IDX ini telah dioptimalkan menggunakan arsitektur **JAMstack (Static JSON + Netlify)** yang **100% gratis selamanya**, tidak memerlukan kartu kredit, dan tidak membutuhkan server backend berbayar.

---

## 📌 Bagaimana Cara Kerjanya?

```
┌────────────────────────────────────────────────────────┐
│                   GitHub Repository                    │
│                                                        │
│  1. GitHub Actions (Cron gratis setiap 18:00 WIB)      │
│     - Mengambil candle terbaru via Yahoo Finance       │
│     - Menjalankan model AI Ensemble (LightGBM + LSTM)  │
│     - Meng-export data terbaru ke 'frontend/data/'     │
│     - Auto-commit & push ke branch 'main'              │
└───────────────────────────┬────────────────────────────┘
                            │ (Auto-trigger build & deploy)
                            ▼
┌────────────────────────────────────────────────────────┐
│                   Netlify CDN (Free)                   │
│                                                        │
│  - Membaca langsung 'frontend/' dan 'frontend/data/'   │
│  - Grafik TradingView memuat lilin & sinyal instan     │
│  - Kecepatan load 60 FPS, ultra cepat & tanpa lag      │
│  - HTTPS SSL gratis otomatis                           │
└────────────────────────────────────────────────────────┘
```

---

## 🛠️ Langkah-Langkah Deploy ke Netlify

### Langkah 1: Push Update Terakhir ke GitHub
Buka terminal (Git Bash / PowerShell), lalu jalankan:

```bash
git add .
git commit -m "feat: setup static JSON architecture and daily GitHub Actions for Netlify"
git push origin main
```

---

### Langkah 2: Hubungkan Repositori ke Netlify

1. Buka [https://app.netlify.com](https://app.netlify.com) dan login menggunakan akun **GitHub** Anda.
2. Klik tombol **`Add new site`** di dashboard ➔ pilih **`Import an existing project`**.
3. Pilih penyedia Git: **`GitHub`**.
4. Cari dan pilih repositori Anda: **`stock-prediction`**.
5. Netlify akan otomatis membaca file `netlify.toml` yang sudah kita konfigurasikan:
   - **Base directory:** *(kosongkan / root)*
   - **Build command:** *(kosongkan)*
   - **Publish directory:** `frontend`
6. Klik tombol **`Deploy stock-prediction`**.
7. Dalam **10–20 detik**, situs Anda sudah langsung **LIVE** dengan domain gratis (contoh: `https://stock-prediction-ali.netlify.app`)! 🎉

---

## ⚙️ Otomatisasi Update Harian (GitHub Actions)

Alur kerja GitHub Actions telah dibuat di:
[`.github/workflows/daily_sync_and_export.yml`](file:///.github/workflows/daily_sync_and_export.yml)

### Kapan Data Diperbarui?
- **Otomatis**: Setiap hari kerja bursa (Senin – Jumat) pukul **18:00 WIB** (11:00 UTC, setelah bursa IDX tutup dan candle harian resmi dirilis oleh Yahoo Finance).
- **Manual (Kapan Saja)**:
  1. Masuk ke tab **`Actions`** di repositori GitHub Anda.
  2. Klik workflow **`Daily Market Data Sync & Static Export`** di panel kiri.
  3. Klik tombol **`Run workflow`** ➔ pilih branch `main` ➔ klik **`Run workflow`**.
  4. Dalam 1–2 menit, data terbaru akan di-generate, di-push ke GitHub, dan Netlify akan otomatis memperbarui situs web Anda!

---

## 🔍 Keunggulan Pendekatan Ini

1. **Bebas Biaya & Tanpa Kartu Kredit:** Netlify dan GitHub Actions 100% gratis untuk repositori publik dan personal.
2. **Tidak Pernah Sleep / Loading Lambat:** Karena menggunakan file statis di CDN global Netlify, web tidak akan mengalami "Cold Start" (server bangun dari tidur) seperti pada free tier Render/Heroku.
3. **Respon Grafik Super Cepat:** Filter timeframe (1M, 3M, 6M, 1Y, ALL) dan pergantian saham diproses instan di browser tanpa menunggu respons server.
4. **Keamanan Maksimal:** Tidak ada port backend terbuka atau celah database publik.
