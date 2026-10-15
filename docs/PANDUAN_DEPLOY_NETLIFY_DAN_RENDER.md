# 🚀 Panduan Deploy: Frontend di Netlify & Backend di Render.com

Panduan langkah demi langkah untuk menghubungkan **Web Dashboard (Netlify)** dengan **FastAPI Machine Learning Backend (Render.com)**.

---

## 📌 Ringkasan Arsitektur
```
[ Pengunjung Web ]
       │
       ▼
┌──────────────────────────┐
│  Netlify (Frontend CDN)  │  ← Menampilkan HTML/CSS/JS, Grafik TradingView
│  https://...netlify.app  │
└────────────┬─────────────┘
             │  (Proxy otomatis via netlify.toml)
             ▼
┌──────────────────────────┐
│  Render.com (Python API) │  ← Menjalankan FastAPI, PyTorch, LightGBM,
│  https://...onrender.com │     Yahoo Finance Live Sync & SQLite DB
└──────────────────────────┘
```

---

## 🛠️ Langkah 1: Push Update Terbaru ke GitHub

Pastikan commit terbaru sudah diunggah ke GitHub:
```bash
git push origin main
```

---

## 🌐 Langkah 2: Deploy Backend di Render.com (Gratis)

1. Buka [https://render.com](https://render.com) lalu login menggunakan akun **GitHub** Anda.
2. Di halaman Dashboard Render, klik tombol **`New +`** di pojok kanan atas, lalu pilih **`Web Service`**.
3. Pilih repositori GitHub Anda: **`stock-prediction`**.
4. Isi konfigurasi berikut (Render biasanya akan mendeteksi otomatis dari `render.yaml`):
   - **Name:** `idx-stock-prediction-api` (atau nama pilihan Anda)
   - **Language:** `Python`
   - **Branch:** `main`
   - **Region:** `Singapore` (paling cepat untuk pengguna Indonesia)
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free`
5. Klik **`Deploy Web Service`**.
6. Tunggu sekitar 2-4 menit hingga proses build selesai dan status berubah menjadi **`Live`**.
7. Salin URL backend publik Anda yang tertera di bagian atas dashboard Render.  
   *Contoh URL:* `https://idx-stock-prediction-api.onrender.com`

> **💡 Catatan Otomatisasi:**  
> Backend telah dilengkapi fungsi auto-startup: database, metadata 5 emiten, dan lilin bursa terbaru dari Yahoo Finance akan otomatis diinisialisasi saat server pertama kali hidup di Render.

---

## 🔗 Langkah 3: Masukkan URL Backend ke `netlify.toml`

1. Buka file [`netlify.toml`](file:///d:/project%20mandiri/stock%20prediction/netlify.toml) di VS Code / Antigravity IDE.
2. Ganti baris target redirect:
   ```toml
   # SEBELUM:
   to = "https://YOUR_BACKEND_URL.onrender.com/api/:splat"

   # SESUDAH (Ganti dengan URL Render Anda):
   to = "https://idx-stock-prediction-api.onrender.com/api/:splat"
   ```
3. Commit dan push perubahan tersebut ke GitHub:
   ```bash
   git add netlify.toml
   git commit -m "chore: connect netlify proxy to render backend url"
   git push origin main
   ```

---

## 🎉 Langkah 4: Selesai! Buka Website Netlify Anda

1. Buka dashboard Netlify Anda. Netlify akan secara otomatis mendeteksi push baru dan melakukan deploy dalam waktu ~30 detik.
2. Buka URL website Netlify Anda (`https://nama-web-anda.netlify.app`).
3. **Hasil yang akan Anda lihat:**
   - Halaman **tidak lagi `Page not found`** karena Netlify langsung mem-publish folder `frontend/`.
   - Status badge di pojok atas akan berwarna hijau: **`Live API Active`**.
   - Kartu 5 emiten, grafik candlestick Stockbit, sinyal trading harian, dan tabel leaderboard akan langsung tampil lengkap secara online!
