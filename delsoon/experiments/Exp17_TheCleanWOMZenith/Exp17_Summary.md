# Ringkasan Eksekutif Eksperimen 17: Exp17_TheCleanWOMZenith & Golden Blend

## 1. Latar Belakang & Masalah yang Diselesaikan
- Pada Exp16, introduksi *Target Encoding* statis (`cinema_day_str` & `genre_day_str`) menyebabkan distorsi pada distribusi hari kerja (*the weekday trap*). Di data uji Kaggle, hari ke-6, 7, 8 adalah hari Senin–Rabu (hari kerja mati), tetapi model Exp16 memprediksi rasio tinggi (~0.60) berdasarkan rata-rata data latih, sehingga skor LB berada di 0.64242.
- Pada Exp17, arsitektur dikembalikan ke kesederhanaan dan kepatuhan kalender murni seperti Exp14 (yang sebelumnya mencetak rekor LB 0.62822), namun diperkuat dengan perbaikan 100% judul film uji dan sinyal Word-of-Mouth (TMDB Ratings).

## 2. Inovasi & Hasil Validasi Out-of-Fold (OOF)

| Metrik / Model | Exp14 (Baseline LB 0.62822) | Exp16 (Weekday Distorted) | Exp17 (The Clean WOM Zenith) |
| :--- | :---: | :---: | :---: |
| **OOF MASE** | 0.4474 | 0.4349 (bocor TE) | **0.4441 (Murni & Bersih)** |
| **Mean Prediksi Tiket** | 131.35 | 154.25 (+1.7M overpredict) | **132.82 (Presisi)** |
| **Rata-rata Rasio Hari 6-8** | 0.48 / 0.42 / 0.39 | 0.61 / 0.59 / 0.55 | **0.50 / 0.45 / 0.40** |
| **Status Metadata Judul** | 20 judul uji hilang metadata | 100% terhubung | **100% terhubung sempurna** |

## 3. Berkas Submission yang Dihasilkan
1. `experiments/Exp17_TheCleanWOMZenith/submission_exp17.csv` (dan disalin ke `submission_exp17.csv` di root)
   - Prediksi murni model Exp17 Clean WOM (LightGBM 0.3526 + XGBoost 0.6474).
2. `experiments/Exp17_TheCleanWOMZenith/submission_golden_blend.csv` (dan disalin ke `submission_golden_blend.csv` di root)
   - **Golden Blend (50% Exp14 + 50% Exp17)**: Jaring pengaman turnamen terkuat. Menggabungkan model yang sudah terbukti di leaderboard (0.62822) dengan model yang menyerap perbaikan metadata film dan sinyal TMDB.
