# Eksperimen 03 (Frankenstein Meta-Ensemble): Cetak Biru Operasional

Dokumen ini adalah panduan teknis (*Standard Operating Procedure*) mutlak untuk mengeksekusi Eksperimen 03. Tujuan utama kita adalah mencapai **Skor MASE < 0.3** dengan menggabungkan peretasan matematis (skala $S_p$), keamanan validasi (`GroupKFold`), arsitektur multi-model (*Stacking Ensemble*), dan pemrosesan bahasa alami (NLP).

## Penjabaran Taktis per Fase (Milestone M1 - M10)

### M1: Setup & Environment Initialization
- **Tujuan:** Mengunci *seed* rawak (42) untuk reproduktibilitas 100% dan memuat seluruh *library* tingkat lanjut.
- **Teknologi:** `lightgbm`, `catboost`, `xgboost`, `sentence_transformers` (untuk NLP Embeddings), `optuna`, `scikit-learn` (Ridge/Lasso untuk meta-learner).

### M2: Data Extraction & Mathematical Preprocessing (Krusial)
- **Ekstraksi Sejarah (D1-D3):** Mengekstrak data awal untuk `train.csv` (dari dirinya sendiri) dan `test.csv` (dari `test_history.csv`) menggunakan *groupby* kombinasi mutlak `['movie_title', 'cinema_ids']`.
- **Peretasan MASE (S_p Scaling):** Menghitung batas bawah MASE, $S_p = max(mean(D1..D3), 1.0)$.
- **Skala Target (Rahasia Skor 0.3):** Mengubah target utama pelatihan dari `total_ticket` menjadi `y_scaled = total_ticket / Sp`. Seluruh algoritma (LGBM, CatBoost, XGBoost) nantinya akan dilatih dengan target ini menggunakan *loss function* MAE (L1 Loss).

### M3: Feature Engineering (Dinamika Kurva & Waktu)
- **Rasio Momentum Eksponensial:** Menghitung peluruhan non-linier seperti `ratio_d3_d1 = (ticket_d3 + 1) / (ticket_d1 + 1)` dan `momentum_d3_d2 = (ticket_d3 - ticket_d2) / (ticket_d2 + 1)`.
- **Fitur Kalender Kompleks:** Mengekstrak *day_number* (D4 = 3, dst), status *weekend*, hari libur, dan estimasi potensi pendapatan (`total_show * price`).
- **Penyatuan Dataset:** Menggabungkan seluruh data tabular (`movies.csv`, `holidays.csv`, `ticket_prices.csv`) dengan sangat aman.

### M4: Categorical Handling & NLP Embeddings (Revolusi)
- **TIDAK ADA PENGHAPUSAN KOLOM TEKS.** Kita melarang keras penghapusan `movie_title` dan `director`.
- **NLP Embeddings (Untuk Meta-Model/XGB/LGB):** Menggunakan `SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')` untuk mengubah `movie_title` dan `genre` menjadi vektor representasi semantik. Vektor ini di-*reduksi* dimensinya dengan PCA (misal: 5-10 dimensi) agar bisa dicerna pohon keputusan.
- **Native Categorical (Untuk CatBoost):** *CatBoost* secara alamiah akan mencerna fitur kategori teks mentah.

### M5: Feature Selection
- Mengamankan daftar akhir dari **Fitur Prediktor (`FEATURES`)**.
- Memastikan **Paritas Kolom 100%** antara *Training Set* dan *Testing Set*. Jika ada yang hilang di *Test*, otomatis diselaraskan atau direkonstruksi.

### M6: Hyperparameter Tuning (Optuna Ekstrem)
- Kita tidak menebak parameter. Kita akan menjalankan dua blok Optuna secara terpisah (satu untuk LightGBM, satu untuk XGBoost/CatBoost).
- **Fungsi Objektif Optuna:** Secarah harfiah meminimalkan validasi L1 pada target terskala (`y_scaled`) menggunakan `GroupKFold(groups=movie_title)`.

### M7: Cross-Validation & Stacking Ensemble (Inti Otak)
- Menggunakan `GroupKFold(n_splits=5)`.
- **Lapis 1 (Base Learners):** Melatih 3 model (LightGBM, CatBoost, XGBoost) pada setiap lipatan (*fold*). Kita akan menyimpan prediksi *Out-of-Fold* (OOF) mereka ke dalam matriks baru.
- **Lapis 2 (Meta-Learner):** Menggunakan model linier kuat seperti `RidgeCV` atau `LinearRegression` untuk mempelajari matriks prediksi Lapis 1. *Ridge* akan otomatis membebani (memberikan bobot probabilitas) ke model yang paling akurat, misal: *0.55 LGB + 0.35 CAT + 0.10 XGB*.

### M8: Evaluasi Kritis
- Mengubah kembali `y_scaled` hasil meta-learner menjadi satuan tiket asli ($y = y\_scaled \times S_p$).
- Menghitung MASE akhir. Kita menargetkan **CV Lokal < 0.40**.
- Melakukan analisis *Feature Importance* (terutama mengecek apakah NLP Embeddings benar-benar terpakai).

### M9: Inferensi (Data Uji Tertutup)
- Menerapkan *Pipeline* M2 hingga M4 yang **100% identik** kepada data `test.csv`. Menjamin ekstraksi NLP dan kalkulasi $S_p$ berjalan tanpa cacat.
- Melakukan inferensi Lapis 1 pada *test*, lalu diumpankan ke Lapis 2 (*Meta-Learner*).
- Memotong (*clipping*) prediksi agar tidak menghasilkan tiket bernilai negatif.

### M10: Submission Generation
- Membentuk *DataFrame* berisi `id` dan `total_ticket`.
- Mengamankan berkas final ke dalam `Exp03_Submission_Ultimate.csv`. Menutup dengan pencetakan 10 baris pertama untuk inspeksi visual.
