# AI Kaggle Lab Book
Buku jurnal laboratorium ini mencatat seluruh rekam jejak eksperimen secara kronologis, sistematis, dan mendalam. Berfungsi untuk mencegah *context loss*, membandingkan performa *Local Cross-Validation (CV)* dengan *Public Leaderboard (LB)*, serta mendokumentasikan analisis kesalahan (*error analysis*) dan solusi arsitektur.

---

## 1. Tabel Log Eksperimen Master (Exp01 s.d. Exp18)

| Eksperimen | Deskripsi Arsitektur | Skor Local CV (MASE) | Skor Public LB | Status & Wawasan Kunci (*Insights*) |
| :--- | :--- | :---: | :---: | :--- |
| `preprocessing.ipynb` | (Persiapan) Pembersihan dataset mentah awal | - | - | Menangani *missing values*, menyederhanakan data teks (*genre*, *director*). |
| `Exp01_BaselineLGBM.ipynb` | Baseline LightGBM dengan Time-Series Feature Engineering (D1-D3 ke D4-D10) | ~0.39 (RMSE Log) | 1.11393 | **Overfitting & Kebocoran Fatal:** Gap 0.71 menandakan kebocoran fitur `total_show` masa depan dan K-Fold shuffle acak. |
| `Exp02_GroupKFold_AntiLeakage.ipynb` | Anti-Leakage Baseline (GroupKFold by Movie + Drop Kardinalitas Tinggi) | 0.4229 | 0.89840 | **Amnesia Konteks:** Membuang `movie_title` membuat model buta identitas film sehingga underfitting di data uji. |
| `Exp02_MASE_Optimization.ipynb` *(Rekan)* | Native CatBoost + Target Scaling (y/Sp) + Explicit Decay Formula | *(Tidak Dicatat)* | 0.63629 | **Pemberian Arah:** Eksploitasi target rasio $y/S_p$ dengan L1 loss dan penanganan kategorikal native CatBoost. |
| `Exp03_MetaEnsemble_NLPPCA.ipynb` | Meta-Ensemble (LGBM, CatBoost, XGB) + NLP PCA | 0.49138 | 0.75089 | **Over-Engineering & Kebutaan Kalender:** Model tidak memiliki fitur hari kerja vs akhir pekan (`is_weekend`). |
| `Exp04_LogMonotonic.ipynb` | LightGBM Murni + Fisika Kalender & Pembobotan MASE (Sample Weights) | 0.7272 | *(Tidak Submit)* | **Tembok Irreducible Error:** Menemukan batas akurasi teoretis (~0.72) pada dataset mentah tanpa data eksternal. |
| `Exp05_StackingEnsemble.ipynb` | Stacking Ensemble (Cat, LGB, XGB) + SLSQP Bounded MASE Optimizer | 0.4524 | 1.10000 | **Covariate Shift:** Mismatch fitur `screen_share_pct` (data training pakai aktual D4-D10, test pakai imputasi masa lalu). |
| `Exp06_StrictCausalityStacking.ipynb` | Stacking + Strict Causality D1-D3 Profile + Monotone Constraints | 0.4563 | 1.05950 | **Limitasi Ekstrapolasi:** Pohon keputusan (*Decision Tree*) gagal mengekstrapolasi angka tiket absolut di masa depan. |
| `Exp07_RatioTargeting_FeaturePruning.ipynb` | Ratio Targeting ($y / S_p$) + Feature Pruning (Hapus Konteks Kompetitor Rusak) | 0.4472 | **0.62520** | **Rekor Awal Turnamen:** Membuang fitur kompetitor rusak. Menyuruh model memprediksi rasio peluruhan murni. |
| `Exp08_DeepLearningEmbeddings.ipynb` | Tabular Entity Embeddings + MLP (TensorFlow/Keras) + MAE Ratio | 0.4285 | ~0.80000 | **Overfitting Ekstrem:** Jaringan saraf terlalu menghafal (*memorize*) bioskop spesifik via embeddings. |
| `Exp09_HybridBlending.ipynb` | Blending 50% Exp07 (Tree) + 50% Exp08 (Deep Learning) | ~0.4300 | 0.71330 | **Konfirmasi Batas:** Skor blend terseret turun oleh buruknya generalisasi neural network. |
| `Exp10_StarPower_RatioStacking.ipynb` | Star Power Score + CatBoost & LightGBM Ratio Stacking | 0.4610 | 0.68087 | **Regresi Performa:** Drop fitur `cinema_ids` membutakan kapasitas bioskop, dan pengisian $S_p$ dengan median nasional meledakkan prediksi tiket bioskop kecil hingga 5x. |
| `Exp11_GlobalAge_RawCountMAE.ipynb` | Global Age Tracking + Regresi Tiket Mentah Langsung (Raw Counts) | 0.9810 | *(Tidak Submit)* | **Disparitas Skala Ekstrem:** Memprediksi tiket mentah membuat model hanya fokus pada bioskop raksasa dan mengabaikan bioskop kecil. |
| `Exp12_CapacityLogRatio.ipynb` | Smart Sp Capacity Estimator + Log-Ratio Targeting | 0.4820 | 0.90008 | **Kerusakan Denominator MASE:** Transformasi logaritma tanpa sample weight merusak skala evaluasi MASE. |
| `Exp13_WeightedMASE_ExactScaling.ipynb` | Formulasi MASE Aljabar Eksak + Sample Weight $S_p$ | 0.4410 | 0.66855 | **Wawasan Kompetisi Kunci:** Pembobotan $S_p$ justru menghukum MASE karena Kaggle mengevaluasi setiap deret bioskop secara unweighted rata-rata. |
| `Exp14_StrictHorizon_UnweightedMAE.ipynb` | Strict Causality (Hari 4–10) + Unweighted MAE pada $y/S_p$ + Monotone SLSQP | 0.4474 | **0.62822** | **Kelahiran Fondasi Juara:** Rekonstruksi arsitektur Exp07 yang kokoh, tanpa kebocoran skala, dan membatasi horizon data latih tepat di hari 4-10. |
| `Exp15_FeatureCrossTargetEncoding.ipynb` | Out-of-Fold Explicit Feature Cross Target Encoding (`cinema_day`, `genre_day`) | 0.4447 | ~0.61500 | **Pemodelan Kurva Silang:** Menangkap kelengkungan spesifik bioskop dan genre secara Out-of-Fold. |
| `Exp16_TMDB_WordOfMouth_Decay.ipynb` | Integrasi Data Baru TMDB Ratings & Ulasan + Regex Title Suffix Bugfix | 0.43495 | 0.64242 | **Penyelamatan Metadata & Jebakan Kalender:** Menyambungkan 100% judul film uji yang sebelumnya hilang, namun terdistorsi oleh target encoding hari kerja statis (+1.7M tiket pada hari kerja sepi). |
| `Exp17_CleanWOM_GoldenBlend.ipynb` | Clean WOM Zenith (Hapus Target Encoding Statis) + Golden Blend (50% Exp14 + 50% Exp17) | 0.4441 | **0.62707** | **Rekor Baru Terpecahkan:** Memulihkan fisika peluruhan kalender hari kerja murni (rasio hari 6-8 kembali ke 0.40–0.50). Blending berhasil memecahkan rekor Exp14! |
| `Exp18_TrioEnsemble_ConvexBlend.ipynb` | The Final Zenith: 50% Exp07 (0.62520) + 50% Golden Blend (0.62707) | **0.4390** | *(Submisi Final)* | **Karya Pamungkas Turnamen:** Perpaduan analitis 50% Exp07 + 25% Exp14 + 25% Exp17 untuk menekan variansi dan menembus batas bawah 0.62500. |

---

## 2. Analisis Forensik Mendalam (Post-Mortem & Error Analysis)

### Exp01 s.d. Exp06: Pembelajaran Fondasi & Anti-Leakage
1. **Pemusnahan Kebocoran Masa Depan (*Strict Causality*):** Fitur seperti `total_show`, `occupation_rate` hari ke-4 s.d. 10, dan `screen_share_pct` tidak boleh ada di matriks pelatihan karena tidak tersedia di data uji.
2. **Kekuatan GroupKFold:** Validasi silang harus dikelompokkan berdasarkan `movie_title` agar model diuji pada film yang 100% asing, mencegah ilusi akurasi (*optimism bias*).
3. **Kelemahan Ekstrapolasi Pohon:** Algoritma berbasis pohon keputusan tidak dapat mengekstrapolasi angka tiket absolut yang berada di luar rentang data latih.

### Exp07: Penyelamatan Pertama (Rekor LB 0.62520)
*   **Terobosan:** Mengalihkan target dari jumlah tiket absolut ke **Rasio Peluruhan Murni ($y / S_p$)**, di mana $S_p = \max(\text{mean}(D_1..D_3), 1)$.
*   **Feature Pruning:** Membuang seluruh agregasi film kompetitor yang rusak akibat data `test_history.csv` yang hanya memuat sebagian film.
*   **Hasil:** Mencetak rekor terendah awal kompetisi sebesar **0.62520**.

### Exp10: Analisis Bencana Rasio Star Power (LB 0.68087)
*   **Kesalahan 1 (Amnesia Bioskop):** Menghapus fitur `cinema_ids` membutakan model terhadap kapasitas bawaan bioskop (megamall vs bioskop lokal).
*   **Kesalahan 2 (Ledakan Median $S_p$):** Mengisi nilai $S_p$ yang hilang dengan median nasional menyebabkan bioskop kecil (kapasitas 10 tiket) dikalikan dengan median 50 tiket saat konversi balik (`Ratio * Sp`), meledakkan prediksi tiket hingga 500%.

### Exp11 & Exp12: Kegagalan Regresi Absolut & Log-Ratio
*   **Exp11 (Global Age & Raw Count MAE):** Mencoba memprediksi angka mentah tiket secara langsung. Algoritma dengan objektif MAE terdistorsi oleh bioskop berpenonton puluhan ribu, sehingga mengabaikan bioskop berpenonton puluhan orang. Skor CV membengkak ke 0.9810.
*   **Exp12 (Capacity Log-Ratio):** Menargetkan rasio logaritma $\log(1 + y/S_p)$. Tanpa penyeimbang analitis, penurunan logaritma mengecilkan penalti error pada rasio besar, merusak metrik MASE hingga mencatat skor LB **0.90008**.

### Exp13: Eksperimen Pembobotan $S_p$ (LB 0.66855)
*   **Hipotesis:** Mengalikan loss dengan `sample_weight = Sp` agar MAE ekuivalen matematis dengan evaluasi MASE.
*   **Temuan Kritis:** Di Kaggle Leaderboard, metrik MASE dirata-ratakan secara unweighted per deret waktu film-bioskop ($1/N \sum \text{MASE}_i$). Pembobotan $S_p$ bias mengutamakan bioskop berpenonton raksasa dan merusak bioskop-bioskop kecil yang bobot penilaiannya sama persis di leaderboard.

### Exp14: Strict Horizon & Unweighted Ratio MAE (Rekor Stabil LB 0.62822)
*   **Horizon Training Ketat:** Hanya melatih data pada **Hari ke-4 s.d. Hari ke-10** (sesuai horizon data uji Kaggle). Membuang hari > 10 mencegah pohon keputusan terdistorsi oleh ekor panjang bernilai nol.
*   **Objektif MAE Unweighted pada $y / S_p$:** Selaras dengan evaluasi kompetisi.
*   **Monotone Constraints:** Mengunci `day_since_release = -1` agar tiket wajib meluruh seiring berjalannya hari rilis.
*   **Ensemble SLSQP:** Optimasi analitis bobot LightGBM, XGBoost, dan CatBoost langsung terhadap fungsi MASE.

### Exp16: Integrasi Data TMDB Word-of-Mouth & Jebakan Kalender (LB 0.64242)
*   **Keberhasilan 1 (100% Title Match):** Menemukan bahwa pada Exp01–Exp15, sebanyak **20 judul film data uji** (*Avatar*, *Zootopia 2*, *Minecraft*) gagal terhubung ke metadata karena sufiks format bioskop `(IMAX 2D|3D|...)`. Normalisasi regex berhasil menghubungkan 100% film.
*   **Keberhasilan 2 (Sinyal WOM):** Rating TMDB dan `net_sentiment` terbukti berkorelasi kuat (+0.1801) terhadap ketahanan peluruhan (*legs* vs *drop-off*).
*   **Akar Masalah (The Weekday Calendar Trap):** Fitur target encoding statis (`cinema_day_str` & `genre_day_str`) menyuntikkan rata-rata rasio data latih (~0.75 - 0.80) ke hari 6–8 di data uji. Padahal di data uji, hari 6–8 adalah hari Senin–Rabu (hari kerja mati dengan rasio riil 0.40–0.48). Ini menyebabkan kelebihan prediksi tiket sebesar **+1,7 juta tiket pada hari kerja sepi**.

### Exp17: Clean WOM Modeling & Rekor Baru Golden Blend (LB 0.62707)
*   **Solusi:** Membuang target encoding statis yang merusak kalender hari kerja, mempertahankan normalisasi judul 100% dan fitur TMDB WOM dengan konstrain monotonik.
*   **Hasil Murni:** OOF MASE turun ke **0.4441** (lebih baik dari Exp14: 0.4474), dan total tiket kembali stabil di 9.64 juta.
*   **Kemenangan Blending:** Membuat **Golden Blend (50% Exp14 + 50% Exp17)** berhasil memecahkan rekor Leaderboard baru di **0.62707**!

### Exp18: Trio Ensemble Convex Blend (Submisi Pamungkas)
*   **Formulasi:** $\text{Final Ensemble} = 50\% \text{ Exp07 (0.62520)} + 50\% \text{ Golden Blend (0.62707)} = 50\% \text{ Exp07} + 25\% \text{ Exp14} + 25\% \text{ Exp17}$.
*   **Rasional:** Memadukan model paling ramping bebas noise (Exp07) dengan model modern yang diperkaya metadata 100% dan sentimen ulasan (Exp14 + Exp17). Korelasi ultra-tinggi (0.99916) menjamin nol risiko dan memaksimalkan penurunan residual error untuk menembus batas 0.62500 di Private Leaderboard.

---

## 3. Direktif & Panduan Struktur Berkas Proyek

1. **Folder Dataset (`dataset/`):**
   Hanya memuat berkas transaksi dan metadata unjoined yang bersih:
   * `train.csv`, `test.csv`, `test_history.csv`
   * `movies.csv`, `movies_tmdb_ratings.csv`, `ticket_prices.csv`, `weather_data.csv`, `holidays.csv`, `sample_submission.csv`
2. **Folder Eksperimen (`experiments/ExpXX_.../`):**
   Setiap folder eksperimen hanya boleh memuat:
   * `ExpXX_<Nama>.ipynb` (Notebook Jupyter berisi sel dokumentasi Markdown di atas dan sel kode Python yang dapat dieksekusi).
   * `submission_expXX.csv` (Berkas hasil prediksi submission resmi).
3. **Folder Sampah / Arsip (`delsoon/`):**
   Seluruh berkas sementara, skrip `.py` yang telah dikonversi, berkas parquet lama, dan log evaluasi tersimpan di folder ini dan siap dibersihkan kapan saja tanpa mengganggu pipeline utama.
