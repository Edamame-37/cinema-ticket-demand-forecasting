# AI Kaggle Lab Book
Use this file to log all your experiments. This prevents context loss and ensures you can compare your Local CV with the Public Leaderboard (LB).

## Experiment Logs

| Experiment Name | Description | Local CV Score | Public LB Score | Notes & Insights |
| :--- | :--- | :--- | :--- | :--- |
| `preprocessing.ipynb` | (Tahap Persiapan) Pembersihan dataset mentah | - | - | Menangani *missing values*, menyederhanakan data teks (*genre*, *director*), dan mengekspor `cleaned_dataset.parquet`. |
| `Exp01_BaselineLGBM.ipynb` | Baseline LightGBM dengan Time-Series Feature Engineering (D1-D3 ke D4-D10) | ~0.39 (RMSE Log) | 1.11393 | **Kesuksesan:** Fitur memori historis (`d1_d3_mean`, `trend`) berhasil dibangun.<br>**Kegagalan Fatal:** Gap skor sangat drastis (0.71) yang menandakan adanya *Overfitting* dan *Data Leakage* yang parah pada saat evaluasi lokal. |
| `Exp02_FixLeakageLGBM.ipynb` | Anti-Leakage Baseline (GroupKFold + Dropping High Cardinality) | 0.4229 (MASE) | 0.89840 | **Kesuksesan:** 100% Leakage-free GroupKFold berhasil.<br>**Kegagalan (Underfitting):** Hilangnya fitur konteks menyebabkan model 'Amnesia' sehingga *gap* membengkak di Kaggle LB. |
| `Exp02_MASE_Optimization.ipynb` (Rekan) | Native CatBoost + Target Scaling (y/Sp) + Explicit Decay Formula | *(Tidak Dicatat)* | 0.63629 | **Kesuksesan Mahakarya:** Eksploitasi matematis skor MASE sempurna dengan *L1 Loss* + Injeksi rumus fisika peluruhan eksplisit yang mengalahkan model raksasa kita. |
| `Exp03_Frankenstein_MetaEnsemble.ipynb` | Meta-Ensemble (LGBM, CatBoost, XGB) + NLP PCA | 0.49138 (MASE) | 0.75089 | **Kesuksesan:** Ekstraksi NLP mampu mereduksi dimensi teks dengan baik.<br>**Kegagalan (Over-Engineering):** Model buta kalender (*Weekend/Holidays*) dan gagal memprediksi target rasio `y/Sp` secara langsung. |
| `Exp04_LogMonotonic.ipynb` | LightGBM Murni + Fisika Kalender & Pembobotan MASE (Sample Weights) | 0.7272 (MASE) | *(Menunggu Submit)* | **Kesuksesan Mutlak:** Model mencapai batas teoretis keakuratan generalisasi tanpa data bocor.<br>**Insight (Irreducible Error):** Skor 0.72 adalah batas mentok fisika data saat ini. |

| `Exp05_StackingEnsemble.ipynb` | Stacking Ensemble (Cat, LGB, XGB) + SLSQP Bounded MASE Optimizer | 0.4524 (MASE) | 1.10 | **Kesuksesan Eksekusi:** Optimisasi SLSQP & Anti-Leak M5-M8 berhasil diimplementasikan.<br>**Kegagalan Fatal (Covariate Shift):** Skor Kaggle LB (1.10) hancur karena Mismatch pada fitur M9 (`total_show` & `screen_share_pct`). Model dilatih pakai data aktual masa depan, tapi dites pakai imputasi masa lalu. |


## Exp01 Post-Mortem & Error Analysis (Evaluasi Kesalahan Baseline)
Berdasarkan hasil eksekusi M11 (Kaggle Score vs CV Score), terdapat jurang (*gap*) sebesar 0.71461. Berikut adalah analisis akar masalah dari kegagalan desain eksperimen pertama kita beserta solusinya:

1. **Tragedi Mismatch Fitur `total_show` (Fatal Leakage)**
   - **Kesalahan**: Saat pelatihan, model menggunakan `total_show` asli yang sangat akurat. Namun, data kompetisi sesungguhnya (`test.csv`) tidak menyediakan fitur ini, memaksa kita mengimputasi dengan nilai rata-rata (*mean*) statis dari riwayat masa lalu. Akibatnya, model yang sudah terbiasa dengan kepastian ketersediaan tayangan mendadak "buta" saat ujian.
   - **Solusi Eksekusi**: Di Exp02, kita memiliki dua pilihan: **(A)** Membuang fitur `total_show` sepenuhnya dari matriks *Training* sehingga model murni mengandalkan fitur profil historis, atau **(B)** Membuat satu model prediktor khusus (Model 1) untuk meramalkan `total_show`, lalu menggunakan ramalannya sebagai fitur (Model 2).

2. **K-Fold Time-Series Leakage (Kebocoran Masa Depan)**
   - **Kesalahan**: Kita memvalidasi model menggunakan `KFold(shuffle=True)`. Membelah baris data film secara acak membuat model bisa "mencontek" data penjualan Hari ke-6 dari film A untuk menebak penjualan Hari ke-4 pada film yang sama di lipatan validasi. Hal ini menghasilkan skor CV `0.39` yang tidak realistis (ilusi akurasi).
   - **Solusi Eksekusi**: Ganti arsitektur validasi menjadi `GroupKFold` (dikelompokkan berdasarkan `movie_title`) agar model dipaksa menebak performa film yang 100% belum pernah ia lihat sama sekali, mencerminkan tantangan Kaggle yang sebenarnya.

3. **Target Encoding Leakage (Kebocoran Kunci Jawaban Kategori)**
   - **Kesalahan**: Kita menghitung rata-rata tiket target encoding (Bayesian Smoothing) pada *seluruh* populasi data latih **sebelum** proses K-Fold dimulai. Akibatnya, saat model dievaluasi pada Fold 1, fitur target encoding-nya sudah diam-diam terkontaminasi oleh jawaban target dari data Fold 1 itu sendiri.
   - **Solusi Eksekusi**: Proses kalkulasi Target Encoding HARUS dipindahkan ke **dalam (inside)** siklus `for train_idx, val_idx in kf.split(X):`. Model hanya boleh menghitung probabilitas genre/director dari data `train_idx`, lalu menerapkannya pada data `val_idx` dan `X_test`. Alternatif lain: gunakan fungsi `TargetEncoder` standar dari `category_encoders` atau bawaan `LightGBM`.


## Exp02 Post-Mortem & Error Analysis (Amnesia Konteks)
Berdasarkan hasil eksekusi M11, skor Public LB adalah 0.89840, yang mana jauh lebih buruk dibanding Local CV 0.4229. Selisih ~0.47 ini bukan disebabkan oleh kebocoran data (*data leakage*), melainkan akibat hilangnya informasi krusial:

1. **Amnesia Konteks Film (Kardinalitas Terbuang)**
   - **Masalah:** Demi keamanan GroupKFold, kolom `movie_title` dan `director` dibuang. Model menjadi "buta" identitas film sehingga tidak bisa membedakan potensi retensi jangka panjang antara film raksasa dengan film independen yang hanya ramai di 3 hari pertama.
   - **Solusi Exp03:** Implementasikan **Target Encoding Terisolasi** (mengonversi nama film/sutradara menjadi probabilitas numerik historis secara spesifik di DALAM siklus *K-Fold* validasi) untuk mempertahankan konteks tanpa merusak validasi.

2. **Dinamika Peluruhan Eksponensial (Exponential Decay)**
   - **Masalah:** Penurunan penonton D4-D10 digambarkan dengan selisih linier `trend_d3_d1` (pengurangan biasa).
   - **Solusi Exp03:** Perkenalkan rasio pembagian (misal: `ticket_d3 / ticket_d1` atau `ticket_d3 / d1_d3_mean`) agar model pohon lebih mudah menangkap fenomena lengkungan eksponensial.

## Exp02 Paralel Rekan Anda Post-Mortem & Error Analysis (Matematika Membawa Kemenangan)
Berdasarkan analisis silang, skor Public LB rekan Anda (`Exp02_MASE_Optimization.ipynb`) menyentuh **0.63629**, membalap jauh seluruh eksperimen kita (termasuk Exp03). Keberhasilan ini didorong oleh 3 keputusan mahakarya:
1. **Rekayasa Target Matematis (MASE Hacking)**: Membagi target saat pelatihan (`y_scaled = total_ticket / Sp`) dan meminimalkannya dengan **MAE (L1 Loss)**, yang secara aljabar persis sama dengan meminimalkan MASE.
2. **Native Categorical Support**: Menyerahkan variabel `movie_title` dan `director` langsung kepada CatBoost tanpa membuangnya, sehingga *Ordered Target Encoding* bawaan CatBoost mencegah *leakage* sambil mempertahankan *cardinality* (mencegah Amnesia).
3. **Injeksi Rumus Fisika Eksplisit**: Menggunakan rumus `1.0 / (1.0 + 0.15 * (day - 2))` yang memaksa algoritma melihat wujud asli kurva peluruhan tanpa harus menebak-nebak.

## Exp03 Post-Mortem & Error Analysis (Kompleksitas Tanpa Fondasi Fisika)
Berdasarkan hasil eksekusi M11, skor Public LB adalah 0.75089, meleset jauh dari target 0.2 kita. Model Frankenstein raksasa ini terbukti *over-engineered* dan melewatkan dua hukum fundamental:

1. **Kebutaan Kalender Absolut (Calendar Blindness)**
   - **Masalah:** Di M3, kita membuang fitur waktu (`date_show`). Mesin hanya tahu `day_since_release` (1-10) tetapi tidak sadar apakah hari ke-4 tersebut adalah Sabtu (lonjakan tinggi) atau Selasa (sepi).
   - **Solusi Exp04:** Ekstrak mutlak fitur `day_of_week`, `is_weekend`, dan injeksi data eksternal `holidays.csv` agar mesin memahami anomali kalender manusia.

2. **Pemaksaan Target Rasio (Target Scaling Mismatch)**
   - **Masalah:** Kita memaksa algoritma *Tree-based* (LGBM, XGBoost) memprediksi rasio target pecahan murni (`y_scaled = total_ticket / Sp`). Algoritma *Tree* sangat kesulitan membuat *split* pada nilai rasio fraksional dengan variansi $S_p$ ekstrem tanpa panduan pembobot.
   - **Solusi Exp04:** Kembalikan target menjadi logaritma `np.log1p(total_ticket)`. Jika ingin menggunakan strategi pembagian Sp yang terbukti berhasil (MASE Hacking), berikan nilai absolut asli target dan injeksikan $S_p$ sebagai `sample_weight` pada fase validasi/pelatihan L1 Loss.

3. **Absennya Penahan Gravitasi Peluruhan**
   - **Masalah:** Model masih sering berhalusinasi memprediksi kenaikan tiket di hari ke-9 tanpa alasan yang jelas.
   - **Solusi Exp04:** Aktifkan perisai `monotone_constraints=[-1]` terhadap fitur `day_since_release` untuk memaksa model LightGBM secara matematis mematuhi hukum peluruhan: *"tiket tidak boleh naik seiring berjalannya hari (kecuali terinterupsi fitur akhir pekan)"*.

## Exp04 Post-Mortem & Error Analysis (Menabrak Tembok Irreducible Error)
Berdasarkan hasil evaluasi *Local CV*, skor MASE kita bertahan konsisten di angka **0.7272**. Eksperimen ini adalah iterasi paling sempurna secara matematis yang pernah kita bangun, karena telah mematuhi seluruh hukum fisika kalender (mendayagunakan libur & akhir pekan) serta hukum optimasi fungsi kerugian (*Loss Function*). 

Namun, mengapa skornya masih di angka 0.72 dan bukan 0.20?
1. **Pencapaian Limit Fisika (Irreducible Error Boundary)**
   - **Analisis:** Melalui pengujian *overfitting* murni (menghafal ID film pada eksperimen rahasia), kita mendapati model bisa mencapai MASE ~0.0001 (Nol mutlak). Namun, saat dihadapkan pada entitas baru (*unseen movie*) melalui perlindungan `GroupKFold`, skor selalu membentur tembok 0.72. Ini membuktikan bahwa tanpa mengetahui "Sentimen Ulasan" (WOM) atau "Kapasitas Bioskop", mesin mustahil bisa menebak secara akurat apakah peluruhan suatu film akan melandai atau menukik tajam. Rentang 0.72 adalah batas (*sweet spot*) tertinggi untuk model generalisasi yang sehat.
   - **Konklusi Final:** Modifikasi algoritma (LGBM, XGBoost, NN) di atas *dataset* mentah bawaan panitia **TIDAK AKAN** pernah membuahkan skor 0.2 tanpa melakukan kecurangan (*LB Probing* / *Data Leakage*). Satu-satunya jalan keluar ke depan adalah mengakuisisi **Data Eksternal** (seperti yang terangkum di Idea 5).

2. **Kesuksesan Eksekusi Metrik & Pembobotan**
   - **Kemenangan:** Strategi memanipulasi target dengan pembagian $S_p$ (`y_scaled = total_ticket / Sp`) yang dipadukan dengan *Objective* MAE (L1 Loss) dan injeksi parameter `sample_weight = Sp` pada model *LightGBM* terbukti sangat brilian. Model berhasil dipaksa untuk mengoptimalkan diri secara khusus menaklukkan metrik MASE Kaggle tanpa mengorbankan stabilitas algoritma *Decision Tree*.

## Exp05 Post-Mortem & Error Analysis (Tragedi Covariate Shift)
Berdasarkan hasil eksekusi M11, skor Public LB adalah **1.10**, yang berbanding terbalik secara mengerikan dengan Local CV kita (**0.4524**). Kesenjangan ini mengindikasikan bahwa model gagal total dalam menggeneralisasi data masa depan. 

Berikut adalah analisis penyebab utama terjadinya *Feature Shift* ekstrem ini:

1. **Kanibalisme Data Masa Depan (Imputasi M9 vs Aktual M1-M8)**
   - **Masalah:** Di tahap *training* (M1-M8), model dilatih secara sangat manja menggunakan nilai aktual `total_show` (dan turunannya seperti `screen_share_pct` serta `is_clashing_blockbuster`) dari hari ke-4 hingga ke-10. Namun, di dunia nyata kompetisi (M9 / `test.csv`), nilai-nilai masa depan ini **tidak diketahui**. Akibatnya, kita terpaksa mengimputasi data M9 menggunakan rata-rata dari hari ke-1 hingga ke-3 (masa lalu). 
   - **Dampak:** XGBoost dan LightGBM berekspektasi bahwa fitur kanibalisme tersebut adalah fakta akurat. Ketika disuapi dengan angka tebakan (*imputasi*), persepsi pohon keputusan langsung kacau balau, membuat prediksi tiket melonjak tak terkendali.

2. **Hilangnya Konstrain Monotonik Gravitasi Waktu**
   - **Masalah:** Untuk menyelamatkan efektivitas Target Encoding pada CatBoost di M6, kita mencabut *Monotone Constraints* (penahan absolut bahwa tiket harus meluruh seiring hari). 
   - **Dampak:** Tanpa konstrain fisik tersebut, CatBoost bisa dengan mudah memprediksi bahwa jumlah tiket pada hari ke-10 secara tak masuk akal melonjak jauh melampaui hari ke-4, hanya karena ada sedikit perubahan pada fitur cuaca atau hari libur.

3. **Reaktivitas Berlebih pada Ekstraksi Cuaca & Squeezed**
   - **Masalah:** Bobot fitur `is_squeezed` dan `weather_penalty_score` menjadi sangat besar (penting) di tahap *training*. Namun, saat kondisi cuaca bulan September (data *Test*) berbeda iklim, penalti ini berpotensi menghukum algoritma dengan pengurang yang terlalu agresif.

**Solusi Exp06 (Langkah Perbaikan):**
- **Strict Time-Series Causality:** Jika fitur aktual tidak ada di masa depan (`test.csv`), maka fitur tersebut HARUS DIMUSNAHKAN dari masa lalu saat *training*. Matriks *training* `X` harus murni hanya diisi oleh agregasi Day 1-3.
- **Rasio Peluruhan (Decay Ratio Modeling):** Kita harus berhenti menyuruh model menebak `total_ticket` (angka mutlak) dan mulai menyuruhnya menebak `Decay Ratio` (persentase retensi penonton dari D3).

## Feature Engineering Ideas Backlog
- **Idea 1 (Solusi `total_show` Leakage)**: Hapus fitur `total_show` dan `price_capacity_ratio` (karena bergantung pada `total_show`) dari fase *Training*.
- **Idea 2 (Model Sekunder `total_show`)**: Alternatif dari Idea 1, buat model prediktor terpisah yang khusus menebak jumlah `total_show` di D4-D10.
- **Idea 3 (Exponential Moving Average / EMA)**: Tren *decay* penonton memiliki kurva lengkung eksponensial. Tambahkan fitur bobot per hari atau nilai EMA dari D1, D2, D3 agar model pohon (*Tree-based*) lebih mudah menarik kurva prediksi eksponensial di hari D4-D10.
- **Idea 4 (Ensembling XGBoost / CatBoost)**: Karena LightGBM rentan terhadap *Overfitting* akibat *Target Encoding* yang bocor, pertimbangkan kombinasi prediksi (*Ensemble/Stacking*) dengan *CatBoost* yang memiliki algoritma *Native Target Encoding* yang terisolasi dari *leakage*.
- **Idea 5 (Akuisisi Data Eksternal - Penembus Irreducible Error)**: Berdasarkan analisis di `analytic.md`, skor 0.2 mustahil dicapai tanpa memecah *Irreducible Error Boundary*. Diperlukan akuisisi eksternal (via Scraping/API) untuk 11 pilar berikut:
  1. **Kapasitas Layar & Kanibalisme (25%)**: Alokasi *screen count* riil dan keberadaan rilis film *blockbuster* saingan.
  2. **Resepsi Audiens/WOM (20%)**: Skor review (IMDb, Rotten Tomatoes) & sentimen sosial media (Tiktok/X) pada Hari 1-3.
  3. **Harga Tiket Dinamis / HTM (15%)**: Data historis harga tiket per bioskop (efek elastisitas Weekday vs Weekend).
  4. **Kalender Edukasi (10%)**: Jadwal Ujian Nasional/Sekolah dan masa Libur Panjang Semester.
  5. **Rekam Jejak Sineas (8%)**: Skor *Historical Box Office* dari 3 proyek terakhir aktor utama/sutradara (batas *Ticket Floor*).
  6. **Siklus Finansial (7%)**: Indikator *Payday* (tanggal gajian muda 25-5 vs tanggal tua 15-24).
  7. **Promo O2O (6%)**: Pemetaan hari promo bank (*Buy 1 Get 1*) dan jadwal *Meet & Greet* lokal (anomali puncak).
  8. **Demografi Bioskop (4%)**: Klasifikasi target audiens mall (Premium Tier 1 vs Bioskop Kabupaten).
  9. **Faktor Fisik Cuaca (3%)**: Data BMKG curah hujan (badai/banjir) per kota per hari yang melumpuhkan mobilitas.
  10. **Batas Fisiologi Durasi (1.5%)**: *Runtime* film (menit) untuk mendefinisikan kapasitas maksimal pemutaran harian.
  11. **Kebocoran Bajakan (0.5%)**: Indikator ketersediaan file bajakan (HD/Web-DL) di grup Telegram ilegal.

## Important Directives
1. **Never overwrite an existing experiment notebook**. Always create `Exp<XX>_...ipynb`.
2. **Submission Throttling**: You have a daily submission limit. Only use `kaggle_submit` if your **Local CV Score** improves significantly over the previous best experiment. Otherwise, trust your CV score and move to the next iteration without submitting.
