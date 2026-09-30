# Ringkasan Kompetisi (COMP_SUMMARY.md)

Dokumen ini merupakan intisari *highlight* dari `COMP.md` yang merangkum hal-hal paling krusial untuk diprioritaskan selama pengerjaan kompetisi JOINTS X INSPIRE UGM 2026.

## 1. Highlight Objektif & Masalah
- **Masalah**: Memprediksi permintaan tiket bioskop secara spesifik untuk kombinasi **Film dan Klaster Bioskop (lokasi)**. Skala nasional tidak relevan di sini.
- **Tugas Utama**: Menggunakan data penjualan 3 hari pertama (**D1-D3**) untuk memprediksi secara akurat penjualan tiket pada 7 hari berikutnya (**D4-D10**).
- **Format Target**: Kita memprediksi `total_ticket` per baris `id` yang merepresentasikan suatu tanggal target per bioskop per film.

## 2. Highlight Evaluasi & Aturan Kritis
- **Metric**: **Mean Absolute Scaled Error (MASE)**.
  - Ini adalah variasi dari *Mean Absolute Error (MAE)*, di mana *error* dibagi dengan "Skala" ($s_p$).
  - Skala dihitung dari rata-rata penjualan tiket D1-D3 (dengan minimum batas bawah/ *clip* bernilai 1).
  - Implikasi Model: Model bisa dilatih menggunakan objektif MAE standar, lalu performanya divalidasi lokal menggunakan *custom function* hitung MASE.
- **ATURAN DISKUALIFIKASI (Reproducibility)**:
  - Peserta **wajib** menggunakan *fixed seed* atau penentu acak yang statis (contoh: `random_state = 2026`).
  - Hasil panitia harus **100% klop/sama persis** dengan yang tersubmit di *leaderboard* Kaggle. Pastikan seluruh pipeline (split data, inisialisasi model, *sampling*) memiliki variabel acak yang dikunci dengan seed tersebut.

## 3. Highlight Dataset
Terdapat 7 file data dengan peran yang sangat berbeda:

### A. Data Prediksi (Test Set)
- **`test.csv` (72.611 baris)**: Berisi kombinasi Film dan Bioskop pada tanggal D4-D10 yang jumlah penjualan tiketnya *harus kita tebak*. Terdapat 160 Film unik dan 121 Bioskop unik di dalamnya.
- **`test_history.csv` (32.323 baris)**: Ini adalah fitur riwayat (D1-D3) resmi untuk set pengujian. Di sinilah kita mendapatkan angka aktual untuk menghitung $s_p$ (Skala) dan fitur *lag* sebelum menebak D4-D10 pada `test.csv`.

### B. Data Pelatihan (Train Set Mentah)
- **`train.csv` (138.959 baris)**: Ini bukan tabel siap latih. Ini adalah daftar transaksi harian *mentah* dari 1 April - 30 September 2025. 
  - **Tantangan Utama**: Kita harus membuat pendekatan sendiri untuk memotong data panjang ini menjadi pasangan-pasangan jendela D1-D3 (sebagai input/fitur) dan D4-D10 (sebagai label latih) agar meniru format pengujian.

### C. Data Pelengkap (External Metadata)
Bisa digabungkan untuk memperkaya akurasi (Feature Engineering):
1. **`movies.csv`**: Mengandung karakteristik film (Genre, Rating Usia, Sutradara, dll). *Note: Ada sedikit data missing & nilai strip (`-`) yang butuh di-handle*.
2. **`holidays.csv`**: Kalender penayangan, menandakan akhir pekan (*weekend*) atau hari libur nasional (*holiday*). Ini merupakan **fitur terpenting** karena penjualan tiket bioskop sangat bergantung pada kalender liburan.
3. **`ticket_prices.csv`**: Detail harga di berbagai kota untuk *Weekday*, *Friday*, dan *Weekend*. Semakin tinggi/mahal tiket bisa mempengaruhi daya beli.

## 4. Prioritas Aksi Selanjutnya
1. Ekstraksi dan merakit data mentah `train.csv` menjadi format (*D1-D3 Fitur $\rightarrow$ D4-D10 Label*) menggunakan *Sliding Window*.
2. Konstruksi fungsi validasi MASE lokal.
3. Penyelenggaraan pipeline dengan seed yang terkunci rapat (`SEED = 2026`).

## 5. Ringkasan Input & Output
- **Input Inferensi (Test Set)**: 
  - Kombinasi unik dari: `movie_title` dan `cinema_ids`.
  - Bertepatan dengan target prediksi: `date_show` yang dituju (Hari ke-4 s.d. Hari ke-10).
  - Diperkaya fitur tambahan (dari proses rekayasa): Historis jumlah penjualan pada D1-D3, metadata film (`genre`, `age_rating`), kalender libur/akhir pekan (`holidays.csv`), dan tarif dasar tiket kota/hari tersebut (`ticket_prices.csv`).
- **Output yang Diharapkan (Prediksi)**:
  - Estimasi volume tiket terjual (`total_ticket`) untuk target kombinasi (D4-D10). Output adalah bilangan $\ge 0$.

## 6. Rekomendasi Proses Eksperimen
1. **Pembangunan Strategi Validasi (CV)**:
   - Mengingat kompetisi ini bersifat musiman (*time-dependent*) antar-tanggal rilis film, K-Fold acak biasa akan menyebabkan kebocoran data (*data leakage*).
   - *Rekomendasi*: Terapkan *Group K-Fold* berdasarkan `movie_title` (sehingga model tidak menghafal tren suatu film secara parsial) atau skema *Time-Series Split* agar validasi mensimulasikan masa lalu secara utuh.
2. **Pola Rekayasa Fitur (Feature Engineering)**:
   - *Lag & Trend Features*: Ekstrak selisih pertumbuhan D2 $\rightarrow$ D1 dan D3 $\rightarrow$ D2 untuk memahami awal popularitas (Apakah sedang *hype*? Apakah menurun?).
   - Masukkan nilai rata-rata D1-D3 secara eksplisit ke model, mengingat nilai itu juga merupakan pembagi (Skala/$s_p$) dalam metrik MASE. Model dapat lebih stabil menangkap ekspektasi dasar volumenya.
   - Kombinasi efek waktu & harga tiket: Flag liburan (`is_holiday`), hari ke-X penayangan, dan status *weekend* memiliki pengaruh absolut.
3. **Penyusunan Pipeline Model & Objektif**:
   - Algoritma berbasis pohon (seperti **LightGBM**, **XGBoost**, atau **CatBoost**) disarankan karena andal tanpa perlu normalisasi (*scaling*) pada *tabular data* berskala lebar, serta baik menangani ID bioskop (*categorical processing* bawaan).
   - Set objektif *Loss Function* menjadi **MAE** atau **Huber Loss**, karena keduanya mengejar nilai median absolut (selaras dengan dasar evaluasi MASE).
   - Rancang skrip eval metric MASE murni pada fungsi pelatihan untuk proses pemantauan metrik secara spesifik.
