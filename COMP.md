# Data Science Competition JOINTS X INSPIRE UGM 2026
Seberapa jauh tiga hari pertama penayangan sebuah film dapat memberi gambaran tentang minggu berikutnya?

## Overview
Selamat datang di Data Science Competition JOINTS 2026! Sebagai salah satu rangkaian acara JOINTS 2026 yang diselenggarakan oleh Departemen Ilmu Komputer dan Elektronika UGM, kompetisi ini bertujuan untuk memberikan pengalaman bagi peserta dalam menyelesaikan permasalahan berbasis data menggunakan pendekatan machine learning dan analisis data. Peserta akan ditantang untuk menguasai siklus data science secara end-to-end, mulai dari pemrosesan data, eksplorasi data (EDA), rekayasa fitur, hingga pemodelan dan evaluasi. Peserta juga diminta untuk berpikir secara analitis dalam mengekstrak wawasan berharga dari data serta menafsirkannya dengan tepat dan terstruktur.

## Tujuan Kompetisi

### Problem Statement
Operator bioskop perlu memperkirakan permintaan tiket beberapa hari ke depan untuk membantu menentukan alokasi layar, jumlah pertunjukan, serta kebutuhan operasional di tiap lokasi. Keputusan ini harus dibuat ketika film baru memiliki sedikit riwayat penjualan, sementara pola permintaan dapat berbeda antarfilm dan antarklaster bioskop.

Film dengan pembukaan yang kuat belum tentu mempertahankan pola yang sama pada hari-hari berikutnya. Sebaliknya, film dengan penjualan awal yang rendah masih dapat mengalami perubahan permintaan karena akhir pekan, kalender, atau perkembangan minat penonton. Perbedaan skala dan karakter tiap klaster bioskop membuat perkiraan tingkat nasional saja tidak cukup untuk mendukung keputusan di tingkat lokasi.

### Objektif
Dalam kompetisi ini, peserta akan menggunakan data penjualan tiket dari tiga hari pertama data penjualan tiket selama periode observasi pada hari ke-1 hingga hari ke-3 untuk memprediksi penjualan harian pada hari ke-4 hingga hari ke-10. Setiap prediksi dibuat untuk kombinasi film dan klaster bioskop, sehingga pola penjualan film dan perbedaan karakter tiap lokasi sama-sama berperan.

## Evaluation
Prediksi dinilai menggunakan Mean Absolute Scaled Error (MASE).

Misalkan p adalah satu pasangan film dan klaster bioskop. Skala untuk pasangan tersebut dihitung dari rata-rata penjualan tiket pada tiga hari pertama:

$$ s_p = \max\left( \frac{1}{3} \sum_{d=1}^3 y_{p,d}, 1 \right) $$

dengan $y_{p,d}$ sebagai jumlah tiket aktual pasangan p pada hari ke-d.

MASE dihitung sebagai rata-rata kesalahan absolut yang dibagi dengan skala pasangan masing-masing:

$$ \text{MASE} = \frac{1}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{s_{p(i)}} $$

dengan:
- $N$ adalah jumlah seluruh baris pada data evaluasi.
- $y_i$ adalah jumlah tiket aktual pada baris ke-i.
- $\hat{y}_i$ adalah jumlah tiket hasil prediksi.
- $p(i)$ adalah pasangan film dan klaster bioskop pada baris ke-i.
- $s_{p(i)}$ adalah skala pasangan tersebut, yang sama untuk seluruh tujuh hari prediksinya.

```python
from sklearn.metrics import mean_absolute_error

def hitung_skala(history):
    """ 
    history: tiga hari observasi (D1-D3) untuk tiap pasangan klaster-film
    """
    total = history.groupby(["movie_title", "cinema_ids"]).total_ticket.sum()
    return (total / 3).clip(lower=1).rename("scale")

def mean_absolute_scaled_error(y_true, y_pred, scale):
    """
    y_true: jumlah tiket terjual
    y_pred: forecast tiket terjual
    scale: rata-rata tiket terjual di tiga hari pertama
    """
    return mean_absolute_error(y_true / scale, y_pred / scale)

scale = target.join(hitung_skala(history), on=["movie_title", "cinema_ids"]).scale
skor = mean_absolute_scaled_error(y_true, y_pred, scale)
```

## Submission File
File submission harus memiliki kolom berikut:

```csv
id,total_ticket
1,100
2,101
3,102
...
```

## Extras

### Pastikan Submisi Bersifat Reproducible
Dalam melakukan validasi hasil submisi tiap peserta, panitia akan mengecek apakah hasil prediksi dapat dibuat ulang. Oleh karena itu, peserta diwajibkan untuk menetapkan konstanta random_state yang akan digunakan pada bagian yang membutuhkan randomness number pada pengerjaan. Kejanggalan (disparitas nilai yang besar antara hasil prediksi di Kaggle dengan hasil produksi ulang oleh panitia dapat mengakibatkan peserta yang bersangkutan didiskualifikasi. Berikut adalah potongan kode contoh penetapan random_state untuk mendapatkan hasil yang mampu dibuat ulang (reproducible).

```python
SEED = 2026
LABEL = "label"

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

X, y = train.drop(LABEL, axis=1), train[LABEL]

## Penetapan seed pada `random_state` untuk memastikan konsistensi hasil prediksi model
model = lightgbm.LGBMClassifier(random_state = SEED)
model.fit(X, y)

y_pred = model.predict(test)
```

## Data

### Berkas
| Berkas | Baris | Keterangan |
| --- | --- | --- |
| `train.csv` | 138.959 | Riwayat transaksi harian dari 1 April sampai 30 September 2025 |
| `test_history.csv` | 32.323 | Transaksi yang tercatat selama D1–D3 film uji |
| `test.csv` | 72.611 | Kombinasi film, klaster, kota, dan tanggal yang perlu diprediksi |
| `sample_submission.csv` | 72.611 | Template file pengumpulan |
| `movies.csv` | 397 | Informasi metadata film |
| `holidays.csv` | 366 | Kalender hari dan hari libur |
| `ticket_prices.csv` | 207 | Daftar harga tiket per kota |

Untuk setiap film uji, D1–D3 adalah tiga tanggal kalender berturut-turut yang berlaku sama di seluruh klaster bioskop. D1 tidak selalu merupakan tanggal pertama film tersebut memiliki transaksi di suatu klaster. D4–D10 adalah tujuh tanggal kalender berikutnya.

Setiap pasangan film dan klaster yang muncul di `test.csv` memiliki tujuh baris prediksi, yaitu D4–D10. Jika pada salah satu tanggal target tidak ada transaksi, jumlah tiket aktualnya adalah nol.

`train.csv` merupakan riwayat transaksi, bukan tabel contoh latih yang sudah diberi penanda D1–D10. Peserta dapat mengolah riwayat tersebut untuk membangun pendekatan dan validasi masing-masing.

### Kolom Data

#### `train.csv` dan `test_history.csv`
| Kolom | Keterangan |
| --- | --- |
| `date_show` | Tanggal penayangan dalam format YYYY-MM-DD |
| `cinema_ids` | ID anonim klaster bioskop |
| `city_name` | Kota lokasi klaster |
| `movie_title` | Judul film, termasuk penanda format jika tercatat |
| `total_ticket` | Jumlah tiket yang terjual pada tanggal tersebut |
| `occupation_rate` | Persentase kursi yang terisi, dari 0 sampai 100 |
| `total_show` | Jumlah pertunjukan yang terlaksana |

#### `test.csv`
Berisi `id`, `movie_title`, `cinema_ids`, `city_name`, dan `date_show`. Kolom `date_show` menunjukkan tanggal D4–D10 yang perlu diprediksi.

#### `sample_submission.csv`
Berisi `id` dan `total_ticket`. Nilai `total_ticket` di berkas ini hanya contoh pengisian, bukan prediksi acuan.

#### `movies.csv`
Berisi `original_title`, `age_rating`, `genre`, `producer`, `director`, `writer`, dan `casts`.

#### `holidays.csv`
Berisi `date`, `day_tipe` (weekday, friday, atau weekend), `holiday_tipe` (normal atau holiday), dan `holiday_name`.

#### `ticket_prices.csv`
Berisi `city_name`, `ceil` sebagai harga tiket dalam rupiah, dan `price_day` untuk kategori Weekday, Friday, atau Weekend.

### Metadata
- **Files**: 7 files
- **Size**: 21.2 MB
- **Type**: csv

### Detail Dataset: `holidays.csv`
Berdasarkan eksplorasi data, file ini memiliki **366 baris** (mencakup 1 tahun, mulai dari 1 April 2025 hingga 1 April 2026) dan **4 kolom**, dengan rincian:
1. **`date`**: Tanggal (Mulai dari 2025-04-01 sampai 2026-04-01). 100% valid (0% missing).
2. **`day_tipe`**: Tipe hari. Terdapat 3 nilai unik: `weekday` (57%), `weekend` (28%), dan lainnya (kemungkinan `friday` - 14%). 100% valid.
3. **`holiday_tipe`**: Tipe liburan. Terdapat 2 nilai unik: `normal` (95%) dan `holiday` (5%). 100% valid.
4. **`holiday_name`**: Nama hari libur. Terdapat 17 nilai unik (contoh: "Idulfitri 1446 H", "Idulfitri 1447 H"). Hanya 5% valid (18 baris), sementara 95% (348 baris) sisanya bernilai *missing* (`[null]`).

### Detail Dataset: `movies.csv`
Berdasarkan eksplorasi data, file ini memiliki **397 baris** (mewakili 397 film) dan **7 kolom**, dengan rincian:
1. **`original_title`**: Judul asli film. Terdapat 397 nilai unik. 100% valid (0% missing).
2. **`age_rating`**: Klasifikasi usia penonton (4 nilai unik). Didominasi oleh kategori `Remaja` (44%) dan `Dewasa` (42%). Terdapat sebagian kecil (1% atau 3 baris) *missing values*.
3. **`genre`**: Genre film (123 nilai unik). Genre terbanyak adalah `Horror` (17%) dan `Drama` (9%). 100% valid.
4. **`producer`**: Produser film (330 nilai unik). 100% valid (0% missing), namun ada sekitar 4% data yang hanya berisi tanda hubung (`-`).
5. **`director`**: Sutradara film (330 nilai unik). 100% valid.
6. **`writer`**: Penulis naskah film (349 nilai unik). 100% valid, namun sekitar 5% data diisi dengan tanda hubung (`-`).
7. **`casts`**: Daftar pemeran film (395 nilai unik). 100% valid.

### Detail Dataset: `sample_submission.csv`
Berdasarkan eksplorasi data, file ini memiliki **72.611 baris** dan **2 kolom**, yang berfungsi murni sebagai *template* submission:
1. **`id`**: Identifier baris (urut dari 1 hingga 72.611). 100% valid.
2. **`total_ticket`**: Kolom target untuk hasil prediksi. Di dalam file contoh ini, seluruh baris (100%) diisi dengan nilai *dummy* yaitu `100.0`.

### Detail Dataset: `test.csv`
Berdasarkan eksplorasi data, file ini memiliki **72.611 baris** dan **5 kolom**, berisi daftar kombinasi film dan bioskop beserta tanggal target prediksi (Hari ke-4 s.d. Hari ke-10):
1. **`id`**: Identifier unik baris (sama dengan di `sample_submission.csv`). 100% valid.
2. **`movie_title`**: Judul film (terdapat 160 nilai unik). 100% valid (0% missing).
3. **`cinema_ids`**: ID klaster bioskop (terdapat 121 nilai unik). 100% valid.
4. **`city_name`**: Nama kota (terdapat 59 nilai unik). Distribusi terbanyak berada di `JAKARTA` (4%) dan `TANGERANG` (4%). 100% valid.
5. **`date_show`**: Tanggal tayang yang perlu diprediksi (rentang dari 4 Oktober 2025 hingga 27 Maret 2026). 100% valid.

### Detail Dataset: `test_history.csv`
Berdasarkan eksplorasi data, file ini memiliki **32.323 baris** dan **7 kolom**, merepresentasikan data riwayat transaksi penjualan tiket pada 3 hari pertama (D1-D3) untuk kombinasi film dan bioskop yang diuji:
1. **`date_show`**: Tanggal riwayat penayangan (mulai 1 Oktober 2025 hingga 20 Maret 2026). 100% valid.
2. **`cinema_ids`**: ID klaster bioskop (terdapat 121 nilai unik). 100% valid.
3. **`city_name`**: Nama kota lokasi bioskop (terdapat 59 nilai unik). 100% valid.
4. **`movie_title`**: Judul film (terdapat 160 nilai unik). 100% valid.
5. **`total_ticket`**: Jumlah tiket terjual pada hari tersebut. (Distribusi menceng ke kanan / *right-skewed*). 100% valid.
6. **`occupation_rate`**: Persentase kursi yang terisi pada hari tersebut (rentang 0 hingga 100). 100% valid.
7. **`total_show`**: Jumlah tayangan/pertunjukan pada hari tersebut (nilai maksimal mencapai 278). 100% valid.

### Detail Dataset: `ticket_prices.csv`
Berdasarkan eksplorasi data, file ini memiliki **207 baris** dan **3 kolom**, mendetailkan daftar harga tiket per kota:
1. **`city_name`**: Nama kota (terdapat 69 nilai unik). 100% valid.
2. **`ceil`**: Harga tiket bioskop dalam rupiah. Berkisar antara 20.000 hingga 70.000, dengan rata-rata 45.100. 100% valid.
3. **`price_day`**: Kategori hari untuk harga tiket (terdapat 3 nilai unik: `Weekday`, `Friday`, dan `Weekend`). 100% valid.

### Detail Dataset: `train.csv`
Berdasarkan eksplorasi data, file ini memiliki **138.959 baris** dan **7 kolom**, merepresentasikan riwayat transaksi harian secara lengkap mulai dari 1 April hingga 30 September 2025:
1. **`date_show`**: Tanggal penayangan riwayat (1 April 2025 - 30 September 2025). 100% valid.
2. **`cinema_ids`**: ID klaster bioskop (terdapat 126 nilai unik). 100% valid.
3. **`city_name`**: Nama kota lokasi bioskop (terdapat 60 nilai unik). Distribusi terbanyak berada di `JAKARTA` (6%) dan `TANGERANG` (5%). 100% valid.
4. **`movie_title`**: Judul film (terdapat 337 nilai unik). 100% valid.
5. **`total_ticket`**: Jumlah tiket terjual (nilai maksimal mencapai 32.800 tiket). Distribusinya sangat menceng ke kanan (*right-skewed*). 100% valid.
6. **`occupation_rate`**: Persentase kursi terisi (skala 0 hingga 100). 100% valid.
7. **`total_show`**: Jumlah pertunjukan harian (nilai maksimal mencapai 285 pertunjukan). 100% valid.
