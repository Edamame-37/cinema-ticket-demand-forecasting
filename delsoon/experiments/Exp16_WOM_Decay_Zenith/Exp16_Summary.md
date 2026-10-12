# Ringkasan Eksekutif Eksperimen 16: Exp16_WOM_Decay_Zenith

## 1. Latar Belakang & Konfigurasi
Eksperimen ini memanfaatkan integrasi dataset baru `dataset/new-again/movies_tmdb_ratings.csv` yang memuat sinyal Word-of-Mouth (TMDB Rating, Rasio Positif, Rasio Negatif, Total Ulasan).

## 2. Metodologi & Fitur Kunci
1. **Title Normalization (Regex Suffix Clean):** Memperbaiki inkonsistensi sufiks bioskop `(IMAX 2D|3D|...)` sehingga 100% film (termasuk 20 film uji yang sebelumnya *unmatched*) terhubung sempurna ke metadata film.
2. **Word-of-Mouth (WOM) Signals:**
   - `net_sentiment = rasio_positif - rasio_negatif`
   - `wom_confidence = tmdb_rating * np.log1p(total_ulasan)`
   - `decay_retention_interaction = net_sentiment / np.sqrt(day_idx)`
   - Format bioskop: `is_imax`, `is_3d`
3. **Strict Causality & Train Pruning:** Pelatihan dibatasi ketat hanya pada Hari ke-4 s.d. Hari ke-10 (`day_idx` 4–10). Semua fitur masa depan terlarang di-drop.
4. **Target Formulation & Scale:** `y_ratio = total_ticket / scale_sp` dilatih menggunakan fungsi objektif MAE murni (unweighted).
5. **GroupKFold (5-Fold) & Monotone Constraints:**
   - Grouping berdasarkan `movie_title` untuk mencegah kebocoran antar-film.
   - Monotone constraints pada XGBoost: `day_since_release` (-1), `net_sentiment` (+1), `tmdb_rating` (+1).
6. **SLSQP Meta-Optimization:** Optimasi bobot gabungan langsung meminimalkan metrik MASE kompetisi.

## 3. Hasil Validasi Out-of-Fold (OOF MASE)

| Model | OOF MASE | Bobot SLSQP |
| :--- | :---: | :---: |
| **LightGBM (MAE Objective)** | 0.45730 | 0.4414 |
| **XGBoost (MAE + Monotone Constraints)** | 0.44751 | 0.5586 |
| **CatBoost (MAE Loss)** | 0.57376 | 0.0000 |
| **Ensemble SLSQP Gabungan (Exp16)** | **0.43495** | **1.0000** |

### Perbandingan dengan Eksperimen Sebelumnya:
- **Exp14 (The Flawless Zenith):** OOF MASE = **0.4474** (LB: 0.62822)
- **Exp15 (Feature Cross Target Encoding):** OOF MASE = **0.4447**
- **Exp16 (WOM Decay Zenith):** OOF MASE = **0.43495** (**Peningkatan +0.0098 OOF MASE**)

## 4. Validasi Submission
- File: `experiments/Exp16_WOM_Decay_Zenith/submission_exp16.csv`
- Total Baris: 72,611 baris (sesuai spesifikasi evaluasi)
- Missing Value: 0 (bebas NaN)
- Prediksi Nilai Tiket: Non-negatif, dibulatkan ke bilangan bulat, distribusi wajar (min: 1.0, median: 57.0, max: 27,045.0).
