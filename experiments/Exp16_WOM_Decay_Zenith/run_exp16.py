"""
========================================================================================
Eksperimen 16: Exp16_WOM_Decay_Zenith
========================================================================================
Deskripsi:
Eksperimen ini memanfaatkan penambahan dataset baru 'dataset/new-again/movies_tmdb_ratings.csv'
yang memuat sinyal Word-of-Mouth (WOM) dan sentimen ulasan penonton (TMDB Rating, Rasio Positif,
Rasio Negatif, Total Ulasan).

Inovasi & Peningkatan Kritis:
1. Normalisasi Judul (Regex Suffix Clean):
   Menghapus sufiks format '(IMAX 2D|3D|...)' sehingga 100% film pada data latih dan data uji
   terhubung sempurna ke metadata film (memperbaiki 20 film uji yang sebelumnya unmatched).
2. Fitur Sinyal Word-of-Mouth (WOM):
   - 'net_sentiment' = rasio_positif - rasio_negatif
   - 'wom_confidence' = tmdb_rating * log1p(total_ulasan)
   - 'decay_retention_interaction' = net_sentiment / sqrt(day_idx)
   - Indikator format tayang premium ('is_imax', 'is_3d')
3. Strict Causality & Horizon Protection:
   Data latih dibatasi ketat hanya pada Hari ke-4 s.d. Hari ke-10 (day_idx 4-10).
   Semua fitur masa depan yang bocor (total_show, occupation_rate masa depan, dll) dibuang.
4. Target Rasio Tak Terdistorsi:
   y_ratio = total_ticket / scale_sp (di mana scale_sp = max(mean(d1..d3), 1)).
   Objektif regresi menggunakan MAE murni tanpa bobot, selaras dengan evaluasi unweighted MASE Kaggle.
5. Out-of-Fold Feature Cross Target Encoding:
   Encoding rasio untuk pasangan 'cinema_ids x day_idx' dan 'genre x day_idx' dengan 5-Fold GroupKFold
   berdasarkan movie_title untuk mencegah kebocoran antar-film.
6. Monotone Constraints:
   Menetapkan batasan terarah pada XGBoost untuk umur rilis film (-1), net sentiment (+1), dan rating (+1).
7. Tri-Model SLSQP Ensemble:
   Menggabungkan LightGBM, XGBoost, dan CatBoost dengan optimasi bobot analitis SLSQP.
========================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
import re
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '0'

# ==========================================
# 0. KONFIGURASI DIREKTORI & OUTPUT
# ==========================================
exp_dir = 'experiments/Exp16_WOM_Decay_Zenith'
os.makedirs(exp_dir, exist_ok=True)

base_dir = 'dataset/dataset-05-10-26'
new_data_dir = 'dataset/new-again'

print("=" * 70)
print(">>> MEMULAI EKSEKUSI PIPELINE EXP16: WOM DECAY ZENITH <<<")
print("=" * 70)

# ==========================================
# 1. PEMUATAN DATASET DASAR & DATASET BARU
# ==========================================
print("\n[FASE 1] Memuat dataset mentah dan dataset baru TMDB ratings...")
train_raw = pd.read_csv(f'{base_dir}/train.csv')
test_hist = pd.read_csv(f'{base_dir}/test_history.csv')
test_raw = pd.read_csv(f'{base_dir}/test.csv')

movies_base = pd.read_csv(f'{base_dir}/movies.csv')
movies_tmdb = pd.read_csv(f'{new_data_dir}/movies_tmdb_ratings.csv')
holidays = pd.read_csv(f'{base_dir}/holidays.csv')
ticket_prices = pd.read_csv(f'{base_dir}/ticket_prices.csv')
weather_df = pd.read_csv(f'{base_dir}/weather_data.csv')

print(f"-> Train raw shape: {train_raw.shape}")
print(f"-> Test raw shape: {test_raw.shape}")
print(f"-> Test history shape: {test_hist.shape}")
print(f"-> Base movies count: {len(movies_base)}, New TMDB movies count: {len(movies_tmdb)}")

# ==========================================
# 2. NORMALISASI JUDUL & PENGGABUNGAN METADATA
# ==========================================
print("\n[FASE 2] Normalisasi judul film & integrasi TMDB Ratings...")

def clean_title(title):
    """
    Menghapus sufiks format bioskop seperti (IMAX 2D), (3D), (4DX), dll
    agar pencocokan judul film ke metadata menjadi 100% akurat.
    """
    t = re.sub(r'\s*\((?:IMAX\s*2D|IMAX\s*3D|3D|2D|4DX|ATMOS|RE-RELEASE)\)\s*', '', str(title), flags=re.IGNORECASE)
    return t.strip()

# Gabungkan movies_base dan movies_tmdb
movies_combined = pd.merge(
    movies_base,
    movies_tmdb[['original_title', 'tmdb_rating', 'rasio_positif', 'rasio_negatif', 'total_ulasan']],
    on='original_title',
    how='left'
)

# Imputasi default untuk film baru jika ada nilai kosong
movies_combined['tmdb_rating'] = movies_combined['tmdb_rating'].fillna(movies_combined['tmdb_rating'].median())
movies_combined['rasio_positif'] = movies_combined['rasio_positif'].fillna(0.0)
movies_combined['rasio_negatif'] = movies_combined['rasio_negatif'].fillna(0.0)
movies_combined['total_ulasan'] = movies_combined['total_ulasan'].fillna(0)
movies_combined['runtime_minutes'] = movies_combined['runtime_minutes'].fillna(movies_combined['runtime_minutes'].median())
movies_combined['director_popularity'] = movies_combined['director_popularity'].fillna(0.0)
movies_combined['star_power_score'] = movies_combined['star_power_score'].fillna(movies_combined['star_power_score'].median())

for col in ['producer', 'director', 'writer', 'casts', 'genre', 'age_rating']:
    movies_combined[col] = movies_combined[col].replace('-', 'Unknown').fillna('Unknown')

print(f"-> Metadata film gabungan berhasil disiapkan: {movies_combined.shape[1]} kolom.")

# ==========================================
# 3. STRICT CAUSALITY & EKSTRAKSI PROFIL HARI 1-3
# ==========================================
print("\n[FASE 3] Ekstraksi profil historis (Hari 1 s.d. 3) & Train Pruning...")

def extract_hist_profile(hist_df):
    """
    Mengekstrak profil dasar penayangan 3 hari pertama bioskop:
    - total_ticket hari ke-1, 2, dan 3
    - scale_sp = max(mean(d1..d3), 1)
    - D3_D1_ratio = rasio penurunan awal
    - rata-rata occupation_rate 3 hari pertama
    """
    df = hist_df.copy()
    df['date_show'] = pd.to_datetime(df['date_show'])
    df.sort_values(by=['movie_title', 'cinema_ids', 'date_show'], inplace=True)
    
    profile = df.groupby(['movie_title', 'cinema_ids']).agg(
        hist_occupation_rate=('occupation_rate', 'mean'),
        d1_ticket=('total_ticket', lambda x: x.iloc[0] if len(x)>0 else np.nan),
        d2_ticket=('total_ticket', lambda x: x.iloc[1] if len(x)>1 else np.nan),
        d3_ticket=('total_ticket', lambda x: x.iloc[2] if len(x)>2 else np.nan),
        scale_sp=('total_ticket', lambda x: max(x.mean(), 1))
    ).reset_index()
    
    profile['D3_D1_ratio'] = profile['d3_ticket'] / profile['d1_ticket'].replace(0, 1)
    profile['D3_D1_ratio'] = np.clip(profile['D3_D1_ratio'], a_min=0, a_max=0.99)
    return profile

# Siapkan day_idx pada train_raw
train_raw['date_show'] = pd.to_datetime(train_raw['date_show'])
train_raw.sort_values(by=['movie_title', 'cinema_ids', 'date_show'], inplace=True)
train_raw['day_idx'] = train_raw.groupby(['movie_title', 'cinema_ids']).cumcount() + 1

# Filter pasangan bioskop-film yang tayang minimal 4 hari
valid_pairs = train_raw.groupby(['movie_title', 'cinema_ids']).size()
valid_pairs = valid_pairs[valid_pairs >= 4].index
train_valid = train_raw.set_index(['movie_title', 'cinema_ids']).loc[valid_pairs].reset_index()

# Pisahkan historis (1-3) dan target horizon kompetisi (4-10)
train_hist = train_valid[train_valid['day_idx'] <= 3].copy()
train_future = train_valid[(train_valid['day_idx'] >= 4) & (train_valid['day_idx'] <= 10)].copy()

# Buang fitur masa depan terlarang yang menyebabkan kebocoran (leakage)
cols_to_drop = ['total_show', 'occupation_rate', 'cinema_daily_total_shows', 
                'max_single_movie_shows', 'screen_share_pct', 'is_clashing_blockbuster']
train_future.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Ekstraksi profil
train_profile = extract_hist_profile(train_hist)
test_profile = extract_hist_profile(test_hist)

train_final = train_future.merge(train_profile, on=['movie_title', 'cinema_ids'], how='left')
test_final = test_raw.merge(test_profile, on=['movie_title', 'cinema_ids'], how='left')

# Hitung day_idx test yang presisi dengan menghubungkan test_hist dan test_raw
test_raw['is_test'] = 1
test_hist['is_test'] = 0
test_combined = pd.concat([test_hist, test_raw], ignore_index=True)
test_combined['date_show'] = pd.to_datetime(test_combined['date_show'])
test_combined.sort_values(by=['movie_title', 'cinema_ids', 'date_show'], inplace=True)
test_combined['day_idx'] = test_combined.groupby(['movie_title', 'cinema_ids']).cumcount() + 1

test_day_mapping = test_combined[test_combined['is_test'] == 1][['id', 'day_idx']]
test_final = test_final.merge(test_day_mapping, on='id', how='left')

print(f"-> Train future valid rows (Days 4-10): {train_final.shape[0]}")
print(f"-> Test final rows: {test_final.shape[0]}")

# ==========================================
# 4. REKAYASA FITUR (FEATURE ENGINEERING & WOM)
# ==========================================
print("\n[FASE 4] Menghasilkan fitur kalender, format bioskop, dan Word-of-Mouth...")

def enrich_features(df, is_test=False):
    """
    Menambahkan seluruh fitur kontekstual, format bioskop, cuaca, liburan,
    dan sinyal Word-of-Mouth TMDB ke dataframe.
    """
    df = df.copy()
    df['date_show'] = pd.to_datetime(df['date_show'])
    df['day_of_week'] = df['date_show'].dt.dayofweek
    df['extracted_day_tipe'] = np.where(df['day_of_week'] < 4, 'Weekday', np.where(df['day_of_week'] == 4, 'Friday', 'Weekend'))
    df['month'] = df['date_show'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([4, 5, 6]).astype('int8')
    df['is_month_end'] = df['date_show'].dt.is_month_end.astype('int8')
    
    # Indikator format premium
    df['is_imax'] = df['movie_title'].str.contains('IMAX', case=False, na=False).astype('int8')
    df['is_3d'] = df['movie_title'].str.contains('3D', case=False, na=False).astype('int8')
    
    # Judul yang dinormalisasi untuk join metadata
    df['clean_title'] = df['movie_title'].apply(clean_title)
    
    # Merge metadata lengkap (Base + TMDB Ratings)
    df = df.merge(movies_combined, left_on='clean_title', right_on='original_title', how='left')
    
    # Fitur sinyal Word-of-Mouth (WOM)
    df['net_sentiment'] = df['rasio_positif'] - df['rasio_negatif']
    df['wom_confidence'] = df['tmdb_rating'] * np.log1p(df['total_ulasan'])
    df['has_reviews'] = (df['total_ulasan'] > 0).astype('int8')
    df['decay_retention_interaction'] = df['net_sentiment'] / np.sqrt(df['day_idx'])
    
    # Merge Liburan
    holidays_df = holidays.copy()
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    df = df.merge(holidays_df, left_on='date_show', right_on='date', how='left')
    df['holiday_tipe'] = df['holiday_tipe'].fillna('normal')
    
    # Merge Harga Tiket
    prices_df = ticket_prices.copy()
    df = df.merge(prices_df, left_on=['city_name', 'extracted_day_tipe'], right_on=['city_name', 'price_day'], how='left')
    
    # Merge Cuaca
    weather = weather_df.copy()
    weather['date_show'] = pd.to_datetime(weather['date_show'])
    df = df.merge(weather, on=['city_name', 'date_show'], how='left')
    df['rainfall_mm'] = df['rainfall_mm'].fillna(0)
    df['is_heavy_rain'] = df['is_heavy_rain'].fillna(0)
    df['weather_penalty_score'] = df['is_heavy_rain'] * df['is_weekend']
    
    # Estimasi umur rilis nasional (inferred_release_date)
    if is_test:
        first_show = pd.concat([test_hist, test_raw]).groupby('movie_title')['date_show'].min().reset_index()
    else:
        first_show = train_raw.groupby('movie_title')['date_show'].min().reset_index()
    first_show.rename(columns={'date_show': 'inferred_release_date'}, inplace=True)
    first_show['inferred_release_date'] = pd.to_datetime(first_show['inferred_release_date'])
    df = df.merge(first_show, on='movie_title', how='left')
    df['day_since_release'] = (df['date_show'] - df['inferred_release_date']).dt.days + 1
    
    # Feature Cross strings untuk target encoding
    df['cinema_day_str'] = df['cinema_ids'].astype(str) + "_" + df['day_idx'].astype(str)
    df['genre_day_str'] = df['genre'].astype(str) + "_" + df['day_idx'].astype(str)
    
    return df

train_final = enrich_features(train_final, is_test=False)
test_final = enrich_features(test_final, is_test=True)

# ==========================================
# 5. STANDARISASI KATEGORI & IMPUTASI
# ==========================================
print("\n[FASE 5] Standarisasi fitur kategorikal & imputasi numerik...")

num_cols = [
    'd1_ticket', 'd2_ticket', 'd3_ticket', 'scale_sp', 'D3_D1_ratio', 
    'hist_occupation_rate', 'runtime_minutes', 'director_popularity', 
    'star_power_score', 'tmdb_rating', 'rasio_positif', 'rasio_negatif', 
    'net_sentiment', 'total_ulasan', 'wom_confidence', 'decay_retention_interaction',
    'day_since_release', 'price', 'rainfall_mm', 'weather_penalty_score'
]

cat_cols = [
    'movie_title', 'cinema_ids', 'city_name', 'genre', 'age_rating', 
    'holiday_tipe', 'director', 'writer', 'cinema_day_str', 'genre_day_str'
]

# Imputasi numerik menggunakan median data latih
for col in num_cols:
    if col in train_final.columns:
        med = train_final[col].median()
        train_final[col] = train_final[col].fillna(med)
        test_final[col] = test_final[col].fillna(med)

# Penyelarasan kategori
for col in cat_cols:
    if col in train_final.columns:
        train_final[col] = train_final[col].fillna('Unknown').astype('category')
        train_cats = list(train_final[col].cat.categories)
        mode_val = train_final[col].mode()[0]
        test_final[col] = np.where(test_final[col].isin(train_cats), test_final[col], mode_val)
        test_final[col] = pd.Categorical(test_final[col], categories=train_cats)

# Kolom yang tidak dipakai dalam pelatihan
drop_cols = [
    'id', 'date_show', 'inferred_release_date', 'total_ticket', 'original_title', 
    'clean_title', 'extracted_day_tipe', 'holiday_name', 'price_day', 'date', 
    'producer', 'casts', 'day_tipe', 'is_test', 'day_idx'
]

X = train_final.drop(columns=[c for c in drop_cols if c in train_final.columns], errors='ignore')
y = train_final['total_ticket']
sp_vals = train_final['scale_sp']
y_ratio = y / sp_vals

X_test = test_final[X.columns].copy()

print(f"-> Total fitur yang digunakan: {X.shape[1]}")
print(f"-> Daftar fitur: {list(X.columns)}")

# ==========================================
# 6. GROUP-K-FOLD OUT-OF-FOLD CROSS VALIDATION
# ==========================================
print("\n[FASE 6] Pelatihan 5-Fold GroupKFold OOF Target Encoding & Tri-Model Ensemble...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

oof_lgb_ratio = np.zeros(len(X))
oof_xgb_ratio = np.zeros(len(X))
oof_cat_ratio = np.zeros(len(X))

X['hist_cinema_ratio'] = 0.0
X['hist_genre_ratio'] = 0.0
X_test['hist_cinema_ratio'] = 0.0
X_test['hist_genre_ratio'] = 0.0

# Monotone constraints untuk XGBoost
# day_since_release: -1 (penjualan tiket meluruh seiring umur rilis)
# net_sentiment: +1 (semakin positif respon penonton, semakin kuat rasio tiket)
# tmdb_rating: +1 (semakin tinggi rating, semakin tinggi ketahanan tiket)
mc_dict = {
    'day_since_release': -1,
    'net_sentiment': 1,
    'tmdb_rating': 1
}
mc_tuple = tuple(mc_dict.get(c, 0) for c in X.columns)
mc_str = '(' + ','.join([str(x) for x in mc_tuple]) + ')'

global_mean_ratio = y_ratio.mean()

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_ratio, groups=train_final['movie_title'])):
    print(f"\n--- FOLD {fold+1}/{n_splits} ---")
    
    # 1. Out-of-fold target encoding
    cinema_means = y_ratio.iloc[train_idx].groupby(X['cinema_day_str'].iloc[train_idx].astype(str)).mean()
    genre_means = y_ratio.iloc[train_idx].groupby(X['genre_day_str'].iloc[train_idx].astype(str)).mean()
    
    X.loc[val_idx, 'hist_cinema_ratio'] = X['cinema_day_str'].iloc[val_idx].astype(str).map(cinema_means).fillna(global_mean_ratio).values
    X.loc[val_idx, 'hist_genre_ratio'] = X['genre_day_str'].iloc[val_idx].astype(str).map(genre_means).fillna(global_mean_ratio).values
    
    X_test['hist_cinema_ratio'] += X_test['cinema_day_str'].astype(str).map(cinema_means).fillna(global_mean_ratio).values / n_splits
    X_test['hist_genre_ratio'] += X_test['genre_day_str'].astype(str).map(genre_means).fillna(global_mean_ratio).values / n_splits
    
    X_tr = X.iloc[train_idx]
    y_tr = y_ratio.iloc[train_idx]
    X_va = X.iloc[val_idx]
    y_va = y_ratio.iloc[val_idx]
    
    # Model 1: LightGBM Regressor (MAE Objective)
    model_lgb = lgb.LGBMRegressor(
        n_estimators=750,
        learning_rate=0.035,
        objective='mae',
        min_data_in_leaf=45,
        cat_smooth=45,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(X_tr, y_tr)
    oof_lgb_ratio[val_idx] = np.clip(model_lgb.predict(X_va), 0, None)
    fold_mase_lgb = np.mean(np.abs(y.iloc[val_idx] - oof_lgb_ratio[val_idx] * sp_vals.iloc[val_idx]) / sp_vals.iloc[val_idx])
    
    # Model 2: XGBoost Regressor (MAE Objective + Monotone Constraints)
    model_xgb = xgb.XGBRegressor(
        n_estimators=750,
        learning_rate=0.035,
        max_depth=7,
        objective='reg:absoluteerror',
        enable_categorical=True,
        monotone_constraints=mc_str,
        random_state=42 + fold,
        n_jobs=-1
    )
    model_xgb.fit(X_tr, y_tr)
    oof_xgb_ratio[val_idx] = np.clip(model_xgb.predict(X_va), 0, None)
    fold_mase_xgb = np.mean(np.abs(y.iloc[val_idx] - oof_xgb_ratio[val_idx] * sp_vals.iloc[val_idx]) / sp_vals.iloc[val_idx])
    
    # Model 3: CatBoost Regressor (MAE Loss Function)
    # CatBoost menangani kolom kategorikal secara native
    cat_feature_indices = [i for i, c in enumerate(X.columns) if str(X[c].dtype) == 'category']
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    for col in cat_cols:
        if col in X_tr_cb.columns:
            X_tr_cb[col] = X_tr_cb[col].astype(str)
            X_va_cb[col] = X_va_cb[col].astype(str)
            
    model_cat = CatBoostRegressor(
        iterations=700,
        learning_rate=0.04,
        depth=6,
        loss_function='MAE',
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(X_tr_cb, y_tr)
    oof_cat_ratio[val_idx] = np.clip(model_cat.predict(X_va_cb), 0, None)
    fold_mase_cat = np.mean(np.abs(y.iloc[val_idx] - oof_cat_ratio[val_idx] * sp_vals.iloc[val_idx]) / sp_vals.iloc[val_idx])
    
    print(f"Fold {fold+1} MASE -> LightGBM: {fold_mase_lgb:.4f} | XGBoost: {fold_mase_xgb:.4f} | CatBoost: {fold_mase_cat:.4f}")

# ==========================================
# 7. EVALUASI INDIVIDUAL & META-OPTIMISASI ENSEMBLE (SLSQP)
# ==========================================
print("\n[FASE 7] Optimasi bobot ensemble multi-model (SLSQP)...")

mase_lgb = np.mean(np.abs(y - oof_lgb_ratio * sp_vals) / sp_vals)
mase_xgb = np.mean(np.abs(y - oof_xgb_ratio * sp_vals) / sp_vals)
mase_cat = np.mean(np.abs(y - oof_cat_ratio * sp_vals) / sp_vals)

print(f"-> OOF MASE LightGBM : {mase_lgb:.5f}")
print(f"-> OOF MASE XGBoost  : {mase_xgb:.5f}")
print(f"-> OOF MASE CatBoost : {mase_cat:.5f}")

# Fungsi objektif meta-optimasi untuk menemukan bobot (w_lgb, w_xgb, w_cat) yang meminimalkan MASE
def ensemble_mase_loss(weights):
    w1, w2, w3 = weights
    pred_ratio = w1 * oof_lgb_ratio + w2 * oof_xgb_ratio + w3 * oof_cat_ratio
    pred_ticket = pred_ratio * sp_vals
    return np.mean(np.abs(y - pred_ticket) / sp_vals)

init_w = [1/3, 1/3, 1/3]
bounds = [(0, 1), (0, 1), (0, 1)]
cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
opt_res = minimize(ensemble_mase_loss, init_w, method='SLSQP', bounds=bounds, constraints=cons)

w_lgb, w_xgb, w_cat = opt_res.x
print(f"\n[HASIL SLSQP] Bobot Optimal: LightGBM = {w_lgb:.4f}, XGBoost = {w_xgb:.4f}, CatBoost = {w_cat:.4f}")

final_oof_ratio = w_lgb * oof_lgb_ratio + w_xgb * oof_xgb_ratio + w_cat * oof_cat_ratio
final_mase = np.mean(np.abs(y - final_oof_ratio * sp_vals) / sp_vals)
print(f"-> [FINAL OOF MASE] Skor MASE Keseluruhan Exp16: {final_mase:.5f}")

# ==========================================
# 8. PELATIHAN MODEL LENGKAP & INFERENSI DATA UJI
# ==========================================
print("\n[FASE 8] Pelatihan model penuh (Full Train) & Inferensi Test Data...")

# Fit penuh LightGBM
full_lgb = lgb.LGBMRegressor(
    n_estimators=750,
    learning_rate=0.035,
    objective='mae',
    min_data_in_leaf=45,
    cat_smooth=45,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
full_lgb.fit(X, y_ratio)
test_pred_lgb = np.clip(full_lgb.predict(X_test), 0, None)

# Fit penuh XGBoost
full_xgb = xgb.XGBRegressor(
    n_estimators=750,
    learning_rate=0.035,
    max_depth=7,
    objective='reg:absoluteerror',
    enable_categorical=True,
    monotone_constraints=mc_str,
    random_state=42,
    n_jobs=-1
)
full_xgb.fit(X, y_ratio)
test_pred_xgb = np.clip(full_xgb.predict(X_test), 0, None)

# Fit penuh CatBoost
X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    if col in X_cb.columns:
        X_cb[col] = X_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

full_cat = CatBoostRegressor(
    iterations=700,
    learning_rate=0.04,
    depth=6,
    loss_function='MAE',
    cat_features=cat_cols,
    random_seed=42,
    verbose=0
)
full_cat.fit(X_cb, y_ratio)
test_pred_cat = np.clip(full_cat.predict(X_test_cb), 0, None)

# Prediksi rasio gabungan ensemble
test_pred_ratio = w_lgb * test_pred_lgb + w_xgb * test_pred_xgb + w_cat * test_pred_cat
test_pred_ticket = np.round(test_pred_ratio * X_test['scale_sp'].values)
test_pred_ticket = np.clip(test_pred_ticket, 0, None)

# ==========================================
# 9. PEMBUATAN & VALIDASI SUBMISSION
# ==========================================
print("\n[FASE 9] Pembuatan & Validasi Submission...")
submission = test_final[['id']].copy()
submission['total_ticket'] = test_pred_ticket.astype(float)

sub_path = f"{exp_dir}/submission_exp16.csv"
submission.to_csv(sub_path, index=False)

print(f"-> File submission berhasil dibuat: {sub_path}")
print(f"-> Shape submission: {submission.shape}")
print(f"-> Missing values dalam submission: {submission['total_ticket'].isnull().sum()}")
print(f"-> Preview 5 baris pertama:")
print(submission.head(5))
print(f"-> Statistik Deskriptif total_ticket:")
print(submission['total_ticket'].describe().round(2))

print("\n" + "=" * 70)
print(f">>> EXP16 SELESAI DENGAN SUKSES! FINAL OOF MASE: {final_mase:.5f} <<<")
print("=" * 70)
