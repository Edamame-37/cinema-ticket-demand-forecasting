"""
========================================================================================
Eksperimen 17: Exp17_TheCleanWOMZenith
========================================================================================
Deskripsi:
Eksperimen ini menyempurnakan Exp14 (Leaderboard 0.62822) dan membenahi distorsi pada Exp16:
1. MENGHAPUS Target Encoding Statis (cinema_day_str & genre_day_str):
   Menghilangkan bias hari kerja yang sebelumnya menyebabkan over-prediksi tiket pada hari 6-8.
   Kendali peluruhan hari kerja dikembalikan sepenuhnya kepada fitur kalender murni
   (day_of_week, is_weekend, extracted_day_tipe, ticket_prices).
2. MEMPERTAHANKAN Normalisasi Judul Regex (100% Match):
   Memperbaiki bug sufiks bioskop '(IMAX 2D|3D|...)' sehingga 100% film uji (Avatar, Zootopia 2, dll)
   terhubung sempurna ke metadata film.
3. MEMPERTAHANKAN Sinyal Word-of-Mouth TMDB:
   Mengintegrasikan tmdb_rating, net_sentiment (rasio positif - rasio negatif), dan wom_confidence
   sebagai fitur numerik langsung dengan monotone constraints terarah.
4. Tri-Model SLSQP Meta-Optimization:
   Optimasi bobot LightGBM dan XGBoost langsung meminimalkan metrik MASE kompetisi.
5. Golden Blend Generation:
   Membuat submission tunggal Exp17 dan submission Golden Blend (50% Exp14 + 50% Exp17)
   sebagai jaring pengaman (safety net) terkuat untuk 2 kesempatan submisi tersisa.
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
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '0'

# ==========================================
# 0. KONFIGURASI DIREKTORI & OUTPUT
# ==========================================
exp_dir = 'experiments/Exp17_TheCleanWOMZenith'
os.makedirs(exp_dir, exist_ok=True)

base_dir = 'dataset/dataset-05-10-26'
new_data_dir = 'dataset/new-again'

print("=" * 70)
print(">>> MEMULAI EKSEKUSI PIPELINE EXP17: THE CLEAN WOM ZENITH <<<")
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

# ==========================================
# 2. NORMALISASI JUDUL & PENGGABUNGAN METADATA
# ==========================================
print("\n[FASE 2] Normalisasi judul film (100% Match) & integrasi sinyal TMDB...")

def clean_title(title):
    """
    Menghapus sufiks format bioskop seperti (IMAX 2D), (3D), (4DX), dll
    sehingga pencocokan judul film ke metadata menjadi 100% akurat.
    """
    t = re.sub(r'\s*\((?:IMAX\s*2D|IMAX\s*3D|3D|2D|4DX|ATMOS|RE-RELEASE)\)\s*', '', str(title), flags=re.IGNORECASE)
    return t.strip()

movies_combined = pd.merge(
    movies_base,
    movies_tmdb[['original_title', 'tmdb_rating', 'rasio_positif', 'rasio_negatif', 'total_ulasan']],
    on='original_title',
    how='left'
)

# Imputasi default untuk metadata
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
# 3. STRICT CAUSALITY & PROFIL HARI 1-3
# ==========================================
print("\n[FASE 3] Ekstraksi profil historis (Hari 1-3) & Train Pruning (Hari 4-10)...")

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
        occupation_rate=('occupation_rate', 'mean'),
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

# Ekstraksi profil historis
train_profile = extract_hist_profile(train_hist)
test_profile = extract_hist_profile(test_hist)

train_final = train_future.merge(train_profile, on=['movie_title', 'cinema_ids'], how='left')
test_final = test_raw.merge(test_profile, on=['movie_title', 'cinema_ids'], how='left')

# ==========================================
# 4. REKAYASA FITUR BERSIH (CLEAN CALENDAR & WOM)
# ==========================================
print("\n[FASE 4] Menghasilkan fitur kalender murni & Word-of-Mouth...")

def merge_features(df, is_test=False):
    """
    Menambahkan seluruh fitur kontekstual, format bioskop, cuaca, liburan,
    dan sinyal Word-of-Mouth TMDB ke dataframe TANPA target encoding statis.
    """
    df = df.copy()
    df['date_show'] = pd.to_datetime(df['date_show'])
    df['day_of_week'] = df['date_show'].dt.dayofweek
    df['extracted_day_tipe'] = np.where(df['day_of_week'] < 4, 'Weekday', np.where(df['day_of_week'] == 4, 'Friday', 'Weekend'))
    df['month'] = df['date_show'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([4, 5, 6]).astype('int8')
    df['is_month_end'] = df['date_show'].dt.is_month_end.astype('int8')
    
    # Title clean & merge 100%
    df['clean_title'] = df['movie_title'].apply(clean_title)
    df = df.merge(movies_combined, left_on='clean_title', right_on='original_title', how='left')
    
    # Fitur Sinyal Word-of-Mouth (WOM)
    df['net_sentiment'] = df['rasio_positif'] - df['rasio_negatif']
    df['wom_confidence'] = df['tmdb_rating'] * np.log1p(df['total_ulasan'])
    df['is_imax'] = df['movie_title'].str.contains('IMAX', case=False, na=False).astype('int8')
    df['is_3d'] = df['movie_title'].str.contains('3D', case=False, na=False).astype('int8')
    
    # Merge Liburan
    holidays_df = holidays.copy()
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    df = df.merge(holidays_df, left_on='date_show', right_on='date', how='left')
    df['holiday_tipe'] = df['holiday_tipe'].fillna('normal')
    
    # Merge Harga Tiket Kota
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
    
    return df

train_final = merge_features(train_final, is_test=False)
test_final = merge_features(test_final, is_test=True)

# ==========================================
# 5. STANDARISASI FITUR & IMPUTASI
# ==========================================
print("\n[FASE 5] Standarisasi fitur kategorikal & imputasi numerik...")

cat_candidates = [
    'movie_title', 'cinema_ids', 'city_name', 'genre', 'age_rating', 
    'holiday_tipe', 'director', 'writer'
]

num_candidates = [
    'ceil', 'd1_ticket', 'd2_ticket', 'd3_ticket', 'scale_sp', 'D3_D1_ratio', 
    'occupation_rate', 'runtime_minutes', 'director_popularity', 'star_power_score',
    'tmdb_rating', 'net_sentiment', 'wom_confidence', 'day_since_release', 
    'price', 'rainfall_mm', 'weather_penalty_score'
]

# Imputasi numerik
for col in num_candidates:
    if col in train_final.columns:
        med = train_final[col].median()
        train_final[col] = train_final[col].fillna(med)
        test_final[col] = test_final[col].fillna(med)

# Penyelarasan kategori
for col in cat_candidates:
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
    'producer', 'casts', 'day_tipe', 'day_idx'
]

X = train_final.drop(columns=[c for c in drop_cols if c in train_final.columns], errors='ignore')
y = train_final['total_ticket']
sp_vals = train_final['scale_sp']
y_ratio = y / sp_vals

X_test = test_final[X.columns].copy()

print(f"-> Total fitur yang digunakan: {X.shape[1]}")
print(f"-> Daftar fitur: {list(X.columns)}")

# ==========================================
# 6. 5-FOLD GROUPKFOLD OOF PELATIHAN
# ==========================================
print("\n[FASE 6] Pelatihan 5-Fold GroupKFold OOF...")

n_splits = 5
kf = GroupKFold(n_splits=n_splits)

oof_lgb_ratio = np.zeros(len(X))
oof_xgb_ratio = np.zeros(len(X))

# Monotone constraints:
# day_since_release: -1 (penjualan tiket meluruh seiring hari rilis)
# net_sentiment: +1 (respon positif memperkuat penjualan)
# tmdb_rating: +1 (rating tinggi memperkuat penjualan)
mc_dict = {
    'day_since_release': -1,
    'net_sentiment': 1,
    'tmdb_rating': 1
}
mc_tuple = tuple(mc_dict.get(c, 0) for c in X.columns)
mc_str = '(' + ','.join([str(x) for x in mc_tuple]) + ')'

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_ratio, groups=train_final['movie_title'])):
    X_train, yr_train = X.iloc[train_idx], y_ratio.iloc[train_idx]
    X_val, yr_val = X.iloc[val_idx], y_ratio.iloc[val_idx]
    
    # 1. LightGBM (MAE Objective)
    model_lgb = lgb.LGBMRegressor(
        n_estimators=700, 
        learning_rate=0.04, 
        objective='mae', 
        min_data_in_leaf=50, 
        cat_smooth=50, 
        random_state=42 + fold, 
        n_jobs=-1, 
        verbose=-1
    )
    model_lgb.fit(X_train, yr_train)
    oof_lgb_ratio[val_idx] = np.clip(model_lgb.predict(X_val), 0, None)
    
    # 2. XGBoost (MAE Objective + Monotone Constraints)
    model_xgb = xgb.XGBRegressor(
        n_estimators=700, 
        learning_rate=0.04, 
        max_depth=7, 
        objective='reg:absoluteerror', 
        enable_categorical=True, 
        monotone_constraints=mc_str, 
        random_state=42 + fold, 
        n_jobs=-1
    )
    model_xgb.fit(X_train, yr_train)
    oof_xgb_ratio[val_idx] = np.clip(model_xgb.predict(X_val), 0, None)
    
    mase_lgb_fold = np.mean(np.abs(y.iloc[val_idx] - oof_lgb_ratio[val_idx] * sp_vals.iloc[val_idx]) / sp_vals.iloc[val_idx])
    mase_xgb_fold = np.mean(np.abs(y.iloc[val_idx] - oof_xgb_ratio[val_idx] * sp_vals.iloc[val_idx]) / sp_vals.iloc[val_idx])
    print(f"Fold {fold+1} -> LightGBM MASE: {mase_lgb_fold:.4f} | XGBoost MASE: {mase_xgb_fold:.4f}")

# ==========================================
# 7. OPTIMASI BOBOT SLSQP
# ==========================================
print("\n[FASE 7] Optimasi bobot SLSQP langsung terhadap MASE...")

def mase_objective(weights):
    w1, w2 = weights
    pred_ratio = w1 * oof_lgb_ratio + w2 * oof_xgb_ratio
    pred_ticket = pred_ratio * sp_vals
    return np.mean(np.abs(y - pred_ticket) / sp_vals)

init_weights = [0.5, 0.5]
bounds = [(0, 1), (0, 1)]
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
res = minimize(mase_objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)

w_lgb, w_xgb = res.x
print(f"Koefisien Emas (LGB, XGB): [{w_lgb:.4f}, {w_xgb:.4f}]")

final_pred_ticket = (w_lgb * oof_lgb_ratio + w_xgb * oof_xgb_ratio) * sp_vals
final_mase = np.mean(np.abs(y - final_pred_ticket) / sp_vals)
print(f"-> [FINAL OOF MASE] Skor MASE Keseluruhan Exp17: {final_mase:.4f}")

# ==========================================
# 8. PELATIHAN MODEL LENGKAP & INFERENSI
# ==========================================
print("\n[FASE 8] Pelatihan model penuh (Full Fit) & Inferensi Data Uji...")

model_lgb_full = lgb.LGBMRegressor(
    n_estimators=700, 
    learning_rate=0.04, 
    objective='mae', 
    min_data_in_leaf=50, 
    cat_smooth=50, 
    random_state=42, 
    n_jobs=-1, 
    verbose=-1
)
model_lgb_full.fit(X, y_ratio)
pred_test_lgb_ratio = np.clip(model_lgb_full.predict(X_test), 0, None)

model_xgb_full = xgb.XGBRegressor(
    n_estimators=700, 
    learning_rate=0.04, 
    max_depth=7, 
    objective='reg:absoluteerror', 
    enable_categorical=True, 
    monotone_constraints=mc_str, 
    random_state=42, 
    n_jobs=-1
)
model_xgb_full.fit(X, y_ratio)
pred_test_xgb_ratio = np.clip(model_xgb_full.predict(X_test), 0, None)

pred_test_ratio = w_lgb * pred_test_lgb_ratio + w_xgb * pred_test_xgb_ratio
pred_test_ticket = np.round(pred_test_ratio * X_test['scale_sp'].values)
pred_test_ticket = np.clip(pred_test_ticket, 0, None)

# ==========================================
# 9. PEMBUATAN SUBMISSION EXP17 & GOLDEN BLEND
# ==========================================
print("\n[FASE 9] Pembuatan berkas submission Exp17 & Golden Blend...")

# 1. Submission Exp17 Murni
submission_17 = test_final[['id']].copy()
submission_17['total_ticket'] = pred_test_ticket.astype(float)
sub17_path = f"{exp_dir}/submission_exp17.csv"
submission_17.to_csv(sub17_path, index=False)
print(f"-> [SUKSES] Berkas Submission Exp17 disimpan di: {sub17_path}")
print(f"-> Statistik Deskriptif Exp17:\n{submission_17['total_ticket'].describe().round(2)}")

# 2. Submission Golden Blend (50% Exp14 + 50% Exp17)
sub14_path = 'experiments/Exp14_TheFlawlessZenith/submission_exp14.csv'
if os.path.exists(sub14_path):
    sub14 = pd.read_csv(sub14_path)
    golden_blend = test_final[['id']].copy()
    golden_blend['total_ticket'] = np.round(0.50 * sub14['total_ticket'] + 0.50 * submission_17['total_ticket']).astype(float)
    golden_path = f"{exp_dir}/submission_golden_blend.csv"
    golden_blend.to_csv(golden_path, index=False)
    print(f"-> [SUKSES] Berkas Golden Blend (50% Exp14 + 50% Exp17) disimpan di: {golden_path}")
    print(f"-> Statistik Deskriptif Golden Blend:\n{golden_blend['total_ticket'].describe().round(2)}")

print("\n" + "=" * 70)
print(f">>> EXP17 SELESAI! FINAL OOF MASE: {final_mase:.4f} <<<")
print("=" * 70)
