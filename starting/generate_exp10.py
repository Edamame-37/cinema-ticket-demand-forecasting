import nbformat as nbf
import os

# Buat direktori eksperimen
exp_dir = 'experiments/Exp10_StarPower_RatioStacking'
os.makedirs(exp_dir, exist_ok=True)

nb = nbf.v4.new_notebook()

md_intro = """# Eksperimen 10: The Zenith (Perfected Architecture)
Telah disempurnakan (Refined) berdasarkan evaluasi mendalam:
1. **Perbaikan Harga Tiket Liburan**: Mengganti harga tiket menjadi tarif 'weekend' jika hari tersebut adalah Cuti Bersama/Libur Nasional.
2. **Transformasi Target Log-Ratio**: Menggunakan `log1p(y / S_p)` untuk mencegah gradien meledak akibat anomali lonjakan tiket (Outlier Ratios).
3. **Anti-Overfitting Lanjutan**: Menerapkan Feature Fraction, L2 Regularization, dan membuang kardinalitas ekstrem (`casts`, `producer`)."""

code_imports = """import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
print("Libraries Loaded.")"""

code_load = """# 1. Load Cleaned Datasets
base_dir = '../../dataset/dataset-05-10-26'

train = pd.read_csv(f'{base_dir}/train.csv')
test_hist = pd.read_csv(f'{base_dir}/test_history.csv')
test = pd.read_csv(f'{base_dir}/test.csv')

movies = pd.read_csv(f'{base_dir}/movies.csv')
weather = pd.read_csv(f'{base_dir}/weather_data.csv')
holidays = pd.read_csv(f'{base_dir}/holidays.csv')
tickets = pd.read_csv(f'{base_dir}/ticket_prices.csv')

# Konversi Tanggal
for df in [train, test_hist, test, weather, holidays]:
    if 'date_show' in df.columns:
        df['date_show'] = pd.to_datetime(df['date_show'])
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

print(f"Train size: {train.shape}, Test size: {test.shape}")"""

code_merge = """# 2. Perfected Causality Merge
def merge_metadata(df):
    # Gabung Data Film
    df = df.merge(movies, left_on='movie_title', right_on='original_title', how='left')
    df.drop(columns=['original_title'], inplace=True, errors='ignore')
    
    # Ekstraksi Hari
    df['day_name'] = df['date_show'].dt.day_name().str.lower()
    df['day_tipe_awal'] = np.where(df['day_name'].isin(['saturday', 'sunday']), 'weekend',
                                   np.where(df['day_name'] == 'friday', 'friday', 'weekday'))
    
    # Gabung Cuaca
    df = df.merge(weather, on=['city_name', 'date_show'], how='left')
    
    # Gabung Liburan
    holidays_clean = holidays[['date', 'holiday_tipe', 'holiday_name']].drop_duplicates('date')
    df = df.merge(holidays_clean, left_on='date_show', right_on='date', how='left')
    df['holiday_tipe'] = df['holiday_tipe'].fillna('normal')
    df['holiday_name'] = df['holiday_name'].fillna('Not a Holiday')
    
    # PENYEMPURNAAN 1: Harga Tiket Liburan = Harga Weekend
    # Di bioskop Indonesia, tanggal merah selalu menggunakan harga tertinggi (Weekend)
    df['day_tipe_final'] = np.where(df['holiday_tipe'] == 'holiday', 'weekend', df['day_tipe_awal'])
    
    # Gabung Harga Tiket berdasarkan day_tipe_final
    df = df.merge(tickets, left_on=['city_name', 'day_tipe_final'], right_on=['city_name', 'price_day'], how='left')
    
    # Bersihkan kolom sementara
    df.drop(columns=['day_tipe_awal', 'day_tipe_final', 'price_day'], inplace=True, errors='ignore')
    
    return df

print("Melakukan penggabungan data (dengan logika harga tiket liburan)...")
train_df = merge_metadata(train)
test_hist_df = merge_metadata(test_hist)
test_df = merge_metadata(test)
print("Selesai.")"""

code_feature = """# 3. Log-Ratio Targeting & Feature Engineering
def calculate_sp_and_ratios(df, is_train=True):
    df = df.sort_values(['movie_title', 'city_name', 'cinema_ids', 'date_show']).reset_index(drop=True)
    df['movie_age_days'] = df.groupby(['movie_title', 'city_name', 'cinema_ids'])['date_show'].rank(method='dense', ascending=True).astype(int)
    
    early_days = df[df['movie_age_days'] <= 3]
    sp_map = early_days.groupby(['movie_title', 'city_name', 'cinema_ids'])['total_ticket'].mean().reset_index()
    sp_map.rename(columns={'total_ticket': 'S_p'}, inplace=True)
    
    df = df.merge(sp_map, on=['movie_title', 'city_name', 'cinema_ids'], how='left')
    df['S_p'] = df['S_p'].fillna(df.groupby('movie_title')['S_p'].transform('median'))
    df['S_p'] = df['S_p'].fillna(df['S_p'].median())
    
    if is_train:
        # PENYEMPURNAAN 2: Log-Ratio Transform untuk meredam pencilan (outlier rasio)
        # Ratio murni (y / S_p) bisa meledak jadi 50x jika S_p kecil tapi tiba-tiba ada rombongan
        # log1p akan mengkompresi lonjakan tersebut.
        df['target_log_ratio'] = np.log1p(df['total_ticket'] / (df['S_p'] + 1.0))
    
    # Ekstraksi Fitur
    df['month'] = df['date_show'].dt.month
    df['is_weekend'] = df['day_name'].isin(['saturday', 'sunday']).astype(int)
    df['is_holiday'] = (df['holiday_tipe'] == 'holiday').astype(int)
    
    # Interaksi Fitur
    df['star_power_x_runtime'] = df['star_power_score'] * df['runtime_minutes']
    
    # Hapus fitur beracun
    drop_cols = ['screen_share_pct', 'cinema_daily_total_shows', 'max_single_movie_shows', 'is_clashing_blockbuster']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    return df

print("Menghitung S_p dan Log-Rasio Penurunan...")
train_df = calculate_sp_and_ratios(train_df, is_train=True)
test_hist_df = calculate_sp_and_ratios(test_hist_df, is_train=True)
test_df = calculate_sp_and_ratios(test_df, is_train=False)

full_train = pd.concat([train_df, test_hist_df], ignore_index=True)
full_train = full_train[full_train['movie_age_days'] > 3].reset_index(drop=True)
print(f"Data Pelatihan Siap: {full_train.shape}")"""

code_train = """# 4. Anti-Overfit Model Training
# PENYEMPURNAAN 3: Membuang kardinalitas ekstrem ('casts', 'producer') yang menghafal data
features = [
    'movie_age_days', 'S_p', 'star_power_score', 'director_popularity', 'runtime_minutes',
    'rainfall_mm', 'is_heavy_rain', 'ceil', 'month', 'is_weekend', 'is_holiday', 'star_power_x_runtime',
    'genre', 'age_rating', 'director', 'city_name', 'day_name'
]
cat_features = ['genre', 'age_rating', 'director', 'city_name', 'day_name']

X = full_train[features]
y_log_ratio = full_train['target_log_ratio']
groups = full_train['movie_title']

oof_cb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
test_cb_preds = np.zeros(len(test_df))
test_lgb_preds = np.zeros(len(test_df))

gkf = GroupKFold(n_splits=5)

for fold, (idx_tr, idx_va) in enumerate(gkf.split(X, y_log_ratio, groups)):
    print(f"--- FOLD {fold+1} ---")
    X_tr, y_tr = X.iloc[idx_tr], y_log_ratio.iloc[idx_tr]
    X_va, y_va = X.iloc[idx_va], y_log_ratio.iloc[idx_va]
    
    # 1. CatBoost (Dengan L2 Leaf Reg & Feature Bagging)
    cb = CatBoostRegressor(
        iterations=1500, learning_rate=0.04, depth=6,
        l2_leaf_reg=5, colsample_bylevel=0.8,
        cat_features=cat_features,
        eval_metric='RMSE', random_seed=42, verbose=False
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
    oof_cb[idx_va] = cb.predict(X_va)
    test_cb_preds += cb.predict(test_df[features]) / 5
    
    # 2. LightGBM (Dengan Bagging & Feature Fraction untuk mencegah overfit di sutradara)
    X_tr_lgb, X_va_lgb, test_lgb = X_tr.copy(), X_va.copy(), test_df[features].copy()
    for c in cat_features:
        X_tr_lgb[c] = X_tr_lgb[c].astype('category')
        X_va_lgb[c] = X_va_lgb[c].astype('category')
        test_lgb[c] = test_lgb[c].astype('category')
        
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=-1
    )
    lgb_model.fit(X_tr_lgb, y_tr, eval_set=[(X_va_lgb, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[idx_va] = lgb_model.predict(X_va_lgb)
    test_lgb_preds += lgb_model.predict(test_lgb) / 5

print("Pelatihan Model Lapis 1 Selesai.")"""

code_stack = """# 5. Meta-Model (Stacking) & Transformasi Balik
print("Melatih Meta-Model (Stacking)...")

stack_X_train = np.column_stack((oof_cb, oof_lgb))
stack_X_test = np.column_stack((test_cb_preds, test_lgb_preds))

# Meta model cukup membatasi prediksi positif dengan Ridge
meta_model = Ridge(alpha=5.0)
meta_model.fit(stack_X_train, y_log_ratio)

print(f"Bobot CatBoost: {meta_model.coef_[0]:.4f}")
print(f"Bobot LightGBM: {meta_model.coef_[1]:.4f}")

# Prediksi Final (dalam bentuk Log Ratio)
test_log_ratio_pred = meta_model.predict(stack_X_test)

# Kembalikan Log Ratio ke Skala Asli: Ratio = expm1(Log Ratio)
test_ratio_pred = np.expm1(test_log_ratio_pred)

# Transformasi Balik ke Total Tiket: (Ratio * (S_p + 1))
test_df['total_ticket'] = test_ratio_pred * (test_df['S_p'] + 1.0)
test_df['total_ticket'] = test_df['total_ticket'].clip(lower=0).round().astype(int)

# 6. Ekspor Hasil
submission = test_df[['id', 'total_ticket']]
sub_path = 'submission_exp10.csv'
submission.to_csv(sub_path, index=False)
print(f"Submission berhasil disimpan di {sub_path}!")"""

nb.cells = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_merge),
    nbf.v4.new_code_cell(code_feature),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_code_cell(code_stack)
]

with open(f'{exp_dir}/Exp10_StarPower_RatioStacking.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"Notebook TELAH DISEMPURNAKAN dan berhasil dibangun di {exp_dir}/Exp10_StarPower_RatioStacking.ipynb")
