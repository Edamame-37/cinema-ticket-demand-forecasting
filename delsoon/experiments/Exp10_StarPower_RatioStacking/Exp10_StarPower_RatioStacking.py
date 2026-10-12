#!/usr/bin/env python
# coding: utf-8

# # Eksperimen 10: The Zenith (Perfected Architecture)
# Telah disempurnakan (Refined) berdasarkan evaluasi mendalam:
# 1. **Perbaikan Harga Tiket Liburan**: Mengganti harga tiket menjadi tarif 'weekend' jika hari tersebut adalah Cuti Bersama/Libur Nasional.
# 2. **Transformasi Target Log-Ratio**: Menggunakan `log1p(y / S_p)` untuk mencegah gradien meledak akibat anomali lonjakan tiket (Outlier Ratios).
# 3. **Anti-Overfitting Lanjutan**: Menerapkan Feature Fraction, L2 Regularization, dan membuang kardinalitas ekstrem (`casts`, `producer`).

# In[ ]:


import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
print("Libraries Loaded.")


# In[ ]:


# 1. Load Cleaned Datasets
base_dir = 'dataset/dataset-05-10-26'

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

print(f"Train size: {train.shape}, Test size: {test.shape}")


# In[ ]:


# 2. Perfected Causality Merge
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
print("Selesai.")


# In[ ]:


# 3. Log-Ratio Targeting & Feature Engineering
print("Menghitung S_p dan Log-Rasio Penurunan...")

train_df['is_test'] = 0
test_hist_df['is_test'] = 0
test_df['is_test'] = 1

all_df = pd.concat([train_df, test_hist_df, test_df], ignore_index=True)

# Urutkan
all_df = all_df.sort_values(['movie_title', 'city_name', 'cinema_ids', 'date_show']).reset_index(drop=True)
all_df['movie_age_days'] = all_df.groupby(['movie_title', 'city_name', 'cinema_ids'])['date_show'].rank(method='dense', ascending=True).astype(int)

# Hitung S_p HANYA dari data yang memiliki total_ticket (is_test == 0)
early_days = all_df[(all_df['movie_age_days'] <= 3) & (all_df['is_test'] == 0)]
sp_map = early_days.groupby(['movie_title', 'city_name', 'cinema_ids'])['total_ticket'].mean().reset_index()
sp_map.rename(columns={'total_ticket': 'S_p'}, inplace=True)

all_df = all_df.merge(sp_map, on=['movie_title', 'city_name', 'cinema_ids'], how='left')

# Fallback S_p jika ada yang kosong
all_df['S_p'] = all_df['S_p'].fillna(all_df.groupby('movie_title')['S_p'].transform('median'))
all_df['S_p'] = all_df['S_p'].fillna(all_df['S_p'].median())

# Target Transformasi (Hanya untuk is_test == 0)
all_df['target_log_ratio'] = np.where(all_df['is_test'] == 0, 
                                      np.log1p(all_df['total_ticket'] / (all_df['S_p'] + 1.0)), 
                                      np.nan)

# Ekstraksi Fitur
all_df['month'] = all_df['date_show'].dt.month
all_df['is_weekend'] = all_df['day_name'].isin(['saturday', 'sunday']).astype(int)
all_df['is_holiday'] = (all_df['holiday_tipe'] == 'holiday').astype(int)
all_df['star_power_x_runtime'] = all_df['star_power_score'] * all_df['runtime_minutes']

# Hapus fitur beracun
drop_cols = ['screen_share_pct', 'cinema_daily_total_shows', 'max_single_movie_shows', 'is_clashing_blockbuster']
all_df.drop(columns=[c for c in drop_cols if c in all_df.columns], inplace=True)

# Split kembali
full_train = all_df[(all_df['is_test'] == 0) & (all_df['movie_age_days'] > 3)].reset_index(drop=True)
test_df = all_df[all_df['is_test'] == 1].reset_index(drop=True)

print(f"Data Pelatihan Siap: {full_train.shape}")


# In[ ]:


# 4. Anti-Overfit Model Training
# PENYEMPURNAAN 3: Membuang kardinalitas ekstrem ('casts', 'producer') yang menghafal data
features = [
    'movie_age_days', 'S_p', 'star_power_score', 'director_popularity', 'runtime_minutes',
    'rainfall_mm', 'is_heavy_rain', 'ceil', 'month', 'is_weekend', 'is_holiday', 'star_power_x_runtime',
    'genre', 'age_rating', 'director', 'city_name', 'day_name'
]
cat_features = ['genre', 'age_rating', 'director', 'city_name', 'day_name']

# SAFETY NET: Mencegah CatBoost Crash karena NaN di Categorical Features
for c in cat_features:
    full_train[c] = full_train[c].fillna('Unknown').astype(str)
    test_df[c] = test_df[c].fillna('Unknown').astype(str)

# SAFETY NET: Numerics
for c in features:
    if c not in cat_features:
        full_train[c] = full_train[c].fillna(0.0)
        test_df[c] = test_df[c].fillna(0.0)

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

print("Pelatihan Model Lapis 1 Selesai.")


# In[ ]:


# 5. Meta-Model (Stacking) & Transformasi Balik
print("Melatih Meta-Model (Stacking)...")

stack_X_train = np.column_stack((oof_cb, oof_lgb))
stack_X_test = np.column_stack((test_cb_preds, test_lgb_preds))

# Meta model cukup membatasi prediksi positif dengan Ridge
meta_model = Ridge(alpha=5.0)
meta_model.fit(stack_X_train, y_log_ratio)

print(f"Bobot CatBoost: {meta_model.coef_[0]:.4f}")
print(f"Bobot LightGBM: {meta_model.coef_[1]:.4f}")

# Evaluasi Metrik OOF (Out of Fold)
from sklearn.metrics import mean_squared_error, mean_absolute_error
oof_log_ratio_pred = meta_model.predict(stack_X_train)
oof_ratio_pred = np.expm1(oof_log_ratio_pred)
oof_ticket_pred = (oof_ratio_pred * (full_train['S_p'] + 1.0)).clip(lower=0).round().astype(int)

y_true = full_train['total_ticket']
naive_pred = full_train['S_p']

rmse = np.sqrt(mean_squared_error(y_true, oof_ticket_pred))
mae = mean_absolute_error(y_true, oof_ticket_pred)
naive_mae = mean_absolute_error(y_true, naive_pred)
mase = mae / (naive_mae + 1e-6)

print(f"\n=== HASIL EVALUASI OOF ===")
print(f"OOF RMSE: {rmse:.4f}")
print(f"OOF MASE (relatif terhadap S_p): {mase:.4f}")
print(f"==========================\n")

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
print(f"Submission berhasil disimpan di {sub_path}!")

