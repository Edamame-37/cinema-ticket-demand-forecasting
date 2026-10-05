import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import lightgbm as lgb

warnings.filterwarnings('ignore')

exp_dir = 'experiments/Exp13_TheUltimateFix'
os.makedirs(exp_dir, exist_ok=True)
base_dir = 'dataset/dataset-05-10-26'

print("=== STARTING EXP 13: THE ULTIMATE KAGGLE HACK (MASE EQUIVALENCY) ===")

# 1. Load Data
print("Loading data...")
train = pd.read_csv(f'{base_dir}/train.csv')
test_hist = pd.read_csv(f'{base_dir}/test_history.csv')
test = pd.read_csv(f'{base_dir}/test.csv')

movies = pd.read_csv(f'{base_dir}/movies.csv')
weather = pd.read_csv(f'{base_dir}/weather_data.csv')
holidays = pd.read_csv(f'{base_dir}/holidays.csv')
tickets = pd.read_csv(f'{base_dir}/ticket_prices.csv')

for df in [train, test_hist, test, weather, holidays]:
    if 'date_show' in df.columns:
        df['date_show'] = pd.to_datetime(df['date_show'])
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

train['source'] = 'train'
test_hist['source'] = 'test_hist'
test['source'] = 'test'
all_df = pd.concat([train, test_hist, test], ignore_index=True)

# 2. Global Age Calculation
print("Calculating Global Age...")
global_release = all_df.groupby('movie_title')['date_show'].min().reset_index()
global_release.rename(columns={'date_show': 'global_release_date'}, inplace=True)
all_df = all_df.merge(global_release, on='movie_title', how='left')
all_df['movie_age_days'] = (all_df['date_show'] - all_df['global_release_date']).dt.days + 1

# 3. Calculate True S_p (Only from Day 1 to Day 3)
print("Calculating S_p...")
early_days = all_df[all_df['movie_age_days'] <= 3]
sp_df = early_days.groupby(['movie_title', 'cinema_ids'])['total_ticket'].mean().reset_index()
sp_df.rename(columns={'total_ticket': 'S_p'}, inplace=True)

all_df = all_df.merge(sp_df, on=['movie_title', 'cinema_ids'], how='left')

# 4. Perfect Metadata Merge
print("Merging metadata...")
all_df = all_df.merge(movies, left_on='movie_title', right_on='original_title', how='left')
all_df.drop(columns=['original_title'], inplace=True, errors='ignore')

all_df = all_df.merge(weather, on=['city_name', 'date_show'], how='left')

holidays_clean = holidays[['date', 'holiday_tipe']].drop_duplicates('date')
all_df = all_df.merge(holidays_clean, left_on='date_show', right_on='date', how='left')
all_df['holiday_tipe'] = all_df['holiday_tipe'].fillna('normal')

all_df['day_name'] = all_df['date_show'].dt.day_name().str.lower()
all_df['day_tipe'] = np.where(all_df['day_name'].isin(['saturday', 'sunday']), 'weekend',
                              np.where(all_df['day_name'] == 'friday', 'friday', 'weekday'))

all_df['price_tier'] = np.where(all_df['holiday_tipe'] == 'holiday', 'weekend', all_df['day_tipe'])
all_df = all_df.merge(tickets, left_on=['city_name', 'price_tier'], right_on=['city_name', 'price_day'], how='left')

all_df['month'] = all_df['date_show'].dt.month
all_df['is_weekend'] = all_df['day_name'].isin(['saturday', 'sunday']).astype(int)
all_df['is_holiday'] = (all_df['holiday_tipe'] == 'holiday').astype(int)
all_df['star_power_x_runtime'] = all_df['star_power_score'] * all_df['runtime_minutes']

# 5. MENGHANCURKAN DATA LEAKAGE: HANYA LATIH PADA HARI > 3 DAN YANG MEMILIKI S_p
print("Filtering Train Data to exactly match Test Data distribution...")
train_mask = (all_df['source'] == 'train') & (all_df['movie_age_days'] > 3) & (all_df['S_p'].notna()) & (all_df['S_p'] > 0)
test_mask = (all_df['source'] == 'test')

full_train = all_df[train_mask].reset_index(drop=True)
test_df = all_df[test_mask].reset_index(drop=True)

# 6. Target Rasio Murni (TIDAK ADA LOG TRANSFORM)
full_train['target_ratio'] = full_train['total_ticket'] / full_train['S_p']

features = [
    'movie_age_days', 'S_p', 'cinema_ids', 'ceil', 'star_power_score', 'director_popularity', 'runtime_minutes',
    'rainfall_mm', 'is_heavy_rain', 'month', 'is_weekend', 'is_holiday', 'star_power_x_runtime',
    'genre', 'age_rating', 'director', 'city_name', 'day_name'
]
cat_features = ['cinema_ids', 'genre', 'age_rating', 'director', 'city_name', 'day_name']

for c in cat_features:
    full_train[c] = full_train[c].fillna('Unknown').astype(str)
    test_df[c] = test_df[c].fillna('Unknown').astype(str)

for c in features:
    if c not in cat_features:
        full_train[c] = full_train[c].fillna(0.0)
        test_df[c] = test_df[c].fillna(0.0)

X = full_train[features]
y_ratio = full_train['target_ratio']
weights = full_train['S_p'] # THE MAGIC BULLET FOR MASE
groups = full_train['movie_title']
y_raw = full_train['total_ticket']

print(f"Train Shape: {X.shape}, Test Shape: {test_df.shape}")

# 7. Model Training (MAE Objective + S_p Sample Weights)
print("Training Models...")
gkf = GroupKFold(n_splits=5)

oof_cb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
test_cb_preds = np.zeros(len(test_df))
test_lgb_preds = np.zeros(len(test_df))

for fold, (idx_tr, idx_va) in enumerate(gkf.split(X, y_ratio, groups)):
    print(f"--- FOLD {fold+1} ---")
    X_tr, y_tr, w_tr = X.iloc[idx_tr], y_ratio.iloc[idx_tr], weights.iloc[idx_tr]
    X_va, y_va, w_va = X.iloc[idx_va], y_ratio.iloc[idx_va], weights.iloc[idx_va]
    
    # CatBoost
    from catboost import Pool
    train_pool = Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_features)
    val_pool = Pool(X_va, y_va, weight=w_va, cat_features=cat_features)
    
    cb = CatBoostRegressor(
        iterations=1500, learning_rate=0.05, depth=6,
        loss_function='MAE', eval_metric='MAE',
        random_seed=42, verbose=False
    )
    cb.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
    oof_cb[idx_va] = cb.predict(X_va)
    test_cb_preds += cb.predict(test_df[features]) / 5
    
    # LightGBM
    X_tr_lgb, X_va_lgb, test_lgb = X_tr.copy(), X_va.copy(), test_df[features].copy()
    for c in cat_features:
        X_tr_lgb[c] = X_tr_lgb[c].astype('category')
        X_va_lgb[c] = X_va_lgb[c].astype('category')
        test_lgb[c] = test_lgb[c].astype('category')
        
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.05, max_depth=6,
        objective='mae', metric='mae',
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr, sample_weight=w_tr, 
        eval_set=[(X_va_lgb, y_va)], eval_sample_weight=[w_va],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    oof_lgb[idx_va] = lgb_model.predict(X_va_lgb)
    test_lgb_preds += lgb_model.predict(test_lgb) / 5

print("\\n=== OOF EVALUATION ===")
# Simple Average Ensemble pada Rasio
oof_ensemble_ratio = (oof_cb + oof_lgb) / 2.0
# Kembalikan ke Absolute Ticket (Ratio * S_p)
oof_ensemble_tickets = oof_ensemble_ratio * full_train['S_p']
oof_ensemble_tickets = np.clip(oof_ensemble_tickets, 0, None).round().astype(int)

naive_pred = full_train['S_p']
naive_mae = mean_absolute_error(y_raw, naive_pred)

mae = mean_absolute_error(y_raw, oof_ensemble_tickets)
mase = mae / (naive_mae + 1e-6)

print(f"OOF MAE: {mae:.4f}")
print(f"OOF MASE (vs S_p Baseline): {mase:.4f}")
print("======================\\n")

# 8. Final Predictions
test_ensemble_ratio = (test_cb_preds + test_lgb_preds) / 2.0
test_ensemble_tickets = test_ensemble_ratio * test_df['S_p']

test_df['total_ticket'] = np.clip(test_ensemble_tickets, 0, None).round().astype(int)

submission = test_df[['id', 'total_ticket']]
submission['id'] = submission['id'].astype(int)
submission = submission.sort_values('id')
sub_path = f'{exp_dir}/submission_exp13.csv'
submission.to_csv(sub_path, index=False)
print(f"Success! {sub_path} generated.")
