import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '0'

exp_dir = 'experiments/Exp14_TheFlawlessZenith'
os.makedirs(exp_dir, exist_ok=True)
base_dir = 'dataset/dataset-05-10-26'

print('\\n========== M1: MEMUAT DATA ===========')
train_raw = pd.read_csv(f'{base_dir}/train.csv')
test_hist = pd.read_csv(f'{base_dir}/test_history.csv')
test_raw = pd.read_csv(f'{base_dir}/test.csv')

def extract_hist_profile(hist_df):
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

print('\\n========== M2: STRICT CAUSALITY & TRAIN PRUNING ===========')
train_raw['date_show'] = pd.to_datetime(train_raw['date_show'])
train_raw.sort_values(by=['movie_title', 'cinema_ids', 'date_show'], inplace=True)
train_raw['day_idx'] = train_raw.groupby(['movie_title', 'cinema_ids']).cumcount() + 1

valid_pairs = train_raw.groupby(['movie_title', 'cinema_ids']).size()
valid_pairs = valid_pairs[valid_pairs >= 4].index
train_valid = train_raw.set_index(['movie_title', 'cinema_ids']).loc[valid_pairs].reset_index()

train_hist = train_valid[train_valid['day_idx'] <= 3].copy()
# THE MAGIC FILTER: HANYA HARI 4 s/d 10 (Sesuai dengan distribusi ujian Kaggle Test Set)
train_future = train_valid[(train_valid['day_idx'] >= 4) & (train_valid['day_idx'] <= 10)].copy()

# HAPUS semua fitur masa depan (Leakage) dari train_future
cols_to_drop = ['total_show', 'occupation_rate', 'cinema_daily_total_shows', 'max_single_movie_shows', 'screen_share_pct', 'is_clashing_blockbuster']
train_future.drop(columns=cols_to_drop, inplace=True, errors='ignore')

train_profile = extract_hist_profile(train_hist)
test_profile = extract_hist_profile(test_hist)

train_final = train_future.merge(train_profile, on=['movie_title', 'cinema_ids'], how='left')
test_final = test_raw.merge(test_profile, on=['movie_title', 'cinema_ids'], how='left')

print('\\n========== M3: INTEGRASI METADATA & HYBRID FEATURES ===========')
def merge_metadata(df, is_test=False):
    df['date_show'] = pd.to_datetime(df['date_show'])
    df['day_of_week'] = df['date_show'].dt.dayofweek
    df['extracted_day_tipe'] = np.where(df['day_of_week'] < 4, 'Weekday', np.where(df['day_of_week'] == 4, 'Friday', 'Weekend'))
    df['month'] = df['date_show'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([4, 5, 6]).astype('int8')
    df['is_month_end'] = df['date_show'].dt.is_month_end.astype('int8')
    
    movies = pd.read_csv(f'{base_dir}/movies.csv')
    for col in ['producer', 'director', 'writer', 'casts']:
        movies[col] = movies[col].replace('-', 'Unknown').fillna('Unknown')
    movies['runtime_minutes'] = movies['runtime_minutes'].fillna(movies['runtime_minutes'].median())
    movies['director_popularity'] = movies['director_popularity'].fillna(0.0)
    df = df.merge(movies, left_on='movie_title', right_on='original_title', how='left')
    df['genre'] = df['genre'].fillna('Unknown')
    df['age_rating'] = df['age_rating'].fillna('Unknown')
    
    holidays = pd.read_csv(f'{base_dir}/holidays.csv')
    holidays['date'] = pd.to_datetime(holidays['date'])
    df = df.merge(holidays, left_on='date_show', right_on='date', how='left')
    df['holiday_tipe'] = df['holiday_tipe'].fillna('normal')
    
    ticket_prices = pd.read_csv(f'{base_dir}/ticket_prices.csv')
    df = df.merge(ticket_prices, left_on=['city_name', 'extracted_day_tipe'], right_on=['city_name', 'price_day'], how='left')
    
    weather_df = pd.read_csv(f'{base_dir}/weather_data.csv')
    weather_df['date_show'] = pd.to_datetime(weather_df['date_show'])
    df = df.merge(weather_df, on=['city_name', 'date_show'], how='left')
    df['rainfall_mm'] = df['rainfall_mm'].fillna(0)
    df['is_heavy_rain'] = df['is_heavy_rain'].fillna(0)
    df['weather_penalty_score'] = df['is_heavy_rain'] * df['is_weekend']
    
    if is_test:
        test_raw['date_show'] = pd.to_datetime(test_raw['date_show'])
        test_hist['date_show'] = pd.to_datetime(test_hist['date_show'])
        first_show = pd.concat([test_hist, test_raw]).groupby('movie_title')['date_show'].min().reset_index()
    else:
        first_show = train_raw.groupby('movie_title')['date_show'].min().reset_index()
    first_show.rename(columns={'date_show': 'inferred_release_date'}, inplace=True)
    first_show['inferred_release_date'] = pd.to_datetime(first_show['inferred_release_date'])
    df = df.merge(first_show, on='movie_title', how='left')
    df['day_since_release'] = (df['date_show'] - df['inferred_release_date']).dt.days + 1
    
    return df

train_final = merge_metadata(train_final, is_test=False)
test_final = merge_metadata(test_final, is_test=True)

print('\\n========== M4: CATEGORY STANDARDIZATION & NaN IMPUTATION ===========')
cat_candidates = ['movie_title', 'cinema_ids', 'city_name', 'genre', 'age_rating', 'holiday_tipe', 'director', 'writer']
num_candidates = ['ceil', 'd1_ticket', 'd2_ticket', 'd3_ticket', 'scale_sp', 'D3_D1_ratio', 
                  'occupation_rate', 'runtime_minutes', 'director_popularity', 'day_since_release']

for col in num_candidates:
    if col in train_final.columns:
        median_val = train_final[col].median()
        train_final[col] = train_final[col].fillna(median_val)
        test_final[col] = test_final[col].fillna(median_val)

for col in cat_candidates:
    if col in train_final.columns:
        train_final[col] = train_final[col].fillna('Unknown').astype('category')
        train_cats = list(train_final[col].cat.categories)
        most_freq = train_final[col].mode()[0]
        test_final[col] = np.where(test_final[col].isin(train_cats), test_final[col], most_freq)
        test_final[col] = pd.Categorical(test_final[col], categories=train_cats)

# EXCLUDE features
exclude = ['id', 'date_show', 'inferred_release_date', 'total_ticket', 'original_title', 'extracted_day_tipe', 'holiday_name', 'price_day', 'date', 'producer', 'casts', 'day_tipe', 'day_idx']
X = train_final.drop(columns=[c for c in exclude if c in train_final.columns], errors='ignore')

y = train_final['total_ticket']
sp_vals = train_final['scale_sp']
y_ratio = y / sp_vals

X_test = test_final[X.columns].copy()

print('\\n========== M5-M7: STACKING RATIO ENSEMBLE ===========')
n_splits = 5
kf = GroupKFold(n_splits=n_splits)

oof_lgb_ratio = np.zeros(len(X))
oof_xgb_ratio = np.zeros(len(X))
oof_cb_ratio = np.zeros(len(X))

mc_dict = {'day_since_release': -1}
mc_tuple = tuple(mc_dict.get(c, 0) for c in X.columns)
mc_str = '(' + ','.join([str(x) for x in mc_tuple]) + ')'

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_ratio, groups=train_final['movie_title'])):
    print(f'========== FOLD {fold+1}/{n_splits} ==========')
    X_train, yr_train = X.iloc[train_idx], y_ratio.iloc[train_idx]
    X_val, yr_val = X.iloc[val_idx], y_ratio.iloc[val_idx]
    
    # 1. LightGBM
    model_lgb = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.04, objective='mae', min_data_in_leaf=50, cat_smooth=50, random_state=42, n_jobs=-1, verbose=-1)
    model_lgb.fit(X_train, yr_train)
    oof_lgb_ratio[val_idx] = np.clip(model_lgb.predict(X_val), 0, None)
    
    # 2. XGBoost
    model_xgb = xgb.XGBRegressor(n_estimators=700, learning_rate=0.04, max_depth=7, objective='reg:absoluteerror', enable_categorical=True, monotone_constraints=mc_str, random_state=42, n_jobs=-1)
    model_xgb.fit(X_train, yr_train)
    oof_xgb_ratio[val_idx] = np.clip(model_xgb.predict(X_val), 0, None)
    
    # 3. CatBoost
    cat_feats = [X.columns.get_loc(c) for c in cat_candidates if c in X.columns]
    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    for c in cat_candidates:
        if c in X_train_cb.columns:
            X_train_cb[c] = X_train_cb[c].astype(str)
            X_val_cb[c] = X_val_cb[c].astype(str)
            
    model_cb = CatBoostRegressor(iterations=700, learning_rate=0.04, depth=7, loss_function='MAE', eval_metric='MAE', cat_features=cat_feats, random_seed=42, verbose=False)
    model_cb.fit(X_train_cb, yr_train)
    oof_cb_ratio[val_idx] = np.clip(model_cb.predict(X_val_cb), 0, None)

print('\\n========== M8: META-OPTIMIZATION (SLSQP BOUNDED MASE) ===========')
def mase_objective(weights):
    pred_ratio = weights[0] * oof_lgb_ratio + weights[1] * oof_xgb_ratio + weights[2] * oof_cb_ratio
    pred_ticket = pred_ratio * sp_vals
    return np.mean(np.abs(y - pred_ticket) / sp_vals)

init_weights = [0.33, 0.33, 0.34]
bounds = [(0, 1), (0, 1), (0, 1)]
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
res = minimize(mase_objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)

w_lgb, w_xgb, w_cb = res.x
print(f"Koefisien Emas (LGB, XGB, CB): [{w_lgb:.4f}, {w_xgb:.4f}, {w_cb:.4f}]")

final_pred_ticket = (w_lgb * oof_lgb_ratio + w_xgb * oof_xgb_ratio + w_cb * oof_cb_ratio) * sp_vals
final_mase = np.mean(np.abs(y - final_pred_ticket) / sp_vals)
print(f"[FINAL OOF] Estimasi Stacking MASE Keseluruhan: {final_mase:.4f}")

print('\\n========== M9: INFERENSI FINAL & SUBMISSION ===========')
model_lgb_full = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.04, objective='mae', min_data_in_leaf=50, cat_smooth=50, random_state=42, n_jobs=-1, verbose=-1)
model_lgb_full.fit(X, y_ratio)
pred_test_lgb_ratio = np.clip(model_lgb_full.predict(X_test), 0, None)

model_xgb_full = xgb.XGBRegressor(n_estimators=700, learning_rate=0.04, max_depth=7, objective='reg:absoluteerror', enable_categorical=True, monotone_constraints=mc_str, random_state=42, n_jobs=-1)
model_xgb_full.fit(X, y_ratio)
pred_test_xgb_ratio = np.clip(model_xgb_full.predict(X_test), 0, None)

X_test_cb = X_test.copy()
for c in cat_candidates:
    if c in X_test_cb.columns:
        X_test_cb[c] = X_test_cb[c].astype(str)
        
model_cb_full = CatBoostRegressor(iterations=700, learning_rate=0.04, depth=7, loss_function='MAE', eval_metric='MAE', cat_features=cat_feats, random_seed=42, verbose=False)
model_cb_full.fit(X.copy().assign(**{c: X[c].astype(str) for c in cat_candidates if c in X.columns}), y_ratio)
pred_test_cb_ratio = np.clip(model_cb_full.predict(X_test_cb), 0, None)

pred_test_ratio = w_lgb * pred_test_lgb_ratio + w_xgb * pred_test_xgb_ratio + w_cb * pred_test_cb_ratio
pred_test_ticket = np.round(pred_test_ratio * X_test['scale_sp'].values)

submission = test_final[['id']].copy()
submission['total_ticket'] = pred_test_ticket
submission.to_csv(f'{exp_dir}/submission_exp14.csv', index=False)
print(f'[SUKSES] File submission Kaggle berhasil disimpan di: {exp_dir}/submission_exp14.csv')
