import pandas as pd
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

nb.cells.append(new_markdown_cell("# Exp08: Deep Learning & Entity Embeddings (The Final Frontier)\n\n"
                                   "Eksperimen ini adalah langkah terakhir pada batas *Irreducible Error* menggunakan arsitektur Keras/TensorFlow. "
                                   "Kita akan menggunakan **Entity Embeddings** untuk fitur kategori berukuran besar (kardinalitas tinggi) seperti `cinema_ids`, "
                                   "`city_name`, dan `genre` agar mesin dapat memetakan kedekatan karakteristik teater/film di ruang vektor, sesuatu "
                                   "yang mustahil dilakukan oleh algoritma Tree (XGBoost/LightGBM)."))

# M1
nb.cells.append(new_code_cell(
'''import pandas as pd
import numpy as np
import warnings
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '0'
tf.keras.utils.set_random_seed(42)

print('\\n========== M1: MEMUAT DATA ===========')
train_raw = pd.read_csv('../../dataset/train.csv')
test_hist = pd.read_csv('../../dataset/test_history.csv')
test_raw = pd.read_csv('../../dataset/test.csv')
print('Data Loaded!')'''
))

# M2 & M3
nb.cells.append(new_code_cell(
'''print('\\n========== M2 & M3: STRICT CAUSALITY & METADATA ===========')
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

train_raw['date_show'] = pd.to_datetime(train_raw['date_show'])
train_raw.sort_values(by=['movie_title', 'cinema_ids', 'date_show'], inplace=True)
train_raw['day_idx'] = train_raw.groupby(['movie_title', 'cinema_ids']).cumcount() + 1

valid_pairs = train_raw.groupby(['movie_title', 'cinema_ids']).size()
valid_pairs = valid_pairs[valid_pairs >= 4].index
train_valid = train_raw.set_index(['movie_title', 'cinema_ids']).loc[valid_pairs].reset_index()

train_hist = train_valid[train_valid['day_idx'] <= 3].copy()
train_future = train_valid[(train_valid['day_idx'] >= 4) & (train_valid['day_idx'] <= 10)].copy()

# Hapus kebocoran
cols_to_drop = ['total_show', 'occupation_rate', 'cinema_daily_total_shows', 'max_single_movie_shows', 'screen_share_pct', 'is_clashing_blockbuster']
train_future.drop(columns=cols_to_drop, inplace=True, errors='ignore')

train_profile = extract_hist_profile(train_hist)
test_profile = extract_hist_profile(test_hist)
train_final = train_future.merge(train_profile, on=['movie_title', 'cinema_ids'], how='left')
test_final = test_raw.merge(test_profile, on=['movie_title', 'cinema_ids'], how='left')

def merge_metadata(df, is_test=False):
    df['date_show'] = pd.to_datetime(df['date_show'])
    df['day_of_week'] = df['date_show'].dt.dayofweek
    df['extracted_day_tipe'] = np.where(df['day_of_week'] < 4, 'Weekday', np.where(df['day_of_week'] == 4, 'Friday', 'Weekend'))
    df['month'] = df['date_show'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([4, 5, 6]).astype('int8')
    df['is_month_end'] = df['date_show'].dt.is_month_end.astype('int8')
    
    movies = pd.read_csv('../../dataset/movies.csv')
    for col in ['producer', 'director', 'writer', 'casts']:
        movies[col] = movies[col].replace('-', 'Unknown').fillna('Unknown')
    movies['runtime_minutes'] = movies['runtime_minutes'].fillna(movies['runtime_minutes'].median())
    movies['director_popularity'] = movies['director_popularity'].fillna(0.0)
    df = df.merge(movies, left_on='movie_title', right_on='original_title', how='left')
    df['genre'] = df['genre'].fillna('Unknown')
    df['age_rating'] = df['age_rating'].fillna('Unknown')
    
    holidays = pd.read_csv('../../dataset/holidays.csv')
    holidays['date'] = pd.to_datetime(holidays['date'])
    df = df.merge(holidays, left_on='date_show', right_on='date', how='left')
    df['holiday_tipe'] = df['holiday_tipe'].fillna('normal')
    
    ticket_prices = pd.read_csv('../../dataset/ticket_prices.csv')
    df = df.merge(ticket_prices, left_on=['city_name', 'extracted_day_tipe'], right_on=['city_name', 'price_day'], how='left')
    
    weather_df = pd.read_csv('../../dataset/weather_data.csv')
    weather_df['date_show'] = pd.to_datetime(weather_df['date_show'])
    df = df.merge(weather_df, on=['city_name', 'date_show'], how='left')
    df['rainfall_mm'] = df['rainfall_mm'].fillna(0)
    df['is_heavy_rain'] = df['is_heavy_rain'].fillna(0)
    
    if is_test:
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
print('Integrasi Selesai!')'''
))

# M4
nb.cells.append(new_code_cell(
'''print('\\n========== M4: PREPROCESSING FOR DEEP LEARNING ===========')
# Deep Learning butuh penanganan khusus: Imputasi ketat, Label Encoding, dan Z-Score Normalization
cat_features = ['movie_title', 'cinema_ids', 'city_name', 'genre', 'age_rating', 'holiday_tipe', 'director', 'writer']
num_features = ['ceil', 'd1_ticket', 'd2_ticket', 'd3_ticket', 'scale_sp', 'D3_D1_ratio', 
                'occupation_rate', 'runtime_minutes', 'director_popularity', 
                'day_of_week', 'month', 'is_weekend', 'is_month_end', 'rainfall_mm', 'is_heavy_rain', 'day_since_release']

# Imputasi Numeric & Standardisasi
scaler = StandardScaler()
for col in num_features:
    median_val = train_final[col].median()
    train_final[col] = train_final[col].fillna(median_val)
    test_final[col] = test_final[col].fillna(median_val)

train_final[num_features] = scaler.fit_transform(train_final[num_features])
test_final[num_features] = scaler.transform(test_final[num_features])

# Imputasi Categorical & Label Encoding
le_dict = {}
for col in cat_features:
    train_final[col] = train_final[col].fillna('Unknown').astype(str)
    test_final[col] = test_final[col].fillna('Unknown').astype(str)
    
    le = LabelEncoder()
    # Menggabungkan untuk membuat vocabulary global
    le.fit(pd.concat([train_final[col], test_final[col]]))
    
    train_final[col] = le.transform(train_final[col])
    test_final[col] = le.transform(test_final[col])
    le_dict[col] = le

# Target (Ratio)
y = train_final['total_ticket'].values
sp_vals = train_final['scale_sp'].values  # Perhatian: ini diambil SEBELUM scaler karena kita butuh nilai aslinya!
# Kita buat variabel khusus untuk membalikkan skala target
sp_vals_asli = train_profile.set_index(['movie_title', 'cinema_ids'])['scale_sp']
train_final['sp_asli'] = train_final.set_index(['movie_title', 'cinema_ids']).index.map(sp_vals_asli)
y_ratio = y / train_final['sp_asli'].values

test_final['sp_asli'] = test_final.set_index(['movie_title', 'cinema_ids']).index.map(test_profile.set_index(['movie_title', 'cinema_ids'])['scale_sp']).fillna(1.0)

print('Normalisasi dan Embedding Preparation Selesai!')'''
))

# M5-M7
nb.cells.append(new_code_cell(
'''print('\\n========== M5-M7: NEURAL NETWORK (ENTITY EMBEDDINGS) ===========')
def build_model():
    inputs = []
    embeddings = []
    
    # Categorical Inputs via Embedding
    for col in cat_features:
        vocab_size = len(le_dict[col].classes_)
        # Rule of thumb for embedding dimension
        embed_dim = min(50, vocab_size // 2 + 1)
        
        inp = Input(shape=(1,), name=f'in_{col}')
        emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, name=f'emb_{col}')(inp)
        emb = Flatten()(emb)
        
        inputs.append(inp)
        embeddings.append(emb)
        
    # Numerical Inputs
    inp_num = Input(shape=(len(num_features),), name='in_num')
    inputs.append(inp_num)
    
    # Concatenate All
    x = Concatenate()(embeddings + [inp_num])
    
    # Deep Layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output Layer (ReLU for non-negative ratio)
    out = Dense(1, activation='relu', name='output')(x)
    
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
    return model

# Persiapan Format Keras Inputs
def get_keras_data(df):
    X = {f'in_{col}': df[col].values for col in cat_features}
    X['in_num'] = df[num_features].values
    return X

kf = GroupKFold(n_splits=5)
oof_pred_ratio = np.zeros(len(train_final))
test_preds = []

# Matriks Input Latih dan Uji
X_train_full = get_keras_data(train_final)
X_test_keras = get_keras_data(test_final)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_final, y_ratio, groups=train_final['movie_title'])):
    print(f'\\n========== FOLD {fold+1}/5 ==========')
    
    X_tr = {k: v[train_idx] for k, v in X_train_full.items()}
    y_tr = y_ratio[train_idx]
    
    X_va = {k: v[val_idx] for k, v in X_train_full.items()}
    y_va = y_ratio[val_idx]
    
    model = build_model()
    
    es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
    
    model.fit(X_tr, y_tr, 
              validation_data=(X_va, y_va),
              epochs=50, 
              batch_size=512, 
              callbacks=[es, rlr], 
              verbose=0)
              
    val_preds = model.predict(X_va, batch_size=1024).flatten()
    oof_pred_ratio[val_idx] = val_preds
    
    # Prediksi Test Set di setiap Fold (Test Time Augmentation/Ensembling)
    test_preds.append(model.predict(X_test_keras, batch_size=1024).flatten())

print('\\n[INFO] Neural Network 5-Fold Selesai Dilatih')'''
))

# M8 & M9
nb.cells.append(new_code_cell(
'''print('\\n========== M8 & M9: EVALUASI LOKAL & SUBMISSION ===========')
# Kembalikan skala dari rasio ke tiket absolut
final_pred_ticket = oof_pred_ratio * train_final['sp_asli'].values
final_mase = np.mean(np.abs(y - final_pred_ticket) / train_final['sp_asli'].values)

final_rmse = np.sqrt(mean_squared_error(y, final_pred_ticket))
final_mae = mean_absolute_error(y, final_pred_ticket)

print(f"[EVALUASI OOF DEEP LEARNING]")
print(f"MASE  : {final_mase:.4f}")
print(f"RMSE  : {final_rmse:.4f}")
print(f"MAE   : {final_mae:.4f}")

# Averaging (Ensembling) hasil prediksi test dari 5 model
pred_test_ratio = np.mean(test_preds, axis=0)
pred_test_ticket = np.round(pred_test_ratio * test_final['sp_asli'].values)

submission = test_final[['id']].copy()
submission['total_ticket'] = pred_test_ticket
submission.to_csv('submission.csv', index=False)
print('\\n[SUKSES] File submission Kaggle berhasil disimpan di: submission.csv')'''
))

with open('experiments/Exp08_DeepLearningEmbeddings/Exp08_DeepLearningEmbeddings.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Notebook Exp08 (Deep Learning) berhasil dibuat!")
