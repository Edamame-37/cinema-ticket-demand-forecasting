import os
import json
import pandas as pd
import numpy as np

print("=== MEMULAI EKSEKUSI M1 & M2 ===")

# Determinisme absolut
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(2026)

print("\n--- [M1] Memuat Dataset ---")
train_df = pd.read_parquet('dataset/cleaned_dataset.parquet')
print(f"Ukuran data awal: {train_df.shape}")

print("\n--- [M1] Validasi Duplikat & Agregasi ---")
dups = train_df.duplicated(subset=['movie_title', 'cinema_ids', 'date_show']).sum()
print(f"Jumlah baris duplikat (movie_title, cinema_ids, date_show): {dups}")
if dups > 0:
    print("PERINGATAN KRITIS: Ditemukan duplikasi! Kemungkinan akibat 'join explosion' dari tabel libur/kota.")
    print("Mengatasi duplikasi dengan menyimpan kemunculan pertama...")
    train_df.drop_duplicates(subset=['movie_title', 'cinema_ids', 'date_show'], keep='first', inplace=True)


if not pd.api.types.is_datetime64_any_dtype(train_df['date_show']):
    print("Mengonversi date_show ke format datetime64...")
    train_df['date_show'] = pd.to_datetime(train_df['date_show'])

print("\n--- [M2] Rekayasa Fitur Dasar ---")
# Drop total_show
if 'total_show' in train_df.columns:
    train_df.drop(columns=['total_show'], inplace=True)
    print("Fitur 'total_show' dihapus (Pencegahan Data Leakage).")

print("Menginjeksi weather_data.csv (karena belum digabungkan di preprocessing.ipynb)...")
weather_df = pd.read_csv('dataset/weather_data.csv')
weather_df['date_show'] = pd.to_datetime(weather_df['date_show'])
train_df = train_df.merge(weather_df, on=['city_name', 'date_show'], how='left')
train_df['rainfall_mm'] = train_df['rainfall_mm'].fillna(0)
train_df['is_heavy_rain'] = train_df['is_heavy_rain'].fillna(0)

# Weather penalty & Cannibalization
train_df['weather_penalty_score'] = train_df['is_heavy_rain'] * train_df['is_weekend']
train_df['rain_intensity_impact'] = train_df['is_heavy_rain'] * train_df['rainfall_mm']
train_df['is_squeezed'] = ((train_df['screen_share_pct'] < 15) & (train_df['is_clashing_blockbuster'] == 1)).astype('int8')

# Day since release
if 'release_date' in train_df.columns:
    train_df['release_date'] = pd.to_datetime(train_df['release_date'])
    train_df['day_since_release'] = (train_df['date_show'] - train_df['release_date']).dt.days + 1
else:
    first_show = train_df.groupby('movie_title')['date_show'].min().reset_index()
    first_show.rename(columns={'date_show': 'inferred_release_date'}, inplace=True)
    train_df = train_df.merge(first_show, on='movie_title', how='left')
    train_df['day_since_release'] = (train_df['date_show'] - train_df['inferred_release_date']).dt.days + 1

print("\n--- [M2] Transformasi Jendela Waktu (D1-D3 -> D4-D10) ---")
train_df.sort_values(by=['movie_title', 'cinema_ids', 'date_show'], inplace=True)
train_df['day_idx'] = train_df.groupby(['movie_title', 'cinema_ids']).cumcount() + 1

pair_counts = train_df.groupby(['movie_title', 'cinema_ids']).size()
valid_pairs = pair_counts[pair_counts >= 4].index

df_valid = train_df.set_index(['movie_title', 'cinema_ids']).loc[valid_pairs].reset_index()

history = df_valid[df_valid['day_idx'] <= 3]
history_agg = history.groupby(['movie_title', 'cinema_ids']).agg(
    d1_ticket=('total_ticket', lambda x: x.iloc[0] if len(x)>0 else np.nan),
    d2_ticket=('total_ticket', lambda x: x.iloc[1] if len(x)>1 else np.nan),
    d3_ticket=('total_ticket', lambda x: x.iloc[2] if len(x)>2 else np.nan),
    scale_sp=('total_ticket', lambda x: max(x.mean(), 1))
).reset_index()

history_agg['D3_D1_ratio'] = history_agg['d3_ticket'] / history_agg['d1_ticket'].replace(0, 1)
p99 = history_agg['D3_D1_ratio'].quantile(0.99)
history_agg['D3_D1_ratio'] = np.clip(history_agg['D3_D1_ratio'], a_min=0, a_max=p99)

targets = df_valid[(df_valid['day_idx'] >= 4) & (df_valid['day_idx'] <= 10)]
train_final = targets.merge(history_agg, on=['movie_title', 'cinema_ids'], how='left')

for col in ['movie_title', 'cinema_ids', 'city_name', 'genre', 'age_rating']:
    if col in train_final.columns:
        train_final[col] = train_final[col].astype('category')
        
print(f"Dimensi data final siap latih (D4-D10): {train_final.shape}")
print("\nCuplikan Fitur Penting:")
print(train_final[['movie_title', 'day_since_release', 'total_ticket', 'scale_sp', 'D3_D1_ratio', 'weather_penalty_score']].head())

print("\n--- MENYIMPAN KE JUPYTER NOTEBOOK ---")
cells = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# Eksperimen 05: Stacking Ensemble (Fase M1 & M2)\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import os\n", "import pandas as pd\n", "import numpy as np\n", "\n", 
        "os.environ['PYTHONHASHSEED'] = '0'\n", "np.random.seed(2026)\n", "\n", 
        "if 'train_df' not in locals():\n", "    train_df = pd.read_parquet('../../dataset/cleaned_dataset.parquet')\n", 
        "    print(f'Ukuran awal: {train_df.shape}')\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## M1: Post-Preprocessing EDA\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "if 'train_df' in locals():\n", 
        "    dups = train_df.duplicated(subset=['movie_title', 'cinema_ids', 'date_show']).sum()\n", 
        "    print(f'Duplikasi: {dups}')\n", 
        "    if dups > 0:\n",
        "        print('Menghapus duplikasi...')\n",
        "        train_df.drop_duplicates(subset=['movie_title', 'cinema_ids', 'date_show'], keep='first', inplace=True)\n", "\n", 
        "    if not pd.api.types.is_datetime64_any_dtype(train_df['date_show']):\n", 
        "        train_df['date_show'] = pd.to_datetime(train_df['date_show'])\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## M2: Rekayasa Fitur Hibrida (Cuaca, Sineas, Kanibalisme)\\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "if 'train_df' in locals():\\n", 
        "    if 'total_show' in train_df.columns:\\n", 
        "        train_df.drop(columns=['total_show'], inplace=True)\\n", "\\n", 
        "    weather_df = pd.read_csv('../../dataset/weather_data.csv')\\n",
        "    weather_df['date_show'] = pd.to_datetime(weather_df['date_show'])\\n",
        "    train_df = train_df.merge(weather_df, on=['city_name', 'date_show'], how='left')\\n",
        "    train_df['rainfall_mm'] = train_df['rainfall_mm'].fillna(0)\\n",
        "    train_df['is_heavy_rain'] = train_df['is_heavy_rain'].fillna(0)\\n", "\\n",
        "    train_df['weather_penalty_score'] = train_df['is_heavy_rain'] * train_df['is_weekend']\\n", 
        "    train_df['rain_intensity_impact'] = train_df['is_heavy_rain'] * train_df['rainfall_mm']\\n", 
        "    train_df['is_squeezed'] = ((train_df['screen_share_pct'] < 15) & (train_df['is_clashing_blockbuster'] == 1)).astype('int8')\\n", "\\n", 
        "    if 'release_date' in train_df.columns:\n", 
        "        train_df['release_date'] = pd.to_datetime(train_df['release_date'])\n", 
        "        train_df['day_since_release'] = (train_df['date_show'] - train_df['release_date']).dt.days + 1\n", 
        "    else:\n", 
        "        first_show = train_df.groupby('movie_title')['date_show'].min().reset_index()\n", 
        "        first_show.rename(columns={'date_show': 'inferred_release_date'}, inplace=True)\n", 
        "        train_df = train_df.merge(first_show, on='movie_title', how='left')\n", 
        "        train_df['day_since_release'] = (train_df['date_show'] - train_df['inferred_release_date']).dt.days + 1\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## M2: Transformasi Jendela Waktu (D1-D3) & Pre-Stacking Prep\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "if 'train_df' in locals():\n", 
        "    train_df.sort_values(by=['movie_title', 'cinema_ids', 'date_show'], inplace=True)\n", 
        "    train_df['day_idx'] = train_df.groupby(['movie_title', 'cinema_ids']).cumcount() + 1\n", "\n", 
        "    pair_counts = train_df.groupby(['movie_title', 'cinema_ids']).size()\n", 
        "    valid_pairs = pair_counts[pair_counts >= 4].index\n", "\n", 
        "    df_valid = train_df.set_index(['movie_title', 'cinema_ids']).loc[valid_pairs].reset_index()\n", "\n", 
        "    history = df_valid[df_valid['day_idx'] <= 3]\n", 
        "    history_agg = history.groupby(['movie_title', 'cinema_ids']).agg(\n", 
        "        d1_ticket=('total_ticket', lambda x: x.iloc[0] if len(x)>0 else np.nan),\n", 
        "        d2_ticket=('total_ticket', lambda x: x.iloc[1] if len(x)>1 else np.nan),\n", 
        "        d3_ticket=('total_ticket', lambda x: x.iloc[2] if len(x)>2 else np.nan),\n", 
        "        scale_sp=('total_ticket', lambda x: max(x.mean(), 1))\n", 
        "    ).reset_index()\n", "\n", 
        "    history_agg['D3_D1_ratio'] = history_agg['d3_ticket'] / history_agg['d1_ticket'].replace(0, 1)\n", 
        "    p99 = history_agg['D3_D1_ratio'].quantile(0.99)\n", 
        "    history_agg['D3_D1_ratio'] = np.clip(history_agg['D3_D1_ratio'], a_min=0, a_max=p99)\n", "\n", 
        "    targets = df_valid[(df_valid['day_idx'] >= 4) & (df_valid['day_idx'] <= 10)]\n", 
        "    train_final = targets.merge(history_agg, on=['movie_title', 'cinema_ids'], how='left')\n", "\n", 
        "    for col in ['movie_title', 'cinema_ids', 'city_name', 'genre', 'age_rating']:\n", 
        "        if col in train_final.columns:\n", 
        "            train_final[col] = train_final[col].astype('category')\n", 
        "            \n", 
        "    print(f'Dimensi data final (siap latih): {train_final.shape}')\n", 
        "    display(train_final[['movie_title', 'day_since_release', 'total_ticket', 'scale_sp', 'D3_D1_ratio', 'weather_penalty_score']].head())\n"
    ]}
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

os.makedirs('experiments/Exp05_StackingEnsemble', exist_ok=True)
with open('experiments/Exp05_StackingEnsemble/Exp05_StackingEnsemble.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook Exp05_StackingEnsemble.ipynb berhasil digenerate di folder experiments/Exp05_StackingEnsemble/")
