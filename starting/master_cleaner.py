import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# 1. Inisialisasi Direktori
out_dir = 'dataset/dataset-05-10-26'
os.makedirs(out_dir, exist_ok=True)

print("==== MASTER CLEANER MEMULAI OPERASI ====")

# Helper: Parsing Tanggal Standar
def parse_dates(df, col='date_show'):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=False).dt.strftime('%Y-%m-%d')
    return df

# A. Membersihkan train.csv, test.csv, test_history.csv
print("[1/5] Memproses Data Utama (Train/Test)...")
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
test_hist = pd.read_csv('dataset/test_history.csv')

train = parse_dates(train)
test = parse_dates(test)
test_hist = parse_dates(test_hist)

train.to_csv(f'{out_dir}/train.csv', index=False)
test.to_csv(f'{out_dir}/test.csv', index=False)
test_hist.to_csv(f'{out_dir}/test_history.csv', index=False)

# B. Membersihkan Weather Data (Ekstrapolasi Masa Depan)
print("[2/5] Memperbaiki Anomali Cuaca (Weather)...")
weather = pd.read_csv('dataset/weather_data.csv')
weather = parse_dates(weather)
weather['date_show'] = pd.to_datetime(weather['date_show'])

# Buat kerangka waktu masa depan (Oktober 2025 - Maret 2026)
cities = weather['city_name'].unique()
future_dates = pd.date_range(start='2025-10-01', end='2026-03-31', freq='D')

future_weather = pd.DataFrame([(city, date) for city in cities for date in future_dates], columns=['city_name', 'date_show'])
# Isi cuaca masa depan dengan cuaca cerah (0) sebagai baseline
future_weather['rainfall_mm'] = 0.0
future_weather['is_heavy_rain'] = 0

weather = pd.concat([weather, future_weather], ignore_index=True)
weather['date_show'] = weather['date_show'].dt.strftime('%Y-%m-%d')
weather.to_csv(f'{out_dir}/weather_data.csv', index=False)

# C. Membersihkan Holidays & Ticket Prices
print("[3/5] Memproses Kalender Liburan & Harga Tiket...")
holidays = pd.read_csv('dataset/holidays.csv')
if 'date' in holidays.columns:
    holidays['date'] = pd.to_datetime(holidays['date'], format='mixed').dt.strftime('%Y-%m-%d')
holidays['holiday_name'] = holidays['holiday_name'].fillna('Not a Holiday')
holidays['holiday_tipe'] = holidays['holiday_tipe'].fillna('normal')
holidays.to_csv(f'{out_dir}/holidays.csv', index=False)

tickets = pd.read_csv('dataset/ticket_prices.csv')
tickets.to_csv(f'{out_dir}/ticket_prices.csv', index=False)

# D. Membersihkan Movies (Sutradara & Hype)
print("[4/5] Memproses Meta-Data Film (Movies)...")
movies = pd.read_csv('dataset/new/movie-right-director.csv')
# Imputasi Missing Values
movies['age_rating'] = movies['age_rating'].fillna(movies['age_rating'].mode()[0])
movies['runtime_minutes'] = movies['runtime_minutes'].fillna(movies['runtime_minutes'].median())
movies['director_popularity'] = movies['director_popularity'].fillna(0.0)
movies.to_csv(f'{out_dir}/movies.csv', index=False) # Ganti nama jadi movies.csv agar standar

# 5. Sanity Check (Jaminan Nol Null)
print("\\n[5/5] SANITY CHECK (Memastikan 0 Null)...")
check_files = ['train.csv', 'test.csv', 'test_history.csv', 'weather_data.csv', 'holidays.csv', 'ticket_prices.csv', 'movies.csv']

all_clean = True
for file in check_files:
    df = pd.read_csv(f'{out_dir}/{file}')
    null_sum = df.isnull().sum().sum()
    if null_sum > 0:
        print(f"❌ [GAGAL] {file} masih memiliki {null_sum} Null values!")
        all_clean = False
    else:
        print(f"✅ [LULUS] {file} 100% Bersih dari Null.")

if all_clean:
    print(f"\\n🏆 SELURUH DATA BERHASIL DICUCI DAN DISIMPAN DI: {out_dir}")
