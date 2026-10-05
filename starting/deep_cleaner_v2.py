import pandas as pd
import numpy as np

base_dir = 'dataset/dataset-05-10-26'
print("==== DEEP CLEANSER V2 DIMULAI ====")

# 1. HOLIDAYS: Cuti Bersama
print("[1/4] Memproses Kalender Liburan (Menambahkan Cuti Bersama)...")
holidays = pd.read_csv(f'{base_dir}/holidays.csv')
holidays['date'] = pd.to_datetime(holidays['date'])

# Cari tanggal raya
target_holidays = ['Idulfitri 1446 H', 'Idulfitri 1447 H', 'Kelahiran Yesus Kristus']
cuti_dates = []

for hw in target_holidays:
    dates = holidays[holidays['holiday_name'] == hw]['date']
    for d in dates:
        cuti_dates.append(d + pd.Timedelta(days=1))
        cuti_dates.append(d + pd.Timedelta(days=2))

# Terapkan Cuti Bersama pada tanggal yang telah dihitung
for cd in cuti_dates:
    idx = holidays[holidays['date'] == cd].index
    if len(idx) > 0:
        holidays.loc[idx, 'holiday_name'] = 'Cuti Bersama'
        holidays.loc[idx, 'holiday_tipe'] = 'holiday'

holidays['date'] = holidays['date'].dt.strftime('%Y-%m-%d')
holidays.to_csv(f'{base_dir}/holidays.csv', index=False)


# 2. MOVIES: Memperbaiki Durasi Outlier (1 Menit)
print("[2/4] Memperbaiki Outlier Durasi Film...")
movies = pd.read_csv(f'{base_dir}/movies.csv')
median_runtime = movies['runtime_minutes'].median()

# Koreksi film berdurasi kurang dari 60 menit
invalid_runtime_idx = movies[movies['runtime_minutes'] < 60].index
if len(invalid_runtime_idx) > 0:
    movies.loc[invalid_runtime_idx, 'runtime_minutes'] = median_runtime

movies.to_csv(f'{base_dir}/movies.csv', index=False)


# 3. WEATHER: Menyuntikkan Kota TUBAN dan MAGELANG
print("[3/4] Menambahkan 'Kota Hantu' ke Data Cuaca...")
weather = pd.read_csv(f'{base_dir}/weather_data.csv')
all_dates = pd.date_range(start='2025-04-01', end='2026-03-31', freq='D').strftime('%Y-%m-%d')

missing_cities = ['TUBAN', 'MAGELANG']
missing_weather = []

for city in missing_cities:
    for date in all_dates:
        missing_weather.append({
            'city_name': city,
            'date_show': date,
            'rainfall_mm': 0.0,
            'is_heavy_rain': 0
        })

missing_weather_df = pd.DataFrame(missing_weather)
weather = pd.concat([weather, missing_weather_df], ignore_index=True)
weather.to_csv(f'{base_dir}/weather_data.csv', index=False)


# 4. TICKET PRICES: Memperbaiki Kapitalisasi Huruf (Typo format)
print("[4/4] Sinkronisasi Kapitalisasi Harga Tiket...")
tickets = pd.read_csv(f'{base_dir}/ticket_prices.csv')
tickets['price_day'] = tickets['price_day'].str.lower()
tickets.to_csv(f'{base_dir}/ticket_prices.csv', index=False)

print("\\n[SELESAI] SELURUH ANOMALI BERHASIL DIBERANTAS! Dataset siap untuk Machine Learning.")
