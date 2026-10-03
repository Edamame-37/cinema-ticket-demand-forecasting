import pandas as pd
import numpy as np
import datetime
import os

# ==============================================================================
# SCRIPT PENARIKAN DATA KALENDER (SIKLUS FINANSIAL & KALENDER EDUKASI)
# Dokumen: implementasi Fase 3 (exp_doc_detail.md)
# ==============================================================================

def generate_calendar_features(start_date='2024-01-01', end_date='2024-12-31'):
    """
    Menghasilkan *Lookup Table* kalender yang memetakan setiap hari ke fitur Payday, 
    dan status liburan sekolah secara deterministik.
    """
    print(f"Menginisialisasi kalender dari {start_date} hingga {end_date}...")
    
    # Buat rentang waktu harian
    date_range = pd.date_range(start=start_date, end=end_date)
    df_cal = pd.DataFrame({'date': date_range})
    
    # Ekstraksi komponen dasar
    df_cal['day'] = df_cal['date'].dt.day
    df_cal['month'] = df_cal['date'].dt.month
    df_cal['day_of_week'] = df_cal['date'].dt.dayofweek
    
    # 1. SIKLUS FINANSIAL (PAYDAY EFFECT)
    # Asumsi umum di Indonesia: Tanggal gajian adalah tanggal 25 (Swasta) hingga tanggal 1 (PNS).
    # Momentum "Dompet Penuh" biasanya bertahan sekitar 5 hari (hingga tanggal 5).
    # Kriteria Payday = Tanggal 25 s.d 31, dan Tanggal 1 s.d 5
    df_cal['is_payday'] = np.where((df_cal['day'] >= 25) | (df_cal['day'] <= 5), 1, 0)
    
    # 2. KALENDER EDUKASI (LIBUR SEKOLAH & UJIAN)
    # Ini merupakan pendekatan *Heuristic* (berdasarkan kalender akademik umum 2024).
    # Libur Semester Genap: Pertengahan Juni hingga pertengahan Juli.
    # Libur Semester Ganjil: Akhir Desember.
    # Ujian Sekolah (UAS): Awal Juni dan Awal Desember (Membunuh jumlah penonton).
    
    df_cal['is_school_holiday'] = 0
    df_cal['is_exam_week'] = 0
    
    for idx, row in df_cal.iterrows():
        m = row['month']
        d = row['day']
        
        # Penandaan Ujian Sekolah (Contoh: 1-15 Juni, 1-10 Desember)
        if (m == 6 and d <= 15) or (m == 12 and d <= 10):
            df_cal.at[idx, 'is_exam_week'] = 1
            
        # Penandaan Libur Sekolah (Contoh: 16 Juni - 15 Juli, 20-31 Desember)
        if (m == 6 and d > 15) or (m == 7 and d <= 15) or (m == 12 and d >= 20):
            df_cal.at[idx, 'is_school_holiday'] = 1
            
    # Hapus kolom pembantu
    df_cal.drop(columns=['day', 'month', 'day_of_week'], inplace=True)
    
    return df_cal

if __name__ == "__main__":
    output_file = "dataset/calendar_features.csv"
    
    # Untuk kompetisi, data mencakup tahun 2023 - 2024
    df_calendar = generate_calendar_features(start_date='2023-01-01', end_date='2024-12-31')
    
    df_calendar.to_csv(output_file, index=False)
    print(f"\\n[SELESAI] Data Siklus Finansial & Edukasi berhasil disimpan ke {output_file}")
    print("Contoh Data:")
    print(df_calendar.head())
