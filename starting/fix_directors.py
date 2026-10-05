import pandas as pd
import wikipedia
import re
import time
import os

# Konfigurasi Wikipedia
wikipedia.set_lang("en")

# 1. Load Datasets
print("[INFO] Membaca dataset...")
movies_df = pd.read_csv('dataset/new/movies_final_features.csv')
audit_df = pd.read_csv('dataset/new/wikipedia_audit_report.csv')

# Gabungkan untuk mendapatkan status
df = movies_df.merge(audit_df[['original_title', 'wiki_status']], on='original_title', how='left')

# Siapkan regex untuk menangkap nama (2 kata huruf kapital)
# Mendukung format: Directed by John Doe, disutradarai oleh Joko Anwar
regex_en = re.compile(r'directed by ([A-Z][A-Za-z\-]+\s[A-Z][A-Za-z\-]+)')
regex_id = re.compile(r'disutradarai oleh ([A-Z][A-Za-z\-]+\s[A-Z][A-Za-z\-]+)')

mismatch_count = 0
fixed_count = 0

print("[INFO] Memulai pencarian sutradara di Wikipedia...")

for idx, row in df.iterrows():
    status = str(row['wiki_status'])
    if 'VALID' not in status and status != 'nan':
        mismatch_count += 1
        title = row['original_title']
        search_query = f"{title} film"
        
        try:
            # Cari kandidat halaman
            search_results = wikipedia.search(search_query, results=3)
            
            if not search_results:
                continue
                
            # Ambil halaman pertama
            page_title = search_results[0]
            summary = wikipedia.summary(page_title, sentences=3, auto_suggest=False)
            
            # Coba ekstrak dengan Regex Bahasa Inggris
            match = regex_en.search(summary)
            if not match:
                # Coba bahasa Indonesia
                wikipedia.set_lang("id")
                try:
                    summary_id = wikipedia.summary(page_title, sentences=3, auto_suggest=False)
                    match = regex_id.search(summary_id)
                except:
                    pass
                wikipedia.set_lang("en") # Kembalikan ke Inggris
                
            if match:
                correct_director = match.group(1)
                print(f"[FIXED] {title}: {row['director']} -> {correct_director}")
                df.at[idx, 'director'] = correct_director
                fixed_count += 1
            else:
                print(f"[FAILED] {title}: Pola regex tidak ditemukan di summary.")
                
            time.sleep(0.5) # Jeda sopan santun untuk API Wikipedia
            
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"[SKIP] {title}: Halaman ambigu (Disambiguation).")
        except wikipedia.exceptions.PageError:
            print(f"[SKIP] {title}: Halaman tidak ditemukan.")
        except Exception as e:
            print(f"[ERROR] {title}: {str(e)}")

# 2. Bersihkan kolom status sebelum disimpan
if 'wiki_status' in df.columns:
    df.drop(columns=['wiki_status'], inplace=True)

# 3. Simpan Hasil
out_path = 'dataset/new/movie-right-director.csv'
df.to_csv(out_path, index=False)

print(f"\\n[SELESAI] Total bermasalah: {mismatch_count}")
print(f"[SELESAI] Berhasil diperbaiki otomatis: {fixed_count}")
print(f"[SELESAI] Dataset bersih disimpan di: {out_path}")
