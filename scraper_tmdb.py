import pandas as pd
import requests
import time
import os
import argparse

# ==============================================================================
# SCRIPT PENARIKAN DATA TMDB (RESEPSI AUDIENS & REKAM JEJAK SINEAS)
# Dokumen: implementasi Fase 1 (exp_doc_detail.md)
# ==============================================================================

# Ganti dengan TMDB API Key Anda atau set lewat environment variable
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://api.themoviedb.org/3"

def search_movie(title):
    """Mencari film di TMDB berdasarkan judul dan mengembalikan metadatanya."""
    url = f"{BASE_URL}/search/movie"
    params = {
        'api_key': TMDB_API_KEY,
        'query': title,
        'language': 'id-ID'  # Prioritaskan metadata bahasa Indonesia
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get('results', [])
        if results:
            # Ambil pencarian paling relevan (indeks 0)
            return results[0]
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for '{title}': {e}")
        return None

def get_movie_details(movie_id):
    """Mengambil durasi (runtime) dan informasi spesifik film."""
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {'api_key': TMDB_API_KEY}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {}

def scrape_tmdb_data(input_csv, output_csv):
    """
    Membaca dataset film, menarik rating, vote count, dan runtime dari TMDB.
    """
    if TMDB_API_KEY == "YOUR_API_KEY_HERE":
        print("[GAGAL] Harap masukkan TMDB_API_KEY yang valid di dalam skrip!")
        return

    print(f"Membaca dataset dari {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Ambil daftar unik judul film untuk menghindari panggilan API berulang
    unique_movies = df['original_title'].unique()
    print(f"Ditemukan {len(unique_movies)} judul unik. Memulai proses scraping...")
    
    scraped_data = []
    
    for i, title in enumerate(unique_movies):
        # Hindari rate-limit TMDB (maksimal 40 request per detik)
        time.sleep(0.1) 
        
        movie_info = search_movie(title)
        
        if movie_info:
            movie_id = movie_info.get('id')
            details = get_movie_details(movie_id)
            
            scraped_data.append({
                'original_title': title,
                'tmdb_rating': movie_info.get('vote_average', 0.0),
                'tmdb_vote_count': movie_info.get('vote_count', 0),
                'tmdb_runtime': details.get('runtime', None),
                'tmdb_popularity': movie_info.get('popularity', 0.0)
            })
            print(f"[{i+1}/{len(unique_movies)}] Sukses: {title} (Rating: {movie_info.get('vote_average')})")
        else:
            scraped_data.append({
                'original_title': title,
                'tmdb_rating': None,
                'tmdb_vote_count': None,
                'tmdb_runtime': None,
                'tmdb_popularity': None
            })
            print(f"[{i+1}/{len(unique_movies)}] GAGAL: {title} tidak ditemukan di TMDB.")
            
    # Gabungkan hasil scraping dengan dataframe asli
    df_scraped = pd.DataFrame(scraped_data)
    df_final = df.merge(df_scraped, on='original_title', how='left')
    
    df_final.to_csv(output_csv, index=False)
    print(f"\\n[SELESAI] Data berhasil disimpan ke {output_csv}")

if __name__ == "__main__":
    # Secara default, skrip ini akan memperbarui dataset movies.csv
    input_file = "dataset/movies.csv"
    output_file = "dataset/movies_tmdb_enriched.csv"
    
    if os.path.exists(input_file):
        scrape_tmdb_data(input_file, output_file)
    else:
        print(f"File {input_file} tidak ditemukan!")
