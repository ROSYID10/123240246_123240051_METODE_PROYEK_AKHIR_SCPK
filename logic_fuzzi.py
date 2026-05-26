import pandas as pd
import numpy as np
import skfuzzy as fuzz
from geopy.distance import geodesic
from skfuzzy import control as ctrl


AIRPORT_COORDS = (-8.7515, 115.1712)


def create_fuzzy_recommender(fokus='Fokus Akses'):
    """Buat sistem fuzzy rekomendasi untuk destinasi wisata dengan fokus prioritas."""
    rating = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'Rating')
    jumlah_rating = ctrl.Antecedent(np.arange(0, 110000, 1), 'Jumlah_Rating')
    aksesibilitas = ctrl.Antecedent(np.arange(0, 401, 1), 'Aksesibilitas')
    kabupaten = ctrl.Antecedent(np.arange(1, 6, 1), 'Kabupaten')
    pilihan_prioritas = ctrl.Antecedent(np.arange(0, 4, 1), 'Pilihan_Prioritas')
    score_rekomendasi = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'Score_Rekomendasi')

    rating['Buruk'] = fuzz.trimf(rating.universe, [0, 0, 2.5])
    rating['Cukup'] = fuzz.trapmf(rating.universe, [2, 2.5, 3.5, 4])
    rating['Baik'] = fuzz.trapmf(rating.universe, [3.5, 4, 4.5, 4.8])
    rating['Sangat Baik'] = fuzz.trimf(rating.universe, [4.5, 5, 5])

    jumlah_rating['Sedikit'] = fuzz.trimf(jumlah_rating.universe, [0, 0, 10000])
    jumlah_rating['Sedang'] = fuzz.trapmf(jumlah_rating.universe, [5000, 15000, 30000, 50000])
    jumlah_rating['Banyak'] = fuzz.trapmf(jumlah_rating.universe, [25000, 50000, 110000, 110000])

    aksesibilitas['Dekat'] = fuzz.trimf(aksesibilitas.universe, [0, 0, 30])
    aksesibilitas['Tidak Dekat'] = fuzz.trapmf(aksesibilitas.universe, [20, 45, 60, 75])
    aksesibilitas['Jauh'] = fuzz.trapmf(aksesibilitas.universe, [60, 80, 90, 105])
    aksesibilitas['Sangat Jauh'] = fuzz.trapmf(aksesibilitas.universe, [90, 120, 400, 400])

    kabupaten['Terbelakang'] = fuzz.trimf(kabupaten.universe, [1, 1, 2.5])
    kabupaten['Sedang'] = fuzz.trapmf(kabupaten.universe, [2, 2.5, 3.5, 4])
    kabupaten['Maju'] = fuzz.trimf(kabupaten.universe, [3.5, 5, 5])

    pilihan_prioritas['Terbaik'] = fuzz.trimf(pilihan_prioritas.universe, [0, 0, 1])
    pilihan_prioritas['Akses'] = fuzz.trimf(pilihan_prioritas.universe, [1, 1, 1.5])
    pilihan_prioritas['Hidden_Gem'] = fuzz.trimf(pilihan_prioritas.universe, [1.5, 2, 2.5])
    pilihan_prioritas['Local_Explorer'] = fuzz.trimf(pilihan_prioritas.universe, [2.5, 3, 3])

    score_rekomendasi['Rendah'] = fuzz.trimf(score_rekomendasi.universe, [0, 0, 4])
    score_rekomendasi['Sedang'] = fuzz.trapmf(score_rekomendasi.universe, [3, 5, 6, 8])
    score_rekomendasi['Tinggi'] = fuzz.trimf(score_rekomendasi.universe, [7, 10, 10])

    rules = [
        
        
        # Fokus Terbaik Keseluruhan: pertimbangkan semua faktor secara seimbang
        ctrl.Rule(pilihan_prioritas['Terbaik'] & rating['Sangat Baik'] & jumlah_rating['Banyak'] & aksesibilitas['Dekat'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Terbaik'] & rating['Sangat Baik'] & jumlah_rating['Banyak'] & aksesibilitas['Tidak Dekat'], score_rekomendasi['Tinggi']),

        ctrl.Rule(pilihan_prioritas['Terbaik'] & rating['Sangat Baik'] & aksesibilitas['Jauh'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Terbaik'] & rating['Sangat Baik'] & aksesibilitas['Sangat Jauh'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Terbaik'] & rating['Baik'] & jumlah_rating['Banyak'] & aksesibilitas['Dekat'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Terbaik'] & rating['Baik'] & jumlah_rating['Banyak'] & aksesibilitas['Tidak Dekat'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Terbaik'] & rating['Baik'] & jumlah_rating['Banyak'] & aksesibilitas['Jauh'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Terbaik'] & rating['Baik'] & jumlah_rating['Banyak'] & aksesibilitas['Sangat Jauh'], score_rekomendasi['Rendah']),
        ctrl.Rule(pilihan_prioritas['Terbaik'] & rating['Cukup'] & jumlah_rating['Banyak'] & aksesibilitas['Jauh'], score_rekomendasi['Rendah']),
        ctrl.Rule(pilihan_prioritas['Terbaik'] & rating['Cukup'] & jumlah_rating['Banyak'] & aksesibilitas['Sangat Jauh'], score_rekomendasi['Rendah']),
        ctrl.Rule(pilihan_prioritas['Terbaik'] & rating['Buruk'], score_rekomendasi['Rendah']),

        # Fokus Akses: prioritaskan aksesibilitas dekat
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Dekat'] & rating['Sangat Baik'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Dekat'] & rating['Baik'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Dekat'] & rating['Cukup'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Dekat'] & rating['Buruk'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Tidak Dekat'] & rating['Sangat Baik'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Tidak Dekat'] & rating['Baik'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Tidak Dekat'] & rating['Cukup'], score_rekomendasi['Rendah']),
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Tidak Dekat'] & rating['Buruk'], score_rekomendasi['Rendah']),
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Jauh'] & rating['Sangat Baik'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Jauh'] & rating['Baik'], score_rekomendasi['Rendah']),
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Jauh'] & (rating['Cukup'] | rating['Buruk']), score_rekomendasi['Rendah']),
        ctrl.Rule(pilihan_prioritas['Akses'] & aksesibilitas['Sangat Jauh'], score_rekomendasi['Rendah']),
        
        
        # Fokus Hidden Gem
        ctrl.Rule(pilihan_prioritas['Hidden_Gem'] & rating['Sangat Baik'] & jumlah_rating['Sedikit'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Hidden_Gem'] & rating['Baik'] & jumlah_rating['Sedikit'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Hidden_Gem'] & rating['Baik'] & jumlah_rating['Sedang'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Hidden_Gem'] & rating['Baik'] & jumlah_rating['Banyak'], score_rekomendasi['Rendah']),
        ctrl.Rule(pilihan_prioritas['Hidden_Gem'] & rating['Sangat Baik'] & jumlah_rating['Sedang'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Hidden_Gem'] & rating['Sangat Baik'] & jumlah_rating['Banyak'], score_rekomendasi['Rendah']),
        ctrl.Rule(pilihan_prioritas['Hidden_Gem'] & rating['Cukup'] & jumlah_rating['Sedikit'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Hidden_Gem'] & rating['Cukup'] & (jumlah_rating['Sedang'] | jumlah_rating['Banyak']), score_rekomendasi['Rendah']),
        ctrl.Rule(pilihan_prioritas['Hidden_Gem'] & rating['Buruk'], score_rekomendasi['Rendah']),

        # Fokus Local Explorer
        ctrl.Rule(pilihan_prioritas['Local_Explorer'] & kabupaten['Terbelakang'] & rating['Sangat Baik'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Local_Explorer'] & kabupaten['Terbelakang'] & rating['Baik'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Local_Explorer'] & kabupaten['Sedang'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Local_Explorer'] & kabupaten['Maju'], score_rekomendasi['Sedang']),
        ctrl.Rule(pilihan_prioritas['Local_Explorer'] & rating['Buruk'], score_rekomendasi['Rendah']),
        ctrl.Rule(pilihan_prioritas['Local_Explorer'] & kabupaten['Terbelakang'] & rating['Sangat Baik'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Local_Explorer'] & kabupaten['Terbelakang'] & rating['Baik'], score_rekomendasi['Tinggi']),
        ctrl.Rule(pilihan_prioritas['Local_Explorer'] & kabupaten['Maju'] & rating['Sangat Baik'], score_rekomendasi['Tinggi']),
        
    ]

    system = ctrl.ControlSystem(rules)
    return system


def load_data(file_path='bali_tourist_destination.csv'):
    """Membaca data dari file CSV destinasi wisata."""
    return pd.read_csv(file_path)


def normalize_kabupaten_names(df):
    """Normalisasi nama kabupaten/kota pada kolom kabupaten_kota."""
    df['kabupaten_normalized'] = df['kabupaten_kota'].apply(
        lambda x: x.strip().replace('Kabupaten ', '').replace('Kota ', '')
    )
    return df


def calculate_airport_distance(df, airport_coords=AIRPORT_COORDS):
    """Hitung jarak dari bandara Bali ke setiap lokasi wisata."""
    df['jarak_bandara_km'] = df.apply(
        lambda row: geodesic(airport_coords, (row['latitude'], row['longitude'])).kilometers,
        axis=1
    )
    return df


def map_kabupaten_priority(df):
    """Beri nilai prioritas untuk setiap kabupaten berdasarkan tingkat perkembangan."""
    kabupaten_mapping = {
        'Jembrana': 5,
        'Bangli': 5,
        'Buleleng': 4,
        'Karangasem': 4,
        'Klungkung': 3,
        'Tabanan': 3,
        'Gianyar': 2,
        'Denpasar': 1,
        'Badung': 1,
    }
    df['kabupaten_priority_level'] = df['kabupaten_normalized'].map(kabupaten_mapping)
    return df


def preprocess_destination_data(df):
    """Pipeline preprocessing data destinasi sebelum perhitungan fuzzy."""
    df = normalize_kabupaten_names(df)
    df = calculate_airport_distance(df)
    df = map_kabupaten_priority(df)
    return df


def compute_score_for_row(row, fuzzy_system, fokus='Fokus Akses'):
    """Hitung score rekomendasi fuzzy untuk satu baris data dengan fokus prioritas."""
    fokus_map = {
        'Terbaik Keseluruhan': 0,
        'Fokus Akses': 1,
        'Fokus Hidden Gem': 2,
        'Fokus Local Explorer': 3,
    }
    fokus_value = fokus_map.get(fokus, 0)

    try:
        simulation = ctrl.ControlSystemSimulation(fuzzy_system)
        simulation.input['Rating'] = max(0, min(5, float(row['rating'])))
        simulation.input['Jumlah_Rating'] = max(0, float(row['jumlah_rating']))
        simulation.input['Aksesibilitas'] = max(0, float(row['jarak_bandara_km']))
        simulation.input['Kabupaten'] = max(1, min(5, float(row['kabupaten_priority_level'])))
        simulation.input['Pilihan_Prioritas'] = fokus_value
        
        simulation.compute()
        
        if 'Score_Rekomendasi' in simulation.output:
            return float(simulation.output['Score_Rekomendasi'])
        else:
            return 5.0  # Default score jika tidak ada output
    except Exception as e:
        print(f"Error computing score for {row.get('nama_tempat_wisata', 'Unknown')}: {e}")
        return 5.0  # Default score jika ada error


def compute_recommendation_scores(df, fuzzy_system, fokus='Fokus Akses'):
    """Hitung kolom Score_Rekomendasi untuk seluruh dataset dengan fokus prioritas."""
    df['Score_Rekomendasi'] = df.apply(lambda row: compute_score_for_row(row, fuzzy_system, fokus), axis=1)
    return df


def rank_destinations(df, top_n=10):
    """Urutkan destinasi berdasarkan score rekomendasi tertinggi."""
    return df.sort_values(by='Score_Rekomendasi', ascending=False).head(top_n)


def display_top_destinations(df, top_n=10):
    """Tampilkan rangking destinasi teratas."""
    ranked = rank_destinations(df, top_n)
    print(ranked[
        [
            'nama_tempat_wisata',
            'kabupaten',
            'rating',
            'jumlah_rating',
            'jarak_bandara_km',
            'Score_Rekomendasi',
        ]
    ])


def main(file_path='bali_tourist_destination.csv', top_n=10):
    df = load_data(file_path)
    df = preprocess_destination_data(df)
    fuzzy_system = create_fuzzy_recommender()
    df = compute_recommendation_scores(df, fuzzy_system)
    display_top_destinations(df, top_n)


if __name__ == '__main__':
    main()
