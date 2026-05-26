import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logic_fuzzi as lf

from logic_fuzzi import (
    preprocess_destination_data,
    create_fuzzy_recommender,
    compute_recommendation_scores,
    rank_destinations,
)

st.set_page_config(
    page_title="Smart Tourism Studio",
    page_icon="🌴",
    layout="wide",
)


# =========================
# CUSTOM STYLES
# =========================
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(120deg, #f5f7ff 0%, #e6f0ff 45%, #ffffff 100%);
            color: #0f172a;
        }

        .header-card {
            background: rgba(255, 255, 255, 0.90);
            border-radius: 24px;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: #ffffff;
            border-radius: 20px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 1.2rem;
            box-shadow: 0 18px 35px rgba(15, 23, 42, 0.06);
        }

        .rule-box {
            background: #f8fafc;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            border: 1px solid #c7d2fe;
            margin-bottom: 0.7rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0f172a;
        }

        .overview-badge {
            color: #2563eb;
            font-weight: 700;
        }

        .stButton>button {
            background-image: linear-gradient(135deg, #2563eb, #38bdf8);
            color: white;
            border: none;
        }

        .stButton>button:hover {
            filter: brightness(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def plot_membership_functions():
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.patch.set_facecolor('#f8fafc')

    rating_universe = np.arange(0, 5.1, 0.1)
    buruk = np.array([max(0, 1 - i / 2.5) if i <= 2.5 else 0 for i in rating_universe])
    cukup = np.array([max(0, min(1, (i - 2) / 0.5, (4 - i) / 0.5)) if 2 <= i <= 4 else 0 for i in rating_universe])
    baik = np.array([max(0, min(1, (i - 3.5) / 0.5, (4.8 - i) / 0.3)) if 3.5 <= i <= 4.8 else 0 for i in rating_universe])
    sangat_baik = np.array([max(0, min(1, (i - 4.5) / 0.5)) if i >= 4.5 else 0 for i in rating_universe])
    ax = axes[0, 0]
    ax.plot(rating_universe, buruk, 'r-', linewidth=2.5, label='Buruk')
    ax.plot(rating_universe, cukup, 'orange', linewidth=2.5, label='Cukup')
    ax.plot(rating_universe, baik, 'g-', linewidth=2.5, label='Baik')
    ax.plot(rating_universe, sangat_baik, 'darkgreen', linewidth=2.5, label='Sangat Baik')
    ax.fill_between(rating_universe, 0, buruk, alpha=0.18, color='red')
    ax.fill_between(rating_universe, 0, cukup, alpha=0.18, color='orange')
    ax.fill_between(rating_universe, 0, baik, alpha=0.18, color='green')
    ax.fill_between(rating_universe, 0, sangat_baik, alpha=0.18, color='darkgreen')
    ax.set_title('Rating Membership Function', fontsize=11, fontweight='bold')
    ax.set_xlabel('Rating (0-5)')
    ax.set_ylabel('Membership')
    ax.grid(True, alpha=0.3)
    ax.legend()

    akses_universe = np.arange(0, 401, 5)
    dekat = np.array([max(0, 1 - i / 30) if i <= 30 else 0 for i in akses_universe])
    tidak_dekat = np.array([max(0, min(1, (i - 20) / 25, (75 - i) / 15)) if 20 <= i <= 75 else 0 for i in akses_universe])
    jauh = np.array([max(0, min(1, (i - 60) / 30, (105 - i) / 15)) if 60 <= i <= 105 else 0 for i in akses_universe])
    sangat_jauh = np.array([max(0, min(1, (i - 90) / 310)) if i >= 90 else 0 for i in akses_universe])
    ax = axes[0, 1]
    ax.plot(akses_universe, dekat, 'g-', linewidth=2.5, label='Dekat')
    ax.plot(akses_universe, tidak_dekat, 'blue', linewidth=2.5, label='Tidak Dekat')
    ax.plot(akses_universe, jauh, 'orange', linewidth=2.5, label='Jauh')
    ax.plot(akses_universe, sangat_jauh, 'r-', linewidth=2.5, label='Sangat Jauh')
    ax.fill_between(akses_universe, 0, dekat, alpha=0.18, color='green')
    ax.fill_between(akses_universe, 0, tidak_dekat, alpha=0.18, color='blue')
    ax.fill_between(akses_universe, 0, jauh, alpha=0.18, color='orange')
    ax.fill_between(akses_universe, 0, sangat_jauh, alpha=0.18, color='red')
    ax.set_title('Aksesibilitas (Jarak km) Membership Function', fontsize=11, fontweight='bold')
    ax.set_xlabel('Jarak (km)')
    ax.set_ylabel('Membership')
    ax.grid(True, alpha=0.3)
    ax.legend()

    jml_universe = np.arange(0, 110000, 5000)
    sedikit = np.array([max(0, 1 - i / 10000) if i <= 10000 else 0 for i in jml_universe])
    sedang = np.array([max(0, min(1, (i - 5000) / 10000, (50000 - i) / 20000)) if 5000 <= i <= 50000 else 0 for i in jml_universe])
    banyak = np.array([max(0, min(1, (i - 25000) / 25000)) if i >= 25000 else 0 for i in jml_universe])
    ax = axes[1, 0]
    ax.plot(jml_universe, sedikit, 'b-', linewidth=2.5, label='Sedikit')
    ax.plot(jml_universe, sedang, 'purple', linewidth=2.5, label='Sedang')
    ax.plot(jml_universe, banyak, 'r-', linewidth=2.5, label='Banyak')
    ax.fill_between(jml_universe, 0, sedikit, alpha=0.18, color='blue')
    ax.fill_between(jml_universe, 0, sedang, alpha=0.18, color='purple')
    ax.fill_between(jml_universe, 0, banyak, alpha=0.18, color='red')
    ax.set_title('Jumlah Rating Membership Function', fontsize=11, fontweight='bold')
    ax.set_xlabel('Jumlah Rating')
    ax.set_ylabel('Membership')
    ax.grid(True, alpha=0.3)
    ax.legend()

    kab_universe = np.arange(1, 5.1, 0.1)
    terbelakang = np.array([max(0, 1 - (i - 1) / 1.5) if i <= 2.5 else 0 for i in kab_universe])
    sedang_kab = np.array([max(0, min(1, (i - 2) / 0.5, (4 - i) / 0.5)) if 2 <= i <= 4 else 0 for i in kab_universe])
    maju = np.array([max(0, min(1, (i - 3.5) / 1.5)) if i >= 3.5 else 0 for i in kab_universe])
    ax = axes[1, 1]
    ax.plot(kab_universe, terbelakang, 'r-', linewidth=2.5, label='Terbelakang')
    ax.plot(kab_universe, sedang_kab, 'orange', linewidth=2.5, label='Sedang')
    ax.plot(kab_universe, maju, 'g-', linewidth=2.5, label='Maju')
    ax.fill_between(kab_universe, 0, terbelakang, alpha=0.18, color='red')
    ax.fill_between(kab_universe, 0, sedang_kab, alpha=0.18, color='orange')
    ax.fill_between(kab_universe, 0, maju, alpha=0.18, color='green')
    ax.set_title('Tingkat Kemajuan Kabupaten Membership Function', fontsize=11, fontweight='bold')
    ax.set_xlabel('Level (1=Terbelakang, 5=Maju)')
    ax.set_ylabel('Membership')
    ax.grid(True, alpha=0.3)
    ax.legend()

    score_universe = np.arange(0, 10.1, 0.1)
    rendah = np.array([max(0, 1 - i / 4) if i <= 4 else 0 for i in score_universe])
    sedang_score = np.array([max(0, min(1, (i - 3) / 2, (8 - i) / 2)) if 3 <= i <= 8 else 0 for i in score_universe])
    tinggi = np.array([max(0, min(1, (i - 7) / 3)) if i >= 7 else 0 for i in score_universe])
    ax = axes[2, 0]
    ax.plot(score_universe, rendah, 'r-', linewidth=2.5, label='Rendah')
    ax.plot(score_universe, sedang_score, 'orange', linewidth=2.5, label='Sedang')
    ax.plot(score_universe, tinggi, 'g-', linewidth=2.5, label='Tinggi')
    ax.fill_between(score_universe, 0, rendah, alpha=0.18, color='red')
    ax.fill_between(score_universe, 0, sedang_score, alpha=0.18, color='orange')
    ax.fill_between(score_universe, 0, tinggi, alpha=0.18, color='green')
    ax.set_title('Score Rekomendasi Output Membership Function', fontsize=11, fontweight='bold')
    ax.set_xlabel('Score (0-10)')
    ax.set_ylabel('Membership')
    ax.grid(True, alpha=0.3)
    ax.legend()

    prior_universe = np.arange(0, 3.5, 0.05)
    terbaik = np.array([max(0, 1 - i / 1.0) if i <= 1 else 0 for i in prior_universe])
    akses = np.array([max(0, 1 - abs(i - 1) / 0.5) if abs(i - 1) <= 0.5 else 0 for i in prior_universe])
    hidden = np.array([max(0, 1 - abs(i - 2) / 0.5) if abs(i - 2) <= 0.5 else 0 for i in prior_universe])
    local_explorer = np.array([max(0, 1 - abs(i - 3) / 0.5) if abs(i - 3) <= 0.5 else 0 for i in prior_universe])
    ax = axes[2, 1]
    ax.plot(prior_universe, terbaik, 'cyan', linewidth=2.5, label='Terbaik')
    ax.plot(prior_universe, akses, 'b-', linewidth=2.5, label='Akses')
    ax.plot(prior_universe, hidden, 'purple', linewidth=2.5, label='Hidden Gem')
    ax.plot(prior_universe, local_explorer, 'darkred', linewidth=2.5, label='Local Explorer')
    ax.fill_between(prior_universe, 0, terbaik, alpha=0.18, color='cyan')
    ax.fill_between(prior_universe, 0, akses, alpha=0.18, color='blue')
    ax.fill_between(prior_universe, 0, hidden, alpha=0.18, color='purple')
    ax.fill_between(prior_universe, 0, local_explorer, alpha=0.18, color='darkred')
    ax.set_title('Pilihan Prioritas Membership Function', fontsize=11, fontweight='bold')
    ax.set_xlabel('Priority Mode')
    ax.set_ylabel('Membership')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Terbaik', 'Akses', 'Hidden Gem', 'Local Explorer'])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_ranking_chart(ranked_df):
    fig, ax = plt.subplots(figsize=(12, max(6, len(ranked_df) * 0.35)))
    fig.patch.set_facecolor('white')
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ranked_df)))
    bars = ax.barh(
        range(len(ranked_df)),
        ranked_df['Score_Rekomendasi'].values,
        color=colors,
        edgecolor='black',
        linewidth=1.2,
    )
    ax.set_yticks(range(len(ranked_df)))
    ax.set_yticklabels(ranked_df['nama_tempat_wisata'].values, fontsize=10)
    for i, (bar, val) in enumerate(zip(bars, ranked_df['Score_Rekomendasi'].values)):
        ax.text(
            val + 0.14,
            i,
            f'{val:.2f}',
            va='center',
            fontsize=9,
            fontweight='bold',
        )
    ax.set_xlabel('Score Rekomendasi', fontsize=12, fontweight='bold')
    ax.set_title('🏆 Top Destination Ranking', fontsize=14, fontweight='bold', pad=18)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.set_xlim(0, max(10, ranked_df['Score_Rekomendasi'].max() * 1.1))
    plt.tight_layout()
    return fig


def plot_dashboard_statistics(df):
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    fig.patch.set_facecolor('#f8fafc')

    ax = axs[0]
    ax.hist(df['rating'], bins=12, color='#2563eb', alpha=0.75, edgecolor='black')
    ax.set_title('Distribusi Rating')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Jumlah Destinasi')

    ax = axs[1]
    df['kategori'].value_counts().head(6).sort_values().plot(
        kind='barh', color='#14b8a6', ax=ax, edgecolor='black'
    )
    ax.set_title('Kategori Teratas')
    ax.set_xlabel('Jumlah Destinasi')
    ax.set_ylabel('Kategori')

    ax = axs[2]
    scatter = ax.scatter(
        df['jarak_bandara_km'],
        df['Score_Rekomendasi'],
        c=df['rating'],
        cmap='viridis',
        s=55,
        alpha=0.85,
        edgecolors='w',
        linewidth=0.6,
    )
    ax.set_title('Jarak vs Score Rekomendasi')
    ax.set_xlabel('Jarak ke Bandara (km)')
    ax.set_ylabel('Score Rekomendasi')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Rating')

    return fig


def build_sidebar(df):
    st.sidebar.markdown('## 🌴 Smart Tourism Studio')
    st.sidebar.markdown('Sistem rekomendasi wisata Bali dengan tampilan lebih modern dan insight statistik.')
    st.sidebar.markdown('---')

    jenis_wisata = st.sidebar.selectbox(
        'Jenis Wisata',
        ['Semua Wisata'] + sorted(df['kategori'].dropna().unique().tolist()),
    )

    fokus = st.sidebar.radio(
        'Fokus Recommendation',
        ['Terbaik Keseluruhan', 'Fokus Akses', 'Fokus Hidden Gem', 'Fokus Local Explorer'],
    )

    top_n = st.sidebar.slider('Jumlah Rekomendasi', min_value=5, max_value=30, value=10)
    generate_btn = st.sidebar.button('Generate Recommendation', use_container_width=True)
    return jenis_wisata, fokus, top_n, generate_btn


def filter_dataset(df, jenis_wisata):
    if jenis_wisata and jenis_wisata != 'Semua Wisata':
        return df[df['kategori'] == jenis_wisata].copy()
    return df.copy()


def build_insights(df, ranked_df, fokus):
    avg_rating = df['rating'].mean()
    avg_reviews = df['jumlah_rating'].mean()
    avg_distance = df['jarak_bandara_km'].mean()
    top_kabupaten = df['kabupaten_normalized'].value_counts().idxmax()
    hidden_gem = df[(df['rating'] >= 4.5) & (df['jumlah_rating'] < 1500)]
    hidden_gem_count = len(hidden_gem)
    return {
        'total_destinations': len(df),
        'avg_rating': round(avg_rating, 2),
        'avg_reviews': int(avg_reviews),
        'avg_distance': round(avg_distance, 1),
        'top_kabupaten': top_kabupaten,
        'hidden_gem_count': hidden_gem_count,
        'best_score': round(ranked_df['Score_Rekomendasi'].max(), 2) if len(ranked_df) else 0,
        'focus_label': fokus,
    }


def render_header():
    st.markdown(
        '<div class="header-card">'
        '<p class="overview-badge">Overview • Sistem Rekomendasi Wisata</p>'
        '<h1 style="margin: 0; font-size: 3rem; color: #0f172a;">Smart Tourism Studio</h1>'
        '<p style="font-size: 1.05rem; color: #475569; margin-top: 0.8rem; max-width: 760px;">'
        'sistem rekomendasi wisata Bali dengan statistik interaktif dan     visualisasi membership fuzzy'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_metrics(insights):
    col1, col2, col3, col4 = st.columns(4, gap='large')
    with col1:
        st.metric('Destinasi Diproses', insights['total_destinations'], delta=None)
    with col2:
        st.metric('Rating Rata-rata', insights['avg_rating'], delta='dari 5.0')
    with col3:
        st.metric('Rata-rata Review', insights['avg_reviews'], delta=None)
    with col4:
        st.metric('Jarak Rata-rata', f"{insights['avg_distance']} km", delta=None)

    col5, col6 = st.columns(2, gap='large')
    with col5:
        st.markdown(f"<div class='metric-card'><strong>Top Kabupaten:</strong> {insights['top_kabupaten']}</div>", unsafe_allow_html=True)
    with col6:
        st.markdown(f"<div class='metric-card'><strong>Hidden gem:</strong> {insights['hidden_gem_count']} destinasi</div>", unsafe_allow_html=True)


def render_dataset_teaser(df, jenis_wisata):
    st.markdown('---')
    st.markdown('<h2 class="section-title">Siap menjelajah Bali?</h2>', unsafe_allow_html=True)
    st.write('Pilih filter di sidebar lalu tekan **Generate Recommendation** untuk melihat rekomendasi dengan visualisasi, statistik, dan preview lokasi.')

    col1, col2, col3 = st.columns(3, gap='large')
    with col1:
        st.metric('Destinasi dalam Filter', len(df), delta=None)
    with col2:
        st.metric('Kategori Teratas', df['kategori'].mode()[0] if len(df) else '-', delta=None)
    with col3:
        st.metric('Kabupaten Terbanyak', df['kabupaten_normalized'].mode()[0] if len(df) else '-', delta=None)

    st.markdown('''
        **Kenapa tombol ini penting?**
        - Filter data akan disesuaikan sebelum hitungan fuzzy.
        - Hasil rekomendasi hanya muncul saat kamu siap.
        - Tampilan jadi lebih interaktif dan tidak langsung penuh.
    ''')


def render_destination_gallery(ranked):
    if ranked.empty:
        return

    st.markdown('#### Spotlight Rekomendasi Teratas')
    top_cards = ranked.head(3)
    cols = st.columns(min(3, len(top_cards)), gap='large')
    for col, (_, row) in zip(cols, top_cards.iterrows()):
        with col:
            st.markdown(f"**{row['nama_tempat_wisata']}**")
            st.markdown(f"{row['kategori']} • {row['kabupaten_kota']}")
            st.markdown(f"Score: **{row['Score_Rekomendasi']:.2f}**")
            st.markdown(f"[Buka Google Maps]({row['link_google_maps']})")


def render_dynamic_rules():
    focus_rules = {
        'Terbaik Keseluruhan': {
            'desc': 'Jalankan rule global untuk memilih destinasi terbaik secara keseluruhan.',
            'rules': [
                'Prioritaskan rating tinggi',
                'Pertimbangkan akses dan jumlah review',
                'Rekomendasi terbaik tanpa filter prioritas khusus',
            ],
        },
        'Fokus Akses': {
            'desc': 'Prioritaskan wisata yang dekat dan mudah dijangkau.',
            'rules': [
                'Boost score untuk aksesibilitas dekat',
                'Kurangi score lokasi sangat jauh',
                'Utamakan wisata populer dan mudah diakses',
            ],
        },
        'Fokus Hidden Gem': {
            'desc': 'Cari wisata bagus tapi belum terlalu ramai.',
            'rules': [
                'Boost score jumlah rating sedikit',
                'Naikkan prioritas rating tinggi',
                'Utamakan destinasi underrated',
            ],
        },
        'Fokus Local Explorer': {
            'desc': 'Eksplorasi wisata lokal turis yang otentik dan beragam.',
            'rules': [
                'Boost kabupaten prioritas tinggi',
                'Kurangi dominasi kota maju',
                'Utamakan wisata potensial lokal',
            ],
        },
    }

    cols = st.columns(len(focus_rules), gap='large')
    for idx, (title, item) in enumerate(focus_rules.items()):
        with cols[idx]:
            st.markdown(f'### {title}')
            st.caption(item['desc'])
            for rule in item['rules']:
                st.markdown(f'<div class="rule-box">⚡ {rule}</div>', unsafe_allow_html=True)


def render_recommendation_section(ranked, df, fokus):
    st.markdown('---')
    st.markdown('<h2 class="section-title">Rekomendasi Unggulan</h2>', unsafe_allow_html=True)
    viz_col, info_col = st.columns([3, 1], gap='large')
    with info_col:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.write('Fokus saat ini:')
        st.write(f'**{fokus}**')
        if len(ranked):
            st.write('Top score:')
            st.write(f'**{ranked.iloc[0].Score_Rekomendasi:.2f}**')
            st.write('Destinasi terbaik:')
            st.write(f'**{ranked.iloc[0].nama_tempat_wisata}**')
            st.markdown(f"[Buka Google Maps]({ranked.iloc[0].link_google_maps})")
        st.markdown('</div>', unsafe_allow_html=True)

    with viz_col:
        st.dataframe(
            ranked[
                [
                    'nama_tempat_wisata',
                    'kategori',
                    'kabupaten_kota',
                    'rating',
                    'jumlah_rating',
                    'jarak_bandara_km',
                    'Score_Rekomendasi',
                ]
            ].reset_index(drop=True),
            use_container_width=True,
        )

    render_destination_gallery(ranked)

    st.markdown('### 📈 Visualisasi Rekomendasi')
    fig_ranking = plot_ranking_chart(ranked)
    st.pyplot(fig_ranking, use_container_width=True)
    plt.close('all')


def render_statistics_tab(df):
    st.markdown('---')
    st.markdown('<h2 class="section-title">Statistik dan Insight</h2>', unsafe_allow_html=True)
    fig_stats = plot_dashboard_statistics(df)
    st.pyplot(fig_stats, use_container_width=True)
    plt.close('all')

    kab_summary = df.groupby('kabupaten_normalized')['Score_Rekomendasi'].mean().sort_values(ascending=False).head(8)
    st.markdown('#### Kabupaten dengan Rata-rata Score Tertinggi')
    st.bar_chart(kab_summary)

    st.markdown('#### Distribusi Destinasi per Kategori')
    st.bar_chart(df['kategori'].value_counts())


def main():
    df = lf.load_data('bali_tourist_destination.csv')
    df = preprocess_destination_data(df)

    jenis_wisata, fokus, top_n, generate_btn = build_sidebar(df)
    filtered_df = filter_dataset(df, jenis_wisata)

    st.markdown('<div class="header-card"></div>', unsafe_allow_html=True)
    render_header()
    render_dynamic_rules()

    if filtered_df.empty:
        st.warning('Tidak ada destinasi yang sesuai filter saat ini. Coba pilih jenis wisata lain.')
        return

    if not generate_btn:
        render_dataset_teaser(filtered_df, jenis_wisata)
        return

    fuzzy_system = create_fuzzy_recommender(fokus)
    scored_df = compute_recommendation_scores(filtered_df, fuzzy_system, fokus)
    ranked = rank_destinations(scored_df, top_n)

    insights = build_insights(scored_df, ranked, fokus)
    render_metrics(insights)

    tab_rekomendasi, tab_stats, tab_membership = st.tabs([
        'Rekomendasi',
        'Statistik',
        'Fuzzy Membership',
    ])

    with tab_rekomendasi:
        render_recommendation_section(ranked, scored_df, fokus)

    with tab_stats:
        render_statistics_tab(scored_df)

    with tab_membership:
        st.subheader('📊 Fuzzy Membership Functions')
        st.caption('Visualisasi fungsi keanggotaan dari semua input dan output fuzzy system')
        fig_membership = plot_membership_functions()
        st.pyplot(fig_membership, use_container_width=True)
        plt.close('all')

    st.markdown('---')
    st.caption('Aplikasi ini menggunakan dataset Bali Touristik lokal dan sistem fuzzy rekomendasi untuk menghasilkan insight wisata.')


if __name__ == '__main__':
    main()
