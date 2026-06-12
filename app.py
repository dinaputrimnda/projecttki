import streamlit as st
import pandas as pd
import requests
import io
import re
import base64
import os
from rank_bm25 import BM25Okapi

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(page_title="Portal Skripsi IT", page_icon="🎓", layout="wide")

# Fungsi konversi gambar lokal ke base64
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return None

img_base64 = get_image_base64("picture1.jpg")

# =====================================================================
# CSS CUSTOM (TEMA PROFESIONAL, BERSIH, & KONTRAST TINGGI)
# =====================================================================
st.markdown(f"""
<style>
    /* Latar Belakang Aplikasi dengan Overlay Putih */
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), 
                    url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}

    /* Sidebar Putih Solid dan Teks Hitam */
    [data-testid="stSidebar"] {{
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }}
    [data-testid="stSidebar"] * {{
        color: #000000 !important;
    }}

    /* Desain Input Pencarian */
    .stTextInput input {{
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cbd5e1 !important;
    }}

    /* Kartu Hasil Pencarian */
    .result-card {{
        background-color: #ffffff !important;
        color: #000000 !important;
        padding: 25px;
        border-radius: 8px;
        border-left: 5px solid #4f46e5 !important;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}

    /* Teks Judul dan Badge Skor */
    .card-title {{
        color: #000000 !important; 
        font-size: 20px; 
        font-weight: 700; 
        margin-bottom: 10px;
        font-family: sans-serif;
    }}
    .score-badge {{
        background-color: #e0e7ff !important;
        color: #3730a3 !important;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# LOGIKA DATA
# =====================================================================
def preprocess_text(text):
    text = str(text).lower().replace('-', ' ')
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

@st.cache_data 
def load_data():
    url = "https://docs.google.com/uc?export=download&id=1VIIiPgJ_mAo7n1KRoKf537fQ0f7y3YJB"
    try:
        response = requests.get(url, timeout=20)
        df = pd.read_csv(io.StringIO(response.text))
        df = df[['ID', 'JUDUL']]
        df.columns = ['id', 'title'] 
        df.fillna('', inplace=True)
        return df
    except:
        return pd.DataFrame()

df_skripsi = load_data()

@st.cache_resource 
def build_bm25(df):
    tokenized_corpus = [preprocess_text(doc) for doc in df['title'].astype(str)]
    return BM25Okapi(tokenized_corpus)

bm25_model = build_bm25(df_skripsi) if not df_skripsi.empty else None

# =====================================================================
# TAMPILAN ANTARMUKA
# =====================================================================
with st.sidebar:
    st.markdown("## 🏛️ Informasi Sistem")
    st.markdown("---")
    st.markdown("**1. Metode Pencarian :** Algoritma BM25")
    st.markdown("**2. Koleksi Basis Data :** Dokumen Skripsi IT")

st.markdown("<h1 style='text-align: center; color: #000000;'>🎓 Sistem Pencarian Judul Skripsi</h1>", unsafe_allow_html=True)
query = st.text_input("🔍 Telusuri (Ketik judul atau kata kunci)...", key="search_bar")

if query:
    if bm25_model:
        tokenized_query = preprocess_text(query)
        doc_scores = bm25_model.get_scores(tokenized_query)
        df_results = df_skripsi.copy()
        df_results['similarity_score'] = doc_scores
        df_results = df_results[df_results['similarity_score'] > 0].sort_values(by='similarity_score', ascending=False)

        if not df_results.empty:
            st.markdown(f"""
            <div style="background-color: #dcfce7; padding: 15px; border-radius: 6px; border-left: 5px solid #16a34a; color: #000000 !important; margin-bottom: 20px;">
                <b>✓ Berhasil menemukan {len(df_results)} hasil relevan.</b>
            </div>
            """, unsafe_allow_html=True)
            
            for _, row in df_results.iterrows():
                st.markdown(f"""
                <div class="result-card">
                    <div class="score-badge">📊 Relevance Score: {row['similarity_score']:.4f}</div>
                    <div class="card-title">📑 {row['title']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #fee2e2; padding: 15px; border-radius: 6px; border-left: 5px solid #dc2626; color: #000000 !important; margin-bottom: 20px;">
                <b>⚠️ Dokumen tidak ditemukan.</b>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Dataset tidak dimuat.")
else:
    st.markdown("""
    <div style="text-align: center; padding: 50px; background-color: #ffffff; border-radius: 8px; color: #000000;">
        <h3>Selamat Datang</h3>
        <p>Silakan masukkan kata kunci untuk mencari referensi judul skripsi.</p>
    </div>
    """, unsafe_allow_html=True)
