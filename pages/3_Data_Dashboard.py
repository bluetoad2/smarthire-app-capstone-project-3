# Resume Interactive Data Dashboard

# Import Library yang diperlukan.
import os
from pathlib import Path
import streamlit as st 
import pandas as pd     
import numpy as np     
import plotly.express as px # Plotly untuk visualisasi interaktif (bar chart, histogram).
import altair as alt    # Altair untuk visualisasi boxplot.

# ---------- Page config ----------
st.set_page_config(page_title="SmartHire | Resume Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# ---------- Title + Logo ----------
# Membuat layout dua kolom untuk header: untuk logo dan judul.
col_logo, col_title = st.columns([1, 8])
with col_logo:
    logo_path = Path("logo.png")
    if logo_path.exists():
        st.image(str(logo_path), width=250)
with col_title:
    st.markdown("<h1 style='margin:0; padding:0'>Resume Dataset Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Interactive Insights into Resume Categories, Lengths, and Content.")

st.markdown("---") # Menambahkan pemisah horizontal

# ---------- Load data ----------
DATA_PATH = "Resume.csv"
@st.cache_data
def load_data(path):
    # Membaca data dari file CSV yang ditentukan ke dalam Pandas DataFrame.
    df = pd.read_csv(path)
    # Memastikan kolom ada
    expected = {"ID", "Resume_str", "Resume_html", "Category"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Membersihkan kolom 'Category' dengan memastikan string dan menghapus spasi.
    df["Category"] = df["Category"].astype(str).str.strip()
    # Menghitung dua fitur baru: 'word_count' (jumlah kata) dan 'char_count' (jumlah karakter) dari teks resume ('Resume_str').
    df["resume_text"] = df["Resume_str"].fillna("").astype(str)
    df["word_count"] = df["resume_text"].str.split().apply(len)
    df["char_count"] = df["resume_text"].str.len()
    return df

try:
    # Load data.
    df = load_data(DATA_PATH)
except Exception as e:
    # Menampilkan error dan stop aplikasi jika load data gagal.
    st.error(f"Error loading data from {DATA_PATH}: {e}")
    st.stop()

# ---------- Global filters ----------
# Mengekstrak semua kategori unik untuk opsi filter.
all_categories = sorted(df["Category"].dropna().unique())
# Membuat filter multi-selection yang membuat user dapat memilih kategori yang disertakan dalam dashboard.
selected_categories = st.multiselect("Select categories to filter dashboard (leave empty = all):",
                                     options=all_categories,
                                     default=all_categories)

# Memfilter DataFrame utama berdasarkan pilihan user
filtered_df = df[df["Category"].isin(selected_categories)] if selected_categories else df.copy()

st.markdown("""
<style>
/* Target value */
[data-testid="stMetricValue"] {
    font-size: 9rem;
}

/* Target label (small text) */
[data-testid="stMetricLabel"] {
    font-size: 7rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Top metrics row ----------
# Menghitung jumlah total resume yang tersisa setelah menerapkan filter kategori.
total_count = len(filtered_df)
# Membuat layout dua kolom untuk tampilan metrik main summary.
col1, col2 = st.columns([1, 2])
with col1:
    # Menampilkan jumlah total resume yang difilter.
    st.metric(label="Total Resumes (Filtered)", value=f"{total_count:,}")
with col2:
    # Menghitung dan menampilkan ringkasan tabel untuk 10 kategori teratas.
    counts = filtered_df["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
    st.write("Top Categories (By Count):")
    st.dataframe(counts.head(10), height=150)

st.markdown("---")

# ---------- Two-column layout: Distribution (left) | Pie (right) ----------
dist_col, pie_col = st.columns([3, 1.3])

with dist_col:
    st.subheader("Resume Category Distribution")
    # Control user untuk memilih kategori diurutkan dalam bar chart
    sort_option = st.selectbox("Sort Categories by:", options=["Count (desc)", "Alphabetical"])

    # Mempersiapkan data untuk bar chart
    category_counts = filtered_df["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
    if sort_option == "Alphabetical":
        category_counts = category_counts.sort_values("Category")
    else:
        category_counts = category_counts.sort_values("Count", ascending=False)

    # Menghasilkan bar chart interaktif menggunakan Plotly
    fig_bar = px.bar(category_counts,
                     x="Category",
                     y="Count",
                     color="Category",
                     title="Resumes per Category",
                     labels={"Count": "Number of Resumes", "Category": "Job Category"})
    # Menyesuaikan layout
    fig_bar.update_layout(xaxis_tickangle=-45, margin=dict(t=50, b=150))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.caption("Click category bars in the chart to focus (doesn't change dashboard filter — use the multi-select above).")

with pie_col:
    st.subheader("Category Share (Pie Chart)")
    if category_counts.empty:
        st.info("No data for selected filter.")
    else:
        # Menghasilkan pie chart dengan Plotly.
        fig_pie = px.pie(category_counts, names="Category", values="Count", title="Category Proportion", hole=0.35)
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ---------- Resume Length and Content Analysis ----------
st.subheader("Resume Length & Content Analysis")

# Membuat layout dua kolom untuk bagian analisis panjang kata dari resume.
len_col1, len_col2 = st.columns([2, 1.2])

with len_col1:
    # Memungkinkan user untuk memilih satu kategori atau 'All'
    inspect_category = st.selectbox("Choose a Category to Inspect (All Shows Combined):",
                                     options=["All"] + all_categories,
                                     index=0)
    # Memfilter data
    length_df = filtered_df.copy()
    if inspect_category != "All":
        length_df = length_df[length_df["Category"] == inspect_category]

    # Membuat slider untuk memfilter rentang jumlah kata
    min_wc = int(length_df["word_count"].min()) if not length_df.empty else 0
    max_wc = int(length_df["word_count"].max()) if not length_df.empty else 1000
    wc_range = st.slider("Word Count Range (Filter Resumes shown below):",
                          min_value=0, max_value=max(1000, max_wc),
                          value=(min_wc, max_wc), step=1)

    # Menerapkan filter slider jumlah kata ke DataFrame khusus panjang.
    length_df = length_df[(length_df["word_count"] >= wc_range[0]) & (length_df["word_count"] <= wc_range[1])]

    # Menghasilkan histogram untuk memvisualisasikan distribusi jumlah kata.
    fig_hist = px.histogram(length_df, x="word_count", nbins=40,
                            title=f"Word Count Distribution ({'All' if inspect_category=='All' else inspect_category})",
                            labels={"word_count": "Word count", "count": "Number of Resumes"})
    st.plotly_chart(fig_hist, use_container_width=True)

with len_col2:
    # Kolom visualisasi Boxplot.
    st.write("Boxplot of Resume Word Counts")
    if length_df.empty:
        st.info("No resumes in current selection/range.")
    else:
        # Menghasilkan boxplot menggunakan Altair, yang menunjukkan median, kuartil, dan outlier dari jumlah kata.
        box = alt.Chart(length_df).mark_boxplot(extent=1.5).encode(
            y=alt.Y("word_count:Q", title="Word count")
        ).properties(
            # Mengatur tinggi agar sesuai dengan kolom.
            height=600
        )
        st.altair_chart(box, use_container_width=True)

st.markdown("---")

# ---------- Detailed table and viewer ----------
st.subheader("Explore Resumes")

# Memilih kolom tertentu dan mengurutkan data berdasarkan jumlah kata untuk preview tabel.
preview_cols = ["ID", "Category", "word_count", "char_count"]
table_df = length_df[preview_cols].sort_values(by="word_count", ascending=False).reset_index(drop=True)

st.write(f"Showing {len(table_df)} Resumes (After Category & Word-count Filter).")
# Menampilkan tabel terbatas (top 50) dari resume yang saat ini ada dalam set yang difilter.
if not table_df.empty:
    st.dataframe(table_df.head(50), height=300)

    # User dapat memilih ID Resume tertentu dari tabel untuk melihat isi lengkapnya.
    selected_id = st.selectbox("Select a Resume ID to View full text:", options=["-- none --"] + table_df["ID"].astype(str).tolist())
    if selected_id != "-- none --":
        # Mengambil baris data lengkap untuk ID yang dipilih.
        sel_row = length_df[length_df["ID"].astype(str) == str(selected_id)]
        if not sel_row.empty:
            # Mengekstrak teks raw dan menampilkan ringkasan.
            resume_text = sel_row.iloc[0]["resume_text"]
            st.markdown(f"**ID:** {selected_id} — **Category:** {sel_row.iloc[0]['Category']}")
            # Expander untuk teks resume lengkap.
            with st.expander("Full resume text", expanded=True):
                st.text_area("Resume text", value=resume_text, height=400)
            # Tombol untuk download teks resume yang ditampilkan sebagai file .txt.
            st.download_button("Download resume text (.txt)", data=resume_text, file_name=f"{selected_id}.txt")
        else:
            st.warning("Selected resume not found in current filtered set.")
else:
    st.info("No resumes match current filters. Try widening the category selection or word-count range.")

st.markdown("---")

# ---------- Footer ----------
st.caption("Dataset: Resume.csv | contains resume text and categories. (Data Features: ID, Resume_str, Resume_html, Category).")
st.caption("© 2025 SmartHire App | by Christopher Daniel S | Capstone Project 3 - AI Engineering")
