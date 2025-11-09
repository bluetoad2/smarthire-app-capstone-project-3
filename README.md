# SmartHire App | AI Resume Assistant

**SmartHire** adalah aplikasi Streamlit RAG (Retrieval-Augmented Generation) untuk membantu proses rekrutmen: mencari kandidat dari dataset resume, menyusun shortlist, membuat email outreach, menghasilkan Interview question pack beserta scorecard, dan menganalisis dataset resume. Pembuatan project ini menggunakan OpenAI (LLM + embeddings), Qdrant sebagai vector DB, serta LangChain / LangGraph untuk agent/tool orchestration.

---
## Project Overview

Project ini merupakan **Capstone Project Module 3 - AI Engineering**, yang berfokus pada **penerapan Large Language Model (LLM)** dan **Retrieval-Augmented Generation (RAG)** untuk mengembangkan aplikasi AI yaitu **SmartHire**.

Tujuan utama dari project ini adalah untuk **mengotomatisasi proses seleksi kandidat** dengan memanfaatkan kemampuan model LLM dalam membaca, memahami, dan menganalisis ribuan resume secara efisien.

Seluruh implementasi dilakukan menggunakan **Streamlit** sebagai framework utama untuk membangun aplikasi interaktif, dengan integrasi **OpenAI API** sebagai model LLM, **Qdrant Vector Database** untuk penyimpanan embedding resume, serta **LangChain** dan **LangGraph** untuk orkestrasi agent dan tools.

Seluruh proses development dibuat di **local environment Python** menggunakan **VS Code / Streamlit**, dan di-deploy ke **Streamlit Cloud**.

---
## Ringkasan Fitur

### Fitur Utama
- Chatbot RAG untuk *retrieval* kandidat berdasarkan query (mis. “Shortlist 5 Data Scientists with Python & SQL”).
- Menampilkan snippets retrieval dari resume yang relevan.

### Fitur Tambahan
#### 1. Shortlist Manager ✅
- Simpan kandidat ke shortlist dari hasil pencarian.
- Tambah / edit notes per kandidat.
- Generate outreach email personal menggunakan LLM (single / bulk).
- Export shortlist ke CSV.

#### 2. Interview Generator & Scorecard ✅
- Pilih kandidat dari shortlist → generate interview pack (technical + behavioral questions) dan rubric dengan LLM.
- Isi scorecard (skor + notes) selama interview.
- Hitung total skor & export CSV.

#### 3. Resume Data Dashboard ✅
- Visualisasi kategori, distribusi panjang resume, top categories, histogram & boxplot.
- Viewer untuk menampilkan teks resume dan tombol download.

---

## Alur Kerja
1. **Ingest**: `Resume.csv` di-*chunk* dan di-*embed* menggunakan model embedding OpenAI, lalu disimpan ke Qdrant.
2. **Retrieval**: Chatbot memanggil tool `retrieve_resumes_tool` untuk mencari similarity di Qdrant.
3. **LLM / Agent**: Agent LangGraph / LangChain dengan model `gpt-4o-mini` menghasilkan shortlist, email, dan interview pack.
4. **UI**: Streamlit multi-page: Chatbot RAG, Shortlist Manager, Interview Generator, Resume Dashboard.

---

## Environment / Secrets
Dibuat file `.streamlit/secrets.toml` yang berisi:
```toml
QDRANT_URL = "https://xxxx.qdrant.cloud"
QDRANT_API_KEY = "xxxx"
QDRANT_COLLECTION = "resumes_v1"
OPENAI_API_KEY = "sk-xxxx"
```
---

## Dependencies
```
streamlit
openai
qdrant-client
langchain-openai
langchain-qdrant
langgraph
python-dotenv
pandas
numpy
plotly
altair
beautifulsoup4
tqdm
```

---

## Deployment (Streamlit Cloud)
Demo Aplikasi SmartHire dapat diakses di link berikut : https://smart-hire.streamlit.app/

---

## Cara Penggunaan Aplikasi
- Di halaman utama: ketik `Shortlist 5 Data Scientists with Python & SQL`.
- Tambahkan kandidat ke shortlist → buka **Shortlist Manager**.
- Generate email outreach / export shortlist.
- Di **Interview Generator** → pilih kandidat → generate interview pack → isi scorecard → export CSV.
- Di **Dashboard** → analisis distribusi resume dan lihat konten resume.

---

## Author
**Christopher Daniel Suryanaga**  
Capstone Project 3 – AI Engineer (Purwadhika)  
