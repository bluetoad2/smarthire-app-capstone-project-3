# Ingest Data dari Resume.csv ke Vector Database

import os
import sys
import uuid
from typing import List
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

# Import Library Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Import Library OpenAI embedding
try:
    from openai import OpenAI
except Exception as e:
    # Keluar jika library openai yang diperlukan (v1+) belum terinstal
    raise SystemExit("install openai v1+: pip install --upgrade openai") from e

# ----------------------------------------------------------------------
# Config
# Variabel untuk file input, chunking, dan interaksi API.
# ----------------------------------------------------------------------
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "Resume.csv"
COLLECTION_NAME = "resumes_v1" # Nama collection di Qdrant untuk menyimpan vector
CHUNK_SIZE = 1000             # Karakter maksimum per bagian teks (chunk) sebelum di-embed (menggunakan 1000)
CHUNK_OVERLAP = 200           # Overlap antar bagian (chunk) berurutan untuk menjaga konteks (menggunakan 200)
EMBEDDING_MODEL = "text-embedding-3-small" # Model embedding OpenAI yang digunakan untuk menghasilkan vektor
EMBED_BATCH_SIZE = 100        # Jumlah chunk teks yang dikirim ke API OpenAI dalam satu request
UPSERT_BATCH_SIZE = 64        # Jumlah poin (vektor) yang dikirim ke Qdrant dalam satu request upsert
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Setting Variabel API Keys dan URL
# Memastikan semua API Keys dan endpoint yang diperlukan telah diset.
# ----------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not (QDRANT_URL and QDRANT_API_KEY and OPENAI_API_KEY):
    raise SystemExit("set QDRANT_URL, QDRANT_API_KEY, and OPENAI_API_KEY env vars before running.")

# Membuat instance client OpenAI v1 untuk menghasilkan embedding
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------------------------------------------
# Fungsi Utility Tambahan untuk Persiapan Teks
# ----------------------------------------------------------------------

def strip_html(html: str) -> str:
    """
    Membersihkan konten HTML dengan menghapus tag dan mengekstrak teks.
    Dilakukan dengan BeautifulSoup untuk mengurai teks.
    """
    if not isinstance(html, str) or not html.strip():
        return ""
    soup = BeautifulSoup(html, "html.parser")
    # Menggunakan '\n' sebagai pemisah untuk menjaga pemisah paragraf
    return soup.get_text(separator="\n").strip()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Membagi teks panjang menjadi chunk yang lebih kecil dan tumpang tindih, untuk embedding.
    Hal ini memastikan setiap vektor merepresentasikan konteks yang menyambung.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # Memindahkan titik awal kembali sebesar jumlah tumpang tindih (overlap)
        start = max(end - overlap, start + 1)
    return chunks

def get_embeddings(texts: List[str], model: str = EMBEDDING_MODEL, batch_size: int = EMBED_BATCH_SIZE):
    """
    Menghasilkan embedding vector untuk daftar string teks 
    dengan memanggil OpenAI Embeddings API dalam mode batch.
    Hasilnya: daftar vector embedding (list[float])
    """
    embeddings = []
    # Memproses teks dalam batch untuk menjaga batas API dan efisien
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            # Memanggil API OpenAI untuk embedding
            resp = openai_client.embeddings.create(model=model, input=batch)
        except Exception as e:
            raise RuntimeError(f"OpenAI embeddings API error: {e}") from e
        
        # Mengekstrak data vector dari respons
        for item in resp.data:
            embeddings.append(item.embedding)
    return embeddings

def create_collection_if_missing(client: QdrantClient, name: str, dim: int):
    """
    Memeriksa apakah koleksi Qdrant ada dan membuatnya jika belum ada.
    Ini mendefinisikan dimensi vektor (dim) dan metrik jarak (COSINE).
    """
    if not client.collection_exists(name):
        # Membuat collection dengan konfigurasi vector yang ditentukan
        client.create_collection(
            collection_name=name, 
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print(f"Created collection {name} with dim {dim}")
    else:
        print(f"Collection {name} already exists")

# ----------------------------------------------------------------------
# Fungsi Utama untuk Ingest Vector
# ----------------------------------------------------------------------

def main():
    # Memuat file CSV sumber ke dalam pandas DataFrame
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    # 1. Menentukan kolom sumber untuk teks yang akan di-embed
    preferred = None
    if "Resume_str" in df.columns:
        preferred = "Resume_str"
        print("Akan menggunakan 'Resume_str' untuk embedding (prioritas).")
    elif "Resume_html" in df.columns:
        preferred = "Resume_html"
        print("Tidak ditemukan Resume_str , akan membersihkan Resume_html dan menggunakannya untuk embedding.")
    else:
        # Fallback: gunakan kolom string dengan panjang rata-rata terpanjang
        candidate_cols = [c for c in df.columns if df[c].dtype == object]
        if not candidate_cols:
            raise SystemExit("Tidak ditemukan kolom teks untuk di-embed.")
        avg_lens = {c: df[c].astype(str).map(len).mean() for c in candidate_cols}
        preferred = max(avg_lens, key=avg_lens.get)
        print(f"Tidak ada Resume_str/Resume_html. Menggunakan kolom '{preferred}' untuk embedding.")

    # Memulai koneksi clien Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Daftar untuk menyimpan chunk teks dan metadata terkait (payload)
    docs = []
    payloads = []

    # 2. Memproses setiap baris (resume) untuk menghasilkan chunk teks dan payload
    for idx, row in df.iterrows():
        text_raw = ""
        
        # Ekstrak dan bersihkan kolom teks utama
        if preferred == "Resume_html":
            text_raw = strip_html(row.get("Resume_html", ""))
        else:
            val = row.get(preferred, "")
            text_raw = str(val) if pd.notna(val) else ""
            # Gunakan 'Resume_html' sebagai cadangan jika kolom utama kosong
            if not text_raw and "Resume_html" in row and pd.notna(row["Resume_html"]):
                text_raw = strip_html(row["Resume_html"])

        # Fallback: jika teks masih kosong, gabungkan semua kolom teks lainnya
        if not text_raw:
            parts = []
            for c in df.columns:
                if c == "Resume_html": continue
                v = row.get(c)
                if pd.isna(v) or v is None: continue
                s = str(v).strip()
                if s: parts.append(f"{c}: {s}")
            text_raw = "\n".join(parts)

        if not text_raw:
            # Lewati baris ini jika tidak ada teks yang dapat diekstrak
            continue

        # Siapkan metadata dasar (payload) untuk baris ini, termasuk semua kolom non-embedded
        row_payload_base = {}
        for c in df.columns:
            if c == preferred: continue
            val = row.get(c)
            if pd.isna(val): continue
            
            # Memotong string panjang dalam payload untuk menghindari melebihi batas ukuran
            if isinstance(val, str) and len(val) > 1000:
                row_payload_base[c] = val[:1000] + " ...[truncated]"
            else:
                row_payload_base[c] = val

        # Chunk teks mentah dan buat entri dokumen/payload untuk setiap chunk
        chunks = chunk_text(text_raw)
        for ci, ch in enumerate(chunks):
            docs.append(ch)
            # Payload setiap chunk mencakup data baris dasar ditambah indeks chunk
            payload = {"row_index": int(idx), "chunk_index": ci, **row_payload_base}
            if "ID" in df.columns and pd.notna(row.get("ID")):
                payload["ID"] = row.get("ID")
            payloads.append(payload)

    total_docs = len(docs)
    print(f"Prepared {total_docs} chunks for embedding and upload.")

    if total_docs == 0:
        print("No text chunks prepared. Exiting.")
        return

    # 3. Embed dan Upsert ke Qdrant
    all_ids = []
    i = 0
    # Menggunakan tqdm untuk menampilkan progres untuk proses yang berjalan lama
    pbar = tqdm(total=total_docs, desc="Embedding+Upserting")
    
    while i < total_docs:
        # Menentukan window batch untuk embedding dan upserting
        j = min(i + EMBED_BATCH_SIZE, total_docs)
        batch_texts = docs[i:j]
        batch_payloads = payloads[i:j]
        
        # Menghasilkan embedding untuk batch chunk teks saat ini
        batch_embs = get_embeddings(batch_texts)

        # Pertama kali dijalankan, tentukan dimensi embedding dan buat koleksi Qdrant
        if i == 0:
            emb_dim = len(batch_embs[0])
            create_collection_if_missing(client, COLLECTION_NAME, emb_dim)

        # Mengkonversi embedding dan payload menjadi Qdrant PointStructs
        points = []
        for k, emb in enumerate(batch_embs):
            uid = str(uuid.uuid4()) # Menghasilkan ID unik untuk setiap poin vektor
            # Teks chunk lengkap disimpan dalam payload untuk pengambilan
            payload = {"text": batch_texts[k], **batch_payloads[k]}
            points.append(PointStruct(id=uid, vector=emb, payload=payload))
            all_ids.append(uid)

        # Mengunggah poin ke Qdrant dalam sub-batch yang lebih kecil
        for b in range(0, len(points), UPSERT_BATCH_SIZE):
            sub = points[b : b + UPSERT_BATCH_SIZE]
            client.upsert(collection_name=COLLECTION_NAME, points=sub, wait=True)

        i = j
        pbar.update(len(batch_texts)) # Update Progress Bar

    pbar.close()
    print("Ingestion done. Total points:", len(all_ids))

    # 4. Tes Query
    # Melakukan tes pencarian sederhana untuk konfirmasi ingestinya berhasil.
    test_query = "senior backend developer with python experience"
    test_emb = get_embeddings([test_query])[0]
    hits = client.search(collection_name=COLLECTION_NAME, query_vector=test_emb, limit=5)
    print("Top hits:")
    # Mencetak ID, skor kemiripan, dan potongan payload yang tersimpan untuk verifikasi
    for h in hits:
        print("id:", h.id, "score:", getattr(h, "score", None), "payload snippet:", str(h.payload)[:200])


if __name__ == "__main__":
    main()
