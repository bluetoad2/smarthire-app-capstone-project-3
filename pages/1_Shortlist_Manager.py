# pages/1_Shortlist_Manager.py

# Import libraries yang diperlukan
import os
import streamlit as st
import pandas as pd
import json
import re
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Ambil API Keys dari Streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# Inisialisasi model ChatOpenAI, dengan model gpt-4o-mini.
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Buat agen ReAct dasar dari LangGraph untuk handling generate email.
email_agent = create_react_agent(model=llm, tools=[])

# --- Fungsi Utama LLM: Generate Outreach Email ---
# Fungsi ini untuk mengirim request ke agen LLM untuk membuat outreach email yang personalized.
def generate_outreach_email(candidate: Dict[str, Any], job_title: str, recruiter_name: str = "Recruiter") -> Dict[str, str]:
    """
    Ask the agent to create a personalized outreach email.
    """
    candidate_id = candidate.get("ID")
    # Ambil dua kutipan resume pertama untuk personalisasi dalam prompt.
    candidate_snippet = " ".join(candidate.get("snippets", [])[:2])
    
    # Susun prompt detail yang membuat LLM untuk hanya menghasilkan objek JSON.
    prompt = (
        "You write short, professional candidate outreach emails for recruiters.\n"
        "Output ONLY a JSON object with keys: subject, body.\n"
        "Be concise and personalize using the candidate's info and a resume highlight.\n\n"
        f"Candidate ID: {candidate_id}\n"
        f"Job title: {job_title}\n"
        f"Highlight (from resume): {candidate_snippet}\n\n"
        "Example output: {\"subject\": \"...\", \"body\": \"...\"}\n"
    )

    # Menyiapkan message input, termasuk instruksi system prompt untuk LLM.
    input_messages = [
        {"role": "system", "content": "You are a recruiter-assistant composing outreach emails in JSON format."},
        {"role": "user", "content": prompt}
    ]
    try:
        # Panggil agen LangGraph untuk menghasilkan konten email berdasarkan prompt.
        result = email_agent.invoke({"messages": input_messages})
        last_message = result.get("messages", [])[-1]
        assistant_content = getattr(last_message, "content", "")
        
        # Ekstrak string JSON raw dari respons agen.
        match = re.search(r'\{.*\}', assistant_content, re.DOTALL)
        if match:
            # Urai string JSON yang diekstrak menjadi dictionary Python.
            parsed = json.loads(match.group(0))
            # Validasi objek yang diurai mengandung key yang diperlukan sebelum di return.
            if isinstance(parsed, dict) and "subject" in parsed and "body" in parsed:
                return {"subject": parsed["subject"], "body": parsed["body"]}
    except Exception:
        # Menangkap error selama call agen atau parsing JSON dan gunakan templat sebagai gantinya.
        pass  

    # Definisikan email template non-LLM sebagai backup jika generate gagal.
    subject = f"Opportunity: {job_title}"
    body = (
        f"Hi Candidate {candidate_id},\n\n"
        f"My name is {recruiter_name}, and I came across your profile. I was impressed by your experience, particularly: '{candidate_snippet or 'your background'}'.\n"
        f"We're hiring for a {job_title} role, and I think you could be a great fit. Are you open to a brief call to discuss?\n\n"
        "Best,\n"
        f"{recruiter_name}"
    )
    return {"subject": subject, "body": body}

# --- Config UI Streamlit ---

# Konfigurasi layout, title, icon.
st.set_page_config(page_title="SmartHire | Shortlist Manager", page_icon="📋", layout="wide")

# Title utama aplikasi.
st.title("📋 Shortlist Manager")

# Tombol untuk kembali ke page chat utama.
if st.button("⬅️ Back to Chat"):
    st.switch_page("Smart_Hire_App.py")

# Periksa apakah shortlist ada di status session; jika tidak, tampilkan pesan dan hentikan rendering.
if not st.session_state.get("shortlist"):
    st.info("Your shortlist is empty. Go to the main page to find and add candidates.")
    st.stop()

# --- Tampilkan Tabel Data Shortlist ---
# Siapkan data kandidat dari status session ke dalam list of dictionaries untuk tampilan tabel.
shortlist_rows = []
for cid, entry in st.session_state.shortlist.items():
    candidate = entry["candidate"]
    shortlist_rows.append({
        "CandidateID": cid,
        "Category": candidate.get("Category"),
        "Notes": entry.get("notes", ""),
    })
    
# Konversi daftar baris menjadi Pandas DataFrame
df_short = pd.DataFrame(shortlist_rows)
# Tampilkan shortlist dalam tabel Streamlit
st.dataframe(df_short, use_container_width=True)

st.markdown("---")
# Header untuk bagian interaksi
st.header("Manage & Contact Candidates")

# --- Loop Manager Kandidat Individu ---
# Ulangi melalui setiap kandidat dalam shortlist status session untuk action per-kandidat.
for cid, entry in st.session_state.shortlist.copy().items():
    candidate = entry["candidate"]
    with st.container(border=True):
        # Tampilkan ID kandidat.
        st.markdown(f"### Candidate ID: `{cid}`")
        
        # Allow user untuk mengedit dan menyimpan notes untuk kandidat, menyimpan dalam status session.
        notes = st.text_area(f"Notes for {cid}", value=entry.get("notes", ""), key=f"short_notes_{cid}", height=80)
        st.session_state.shortlist[cid]["notes"] = notes

        # Input column untuk menentukan job title yang akan dijadikan subjek email.
        job_title = st.text_input(f"Job title for outreach email", value="Software Engineer", key=f"job_{cid}")

        col_a, col_b = st.columns([1, 2])
        with col_a:
            # Tombol untuk invoke fungsi pembuatan email LLM untuk kandidat.
            if st.button(f"Generate outreach email", key=f"gen_email_{cid}"):
                with st.spinner("Generating outreach email..."):
                    # Panggil fungsi untuk menghasilkan email dan simpan hasilnya.
                    email = generate_outreach_email(candidate, job_title)
                    st.session_state.shortlist[cid]["outreach"] = email
                    st.success("Outreach generated.")
                    # Jalankan ulang skrip untuk segera memperbarui UI dengan konten email baru.
                    st.rerun()

        # Tampilkan email yang dihasilkan
        if st.session_state.shortlist[cid].get("outreach"):
            with st.expander("View Generated Outreach Email", expanded=True):
                out = st.session_state.shortlist[cid]["outreach"]
                # Tampilkan subjek dan isi dalam input teks
                st.markdown("**Subject:**")
                st.text_input("Subject", value=out.get("subject"), key=f"out_subj_{cid}", label_visibility="collapsed")
                st.markdown("**Body:**")
                st.text_area("Body", value=out.get("body"), key=f"out_body_{cid}", height=200, label_visibility="collapsed")

st.markdown("---")
# Header untuk action yang mempengaruhi seluruh shortlist (Bulk).
st.header("Bulk Actions")

col1, col2 = st.columns(2)

with col1:
    # --- Pembuatan Email secara Bulk (All) ---
    # Tombol untuk membuat email outreach untuk semua kandidat.
    if st.button("Generate outreach for all shortlisted (bulk)", use_container_width=True):
        # Inisialisasi progress bar untuk menampilkan status secara visual.
        progress_bar = st.progress(0, "Starting bulk outreach generation...")
        shortlist_items = list(st.session_state.shortlist.items())
        total = len(shortlist_items)
        with st.spinner("Generating emails..."):
            # Ulangi melalui semua kandidat dan buat email hanya jika email belum ada.
            for i, (cid, entry) in enumerate(shortlist_items):
                if not entry.get("outreach"):
                    candidate = entry["candidate"]
                    # Gunakan posisi pekerjaan default untuk bulk action.
                    email = generate_outreach_email(candidate, job_title="Software Engineer")
                    st.session_state.shortlist[cid]["outreach"] = email
                # Update progress bar setiap kandidat diproses.
                progress_bar.progress((i + 1) / total, f"Generated email for Candidate {cid}...")
        st.success("Bulk outreach generation complete.")
        # Jalankan ulang untuk menampilkan email yang baru dihasilkan di bagian kandidat individual.
        st.rerun()

with col2:
    # --- Export to CSV ---
    # Struktur data untuk menyertakan semua kolom yang diperlukan untuk ekspor, termasuk konten email yang dihasilkan.
    rows_to_export = []
    for cid, entry in st.session_state.shortlist.items():
        cand = entry["candidate"]
        outreach = entry.get("outreach") or {}
        rows_to_export.append({
            "CandidateID": cid,
            "Category": cand.get("Category"),
            "Notes": entry.get("notes", ""),
            "Outreach_Subject": outreach.get("subject", ""),
            "Outreach_Body": outreach.get("body", "")
        })
        
    # Buat DataFrame akhir dari data Export.
    df_export = pd.DataFrame(rows_to_export)
    # Konversi DataFrame ke string CSV dan encode menjadi byte untuk tombol Download.
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    # Tampilkan tombol Download Streamlit.
    st.download_button(
        "Download Shortlist as CSV",
        data=csv_bytes,
        file_name="candidate_shortlist.csv",
        mime="text/csv",
        use_container_width=True
    )
