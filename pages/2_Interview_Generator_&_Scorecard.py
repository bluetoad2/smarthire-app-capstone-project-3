# pages/2_Interview_Generator_&_Scorecard.py

# Import Library yang diperlukan
import os
import streamlit as st
import pandas as pd
import json
import re
from typing import Dict, Any

# Import Library yang diperlukan dari LangChain dan LangGraph untuk agen LLM
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# --- Konfigurasi ---
# Ambil API Keys OpenAI dari Streamlit secrets.toml
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
# Inisialisasi model ChatOpenAI, menggunakan gpt-4o-mini
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
# Buat agen ReAct untuk menggunakan tools
interview_agent = create_react_agent(model=llm, tools=[])


# --- Main Logic App: Pembuatan Set Pertanyaan Interview---
def generate_interview_pack(candidate: Dict[str, Any], job_title: str) -> Dict[str, Any]:
    """
    Use the agent to generate interview questions, rubric, etc.
    """
    # Ekstrak dua snippet kandidat (highlight) pertama untuk menyesuaikan prompt.
    candidate_snippet = " ".join(candidate.get("snippets", [])[:2])
    
    # Tentukan prompt detail yang menginstruksikan peran agen dan format JSON hasil.
    prompt = (
        "You are an interviewing assistant. Return ONLY a JSON object with keys: "
        "technical_questions, behavioral_questions, rubric.\n"
        "technical_questions: array of {\"q\": string, \"suggested_max_score\": int}\n"
        "behavioral_questions: array of {\"q\": string, \"suggested_max_score\": int}\n"
        "rubric: array of {\"criterion\": string, \"description\": string}\n\n"
        f"Job title: {job_title}\n"
        f"Candidate highlight: {candidate_snippet}\n\n"
        "Provide 4-5 technical Qs, 3-4 behavioral Qs, and 3-4 rubric criteria."
    )
    # Susun input sebagai daftar objek message untuk agen chatbot.
    input_messages = [{"role": "system", "content": "You are an interviewing assistant that returns JSON only."}, {"role": "user", "content": prompt}]
    
    try:
        # Invoke agen LLM dengan prompt yang terstruktur.
        result = interview_agent.invoke({"messages": input_messages})
        # Ekstrak konten dari chat terakhir asisten.
        assistant_content = result.get("messages", [])[-1].content
        # Gunakan regex untuk menemukan dan mengekstrak objek JSON.
        match = re.search(r'\{.*\}', assistant_content, re.DOTALL)
        if match:
            # Uraikan string JSON yang diekstrak dan return interview pack.
            return json.loads(match.group(0))
    except Exception:
        # Jika invoke LLM gagal atau terjadi error parsing JSON, catch exception.
        pass

    # Backup Interview Pack: return default jika invoke LLM gagal.
    return {
        "technical_questions": [{"q": f"Explain a project relevant to {job_title}.", "suggested_max_score": 5}],
        "behavioral_questions": [{"q": "Tell me about a time you worked with a difficult stakeholder.", "suggested_max_score": 5}],
        "rubric": [{"criterion": "Technical Proficiency", "description": "Correctness and depth of knowledge."}]
    }


# --- Config UI Streamlit ---
# Config layout, title, dan icon.
st.set_page_config(page_title="SmartHire | Interview Tools", page_icon="📝", layout="wide")
st.title("📝 Interview Generator & Scorecard")

# Tambahkan tombol untuk kembali ke page chatbot utama.
if st.button("⬅️ Back to Chat"):
    st.switch_page("Smart_Hire_App.py")

# Periksa apakah data yang diperlukan (shortlist) ada dalam session state.
if not st.session_state.get("shortlist"):
    # Tampilkan pesan informasi dan stop eksekusi skrip jika shortlist masih kosong.
    st.info("Your shortlist is empty. Add candidates on the main page to generate interview packs.")
    st.stop()

# --- UI Dropdown untuk memilih kandidat ---
# Buat dropdown untuk memilih kandidat dari shortlist yang ada.
selected_for_interview = st.selectbox(
    "Pick a Shortlisted Candidate to Generate an Interview Pack for:",
    options=list(st.session_state.shortlist.keys()),
    format_func=lambda x: f"Candidate ID: {x}" if x else "Select a candidate",
    index=None,
    placeholder="Select a candidate..."
)

# --- Membuat Interview Question Pack ---
if selected_for_interview:
    # Ambil entri lengkap dan detail kandidat dari session state.
    entry = st.session_state.shortlist[selected_for_interview]
    candidate = entry["candidate"]

    st.markdown("---")
    st.header(f"Generate Interview Pack for `{selected_for_interview}`")

    # Kolom input bagi user untuk menentukan job title.
    job_title_input = st.text_input("Job title / role for Interview Pack", value="Software Engineer", key=f"int_job_{selected_for_interview}")
    
    # Tombol untuk invoke LLM untuk membuat interview pack.
    if st.button("Generate Interview Pack"):
        with st.spinner("Generating interview questions and rubric..."):
            # Panggil fungsi untuk menghasilkan konten.
            pack = generate_interview_pack(candidate, job_title_input)
            # Simpan interview pack yang baru dibuat dalam session state.
            st.session_state.shortlist[selected_for_interview]["interview_pack"] = pack
            st.success("Interview pack generated.")
            # Hapus data scorecard yang ada untuk memulai yang baru dengan pertanyaan baru.
            st.session_state.scorecards.pop(f"scorecard_{selected_for_interview}", None)
            # Jalankan ulang skrip untuk segera menampilkan interview pack yang dihasilkan.
            st.rerun()

    # --- Tampilan Interview Pack & Scorecard ---
    # Lanjutkan hanya jika Interview Pack telah berhasil dibuat dan disimpan.
    if "interview_pack" in st.session_state.shortlist[selected_for_interview]:
        pack = st.session_state.shortlist[selected_for_interview]["interview_pack"]
        
        st.markdown("---")
        st.header(f"Interview Scorecard for `{selected_for_interview}`")

        # Gunakan kolom untuk memisahkan pertanyaan dan rubrik secara rapi.
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Generated Questions")
            # Ulangi dan tampilkan semua pertanyaan teknis dan skor maksimumnya.
            st.markdown("**Technical Questions**")
            for i, tq in enumerate(pack.get("technical_questions", []), 1):
                st.markdown(f"**{i}.** {tq.get('q')} *(max: {tq.get('suggested_max_score', 5)})*")
            # Ulangi dan tampilkan semua pertanyaan perilaku dan skor maksimumnya.
            st.markdown("**Behavioral Questions**")
            for i, bq in enumerate(pack.get("behavioral_questions", []), 1):
                st.markdown(f"**{i}.** {bq.get('q')} *(max: {bq.get('suggested_max_score', 5)})*")
        
        with col2:
            st.subheader("Suggested Rubric")
            # Tampilkan setiap kriteria rubrik dan deskripsi rincinya.
            for r in pack.get("rubric", []):
                st.markdown(f"- **{r.get('criterion')}**: {r.get('description')}")
        
        st.markdown("---")
        st.subheader("Enter Scores and Notes")
        
        sc_key = f"scorecard_{selected_for_interview}"
        # Inisialisasi struktur kartu skor dalam status sesi jika belum ada.
        if sc_key not in st.session_state.scorecards:
            rows = []
            # Buat entri untuk setiap pertanyaan teknis.
            for tq in pack.get("technical_questions", []):
                rows.append({"question": tq["q"], "type": "technical", "max_score": tq.get("suggested_max_score", 5), "score": 0, "notes": ""})
            # Buat entri untuk setiap pertanyaan perilaku.
            for bq in pack.get("behavioral_questions", []):
                rows.append({"question": bq["q"], "type": "behavioral", "max_score": bq.get("suggested_max_score", 5), "score": 0, "notes": ""})
            st.session_state.scorecards[sc_key] = rows

        # Render scorecard yang dapat diedit
        rows = st.session_state.scorecards[sc_key]
        total_score, max_score = 0, 0
        # Ulangi setiap pertanyaan untuk membuat kolom input untuk skor dan catatan.
        for idx, row in enumerate(rows):
            with st.container(border=True):
                # Tampilkan pertanyaan dan jenisnya (Teknis/Perilaku).
                st.markdown(f"**{idx+1}. [{row['type'].capitalize()}]** {row['question']}")
                c1, c2 = st.columns([1, 2])
                # Input numerik untuk skor, dibatasi oleh skor maksimum.
                score = c1.number_input(f"Score", min_value=0, max_value=int(row['max_score']), value=int(row['score']), key=f"sc_{sc_key}_{idx}", label_visibility="collapsed")
                # Area teks untuk catatan user tentang jawaban.
                note = c2.text_area(f"Notes", value=row['notes'], key=f"sc_notes_{sc_key}_{idx}", height=60, label_visibility="collapsed", placeholder="Interviewer notes...")
                
                # Perbarui skor dan notes dalam session state segera setelah input berubah.
                st.session_state.scorecards[sc_key][idx]["score"] = score
                st.session_state.scorecards[sc_key][idx]["notes"] = note
                # Akumulasi skor saat ini dan skor total yang mungkin.
                total_score += score
                max_score += row['max_score']
        
        # Tampilkan metrik total skor akhir yang dihitung kepada user.
        st.metric(label="Total Score", value=f"{total_score} / {max_score}")

        # --- Export Scorecard ---
        # Data terstruktur untuk export CSV.
        sc_rows = []
        for r in st.session_state.scorecards[sc_key]:
            sc_rows.append({
                "CandidateID": selected_for_interview,
                "Question": r["question"], "Type": r["type"],
                "MaxScore": r["max_score"], "Score": r["score"], "Notes": r["notes"]
            })
        # Konversi daftar dictionary scorecard menjadi DataFrame Pandas.
        df_sc = pd.DataFrame(sc_rows)
        # Konversi DataFrame menjadi string CSV dan kodekan untuk download.
        csv_bytes = df_sc.to_csv(index=False).encode("utf-8")
        # Tombol Streamlit untuk download file CSV scorecard.
        st.download_button("Download Scorecard as CSV", data=csv_bytes, file_name=f"scorecard_{selected_for_interview}.csv", mime="text/csv")
