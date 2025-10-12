# SmartHire App - Dataset Resumes
# Capstone Project 3 - AI Engineering
# By Christopher Daniel Suryanaga

"""
SmartHire - Streamlit RAG app
- Page chatbot utama untuk retrieval kandidat.
"""

# Import Modul untuk operasi sistem dan handling data.
import os
import json
import re
from typing import List, Dict, Any, Optional

# Import Streamlit untuk membuat UI Aplikasi.
import streamlit as st
# Import utility untuk load variabel enviroment dari file .env (API Keys).
from dotenv import load_dotenv

# Import LangChain dan OpenAI untuk LLM, Embeddings, dan Vector Store.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

# Import Qdrant client untuk retrieval.
from qdrant_client import QdrantClient

# Load env variabel yang berisi API Keys dan URL.
load_dotenv()

# ----------------------------
# Page Configuration
# ----------------------------
# Mengatur layout, titile, dan icon page Streamlit.
st.set_page_config(
    page_title="SmartHire | AI Resume Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi Layar Login
def login_screen():
    """Menampilkan layar login dan autentikasi menggunakan users.json."""
    # Hide sidebar di layar login
    st.markdown(
        """<style>[data-testid="stSidebar"] {display: none !important;}</style>""",
        unsafe_allow_html=True
    )

    # Center kolom login menggunakan kolom Streamlit.
    _, center_col, _ = st.columns([1, 1.2, 1])

    with center_col:
        col_left, col_img, col_right = st.columns([1, 1, 1])

        with col_img:
            # Tampilkan logo aplikasi.
            st.image("logo.png", width=300) 
            
        # Teks intro dan credentials untuk demo.
        st.markdown("<p style='text-align: center;'>Welcome to SmartHire!, your Generative AI Resume Assistant.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>For Demo Purposes, use this credentials : (user: admin, password: admin123)</p>", unsafe_allow_html=True)

        # Buat form login untuk input user.
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            submitted = st.form_submit_button("Login")

            if submitted:
                try:
                    # Read credential dari file JSON.
                    with open("users.json", "r") as f:
                        credentials = json.load(f)
                    
                    correct_username = credentials.get("username")
                    correct_password = credentials.get("password")
                    
                    # Verifikasi credential.
                    if username == correct_username and password == correct_password:
                        # Atur status session menjadi terautentikasi (True).
                        st.session_state.authenticated = True
                        # Jalankan ulang skrip untuk memuat aplikasi utama.
                        st.rerun()
                    else:
                        st.error("The username or password you entered is incorrect.")
                except FileNotFoundError:
                    st.error("`users.json` file not found.")
                except json.JSONDecodeError:
                    st.error("Error reading `users.json`.")


# --- APLIKASI UTAMA (MAIN APP) ---
def main_app():
    # Mengatur content dan instruction untuk sidebar aplikasi.
    with st.sidebar:
        st.image("logo.png", width=200)
        st.header("SmartHire Tools")
        # Teks untuk tutorial cara menggunakan aplikasi.
        st.caption(
            """
            Welcome! Here's how to use the app:
            1.  **Chat to Find Candidates:** On this main page, type your requirements into the chatbot (e.g., "Find me 5 Best Chef Candidates").
            2.  **Add to Shortlist:** Review the retrieved candidates. Use the 'Add to shortlist' button for any promising profiles.
            3.  **Manage Shortlist:** Navigate to the **Shortlist Manager** page to see all your saved candidates, add notes, and generate personalized outreach emails.
            4.  **Create Interview Packs:** Go to the **Interview Generator** page. Select a candidate from your shortlist to generate interview questions and a scoring rubric.
            """
        )

    # Load API Keys dan URL dari file secrets (secrets.toml).
    QDRANT_URL = st.secrets.QDRANT_URL
    QDRANT_API_KEY = st.secrets.QDRANT_API_KEY
    OPENAI_API_KEY = st.secrets.OPENAI_API_KEY
    COLLECTION_NAME = st.secrets.QDRANT_COLLECTION

    # Stop aplikasi jika ada credential yang kurang atau hilang.
    if not all([QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY]):
        st.error("Missing required secrets. Set QDRANT_URL, QDRANT_API_KEY, and OPENAI_API_KEY in Streamlit secrets.")
        st.stop()

    # Inisialisasi Qdrant client untuk action database vektor.
    qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Menggunakan caching resource Streamlit (@st.cache_resource) untuk menghindari inisialisasi ulang-
    # -objek yang berat (LLM, Embeddings, Qdrant Client) pada setiap rerun.
    @st.cache_resource
    def get_llm_and_qdrant():
        # Inisialisasi LLM (GPT-4o-mini)
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
        # Inisialisasi model embedding untuk mengubah teks menjadi representasi vektor.
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        # Inisialisasi instance Qdrant Client untuk wrapper LangChain.
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Siapkan wrapper LangChain Qdrant untuk pencarian kemiripan (similarity search).
        qdrant = Qdrant(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
            # Tentukan kunci payload mana yang berisi konten teks resume.
            content_payload_key="text",
            metadata_payload_key=None,
        )
        return llm, qdrant
    
    # Ambil instance LLM dan Qdrant yang sudah diinisialisasi.
    llm, qdrant = get_llm_and_qdrant()

    # Fungsi pembantu untuk membagi blok teks menjadi kalimat individual.
    def _split_sentences(text: str) -> List[str]:
        # Memisahkan teks berdasarkan tanda baca akhir kalimat.
        sentences = re.split(r'(?<=[\.\?\!\n])\s+', text.replace("\r", " ").strip())
        return [s.strip() for s in sentences if s.strip()] or [text.strip()]

    # Fungsi untuk mengekstrak kalimat (snippet) yang paling relevan dari dokumen berdasarkan query.
    def extract_snippets(text: str, query: str, n: int = 3) -> List[str]:
        # Tokenisasi query untuk mengidentifikasi kata kunci penting.
        query_tokens = set(re.findall(r'\w+', query.lower()))
        sentences = _split_sentences(text)
        # Beri skor setiap kalimat berdasarkan jumlah kata kunci query yang ada.
        scored = [(len(set(re.findall(r'\w+', s.lower())) & query_tokens), len(s), s) for s in sentences]
        # Urutkan berdasarkan skor relevansi (menurun) dan kemudian panjang (menurun untuk tie-breaking).
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        # Kembalikan N kalimat teratas yang mengandung setidaknya satu kata kunci query.
        top = [s for score, ln, s in scored if score > 0][:n]
        return top or sentences[:n]

    # Mengurai konten JSON dari ToolMessage, dan handling error.
    def parse_tool_message_json(tm: str) -> Optional[List[Dict[str, Any]]]:
        try:
            # Muat string JSON ke dalam objek Python.
            parsed = json.loads(tm)
            # Normalisasi output agar selalu berupa list.
            return [parsed] if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else None
        except Exception:
            return None

    # Fungsi utama untuk query Qdrant dan mengambil data resume lengkap yang diformat.
    def get_relevant_resumes(query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Melakukan pencarian kemiripan vektor menggunakan wrapper LangChain.
        results_langchain = qdrant.similarity_search_with_score(query, k=k)
        # Ekstrak ID Qdrant internal dari hasil pencarian.
        qdrant_ids = [str(doc.metadata.get("_id")) for doc, _ in results_langchain if doc.metadata.get("_id")]
        
        if not qdrant_ids: return []
        
        try:
            # Gunakan Qdrant Client untuk mengambil payload dokumen lengkap berdasarkan ID.
            full_points = qclient.retrieve(collection_name=COLLECTION_NAME, ids=qdrant_ids, with_payload=True)
        except Exception as e:
            return [{"error": f"Failed to retrieve full data from Qdrant: {e}"}]

        # Buat kamus (dictionary) untuk pencarian cepat payload berdasarkan ID Qdrant.
        payload_lookup = {str(p.id): (p.payload or {}) for p in full_points}
        formatted_results = []
        # Gabungkan data dari pencarian kemiripan dan payload lengkap.
        for doc, score in results_langchain:
            qdrant_id = str(doc.metadata.get("_id"))
            native_payload = payload_lookup.get(qdrant_id, {})
            # Tentukan ID unik dan kategori kandidat.
            candidate_id = native_payload.get("ID") or native_payload.get("id") or qdrant_id
            category = native_payload.get("Category") or native_payload.get("category")
            # Ambil konten teks utama dari resume.
            text = doc.page_content or native_payload.get("text") or native_payload.get("Resume_str") or ""
            
            # Susun output akhir, termasuk skor relevansi dan snippet yang diekstrak.
            formatted_results.append({
                "qdrant_id": qdrant_id, "ID": candidate_id, "Category": category,
                "content": text, "snippets": extract_snippets(text, query, n=3),
                "score": float(score)
            })
        return formatted_results

    # Definisikan tools LangChain kustom yang dapat digunakan oleh agen untuk retrieval.
    @tool
    def retrieve_resumes_tool(query: str, k: int = 5):
        """Tool to retrieve relevant resumes. Returns JSON of candidate data."""
        results = get_relevant_resumes(query, k=k)
        # Alat mengembalikan hasil sebagai string JSON agar dapat diproses oleh LLM.
        return json.dumps(results, ensure_ascii=False, default=str)

    # Definisikan system prompt, yang akan mengatur instruksi untuk agen.
    AGENT_PROMPT = (
        "You are SmartHire, an assistant for shortlisting candidates. "
        "Use 'retrieve_resumes_tool(query,k)' to fetch resumes. 'ID' is the unique identifier. "
        "When asked to shortlist candidates, return a numbered shortlist with concise reasons for each candidate (skills match, experience, keywords), "
        "Keep responses professional and HR-friendly."
        "Strictly Answer in the same language as the user input."
    )
    # Inisialisasi agen ReAct, memberikan LLM dan alat retrieval.
    agent = create_react_agent(model=llm, tools=[retrieve_resumes_tool])

    # Fungsi untuk menjalankan agen dengan query pengguna dan memproses seluruh rantai respons.
    def invoke_agent(user_query: str) -> Dict[str, Any]:
        # Susun input dengan pesan sistem dan pengguna.
        input_messages = [{"role": "system", "content": AGENT_PROMPT}, {"role": "user", "content": user_query}]
        # Invoke agent, yang mungkin menggunakan tool calls internal.
        result = agent.invoke({"messages": input_messages})
        messages = result.get("messages", [])
        # Ekstrak jawaban akhir dari agen.
        assistant_message = messages[-1].content if messages else "(no assistant content)"
        
        parsed_tool_results = []
        # Iterasi melalui semua pesan untuk menemukan dan mengurai hasil dari tool calls.
        for m in messages:
            if isinstance(m, ToolMessage):
                parsed = parse_tool_message_json(m.content)
                if parsed: parsed_tool_results.extend(parsed)
                
        # Hitung perkiraan penggunaan token untuk estimasi biaya.
        total_input_tokens, total_output_tokens = 0, 0
        for m in messages:
            # Ambil metadata penggunaan token.
            usage = (getattr(m, "response_metadata", {}).get("token_usage") or {})
            total_input_tokens += usage.get("prompt_tokens", 0)
            total_output_tokens += usage.get("completion_tokens", 0)
            
        # Perkirakan biaya panggilan LLM (asumsi 1 USD = 17000 rupiah).
        price_idr = 17000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000
        
        # Kembalikan respons agen akhir, tools data, dan metric penggunaan.
        return {"answer": assistant_message, "parsed_tool_results": parsed_tool_results, "total_input_tokens": total_input_tokens, "total_output_tokens": total_output_tokens, "price_idr": price_idr}

    # Main title dan deskripsi aplikasi.
    st.title("SmartHire | AI Resume Assistant 📝⭐")
    st.caption("Search for Candidates, Summarize Resumes, and Shortlist using RAG + LLM")

    # Inisialisasi semua variabel status sesi Streamlit.
    if "messages" not in st.session_state: st.session_state.messages = [] # Menyimpan history chat.
    if "shortlist" not in st.session_state: st.session_state.shortlist = {} # Menyimpan kandidat yang masuk shortlist.
    if "last_response" not in st.session_state: st.session_state.last_response = None # Menyimpan data respons penuh agen terakhir.
    if "scorecards" not in st.session_state: st.session_state.scorecards = {} # Placeholder untuk data dari halaman lain.

    # Tampilkan chat messages yang tersimpan di chat window.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Menghandle input user baru dari chat box.
    if user_input := st.chat_input("Ask about candidates (example: 'Shortlist 5 Data Scientists with Python & SQL Skill')"):
        # Tambahkan pesan pengguna ke history chat dan tampilkan.
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        
        # Tampilkan indikator "Processing" saat agen loading.
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # Panggil agen dengan query pengguna.
                resp = invoke_agent(user_input)
                # Tampilkan respons teks akhir dari agen.
                st.markdown(resp["answer"])
                # Simpan respons akhir dan data raw dari agen.
                st.session_state.messages.append({"role": "assistant", "content": resp["answer"]})
                st.session_state.last_response = resp

    # Setelah query, periksa apakah respons terakhir berisi hasil tool calls (kandidat yang diambil).
    if resp := st.session_state.get("last_response"):
        # Gunakan expander untuk menampilkan output tool calls (data kandidat).
        with st.expander("Tool Calls / Retrieved Items", expanded=True):
            if results := resp.get("parsed_tool_results"):
                st.write(f"Found {len(results)} candidate items.")
                # Ulangi setiap kandidat yang diambil untuk menampilkan detail dan opsi shortlist.
                for i, c in enumerate(results, start=1):
                    # Pastikan data kandidat terstruktur dengan benar untuk ditampilkan.
                    candidate = {"qdrant_id": c.get("qdrant_id", ""),"ID": c.get("ID") or c.get("id"),"Category": c.get("Category"),"content": c.get("content") or "","snippets": c.get("snippets") or [],"score": float(c.get("score") or 0)}
                    
                    # Tampilkan informasi ringkasan kandidat.
                    st.markdown(f"--- \n ### {i}. Candidate ID: `{candidate['ID']}` — Score: {candidate['score']:.4f}")
                    st.markdown(f"**Category:** {candidate.get('Category') or '—'}")
                    
                    # Tampilkan snippet teks yang paling relevan dari resume sebagai bukti.
                    if candidate["snippets"]:
                        st.markdown("**Evidence (snippets):**")
                        for s in candidate["snippets"]: st.code(s[:800], language=None)
                        
                    # Buat dua kolom untuk notes dan shortlist.
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # Izinkan user melihat dan mengedit notes untuk kandidat.
                        existing_notes = st.session_state.shortlist.get(str(candidate["ID"]), {}).get("notes", "")
                        new_notes = st.text_area("Add notes for candidate", value=existing_notes, key=f"notes_{candidate['ID']}_{i}", height=80)
                    with col2:
                        # Tombol untuk menambahkan kandidat ke shortlist dengan notes saat ini.
                        if st.button("Add to shortlist", key=f"btn_add_{candidate['ID']}_{i}"):
                            st.session_state.shortlist[str(candidate["ID"])] = {"candidate": candidate, "notes": new_notes}
                            st.success(f"Added Candidate ID `{candidate['ID']}` to shortlist.")
                            # Jalankan ulang untuk memperbarui status tombol.
                            st.rerun()
                        # Tombol untuk menghapus kandidat dari shortlist.
                        if str(candidate["ID"]) in st.session_state.shortlist and st.button("Remove from shortlist", key=f"btn_remove_{candidate['ID']}_{i}"):
                            st.session_state.shortlist.pop(str(candidate["ID"]), None)
                            st.info(f"Removed Candidate ID `{candidate['ID']}` from shortlist.")
                            st.rerun()
            else:
                st.write("No tool calls were made for this query.")
        # Tampilkan estimasi penggunaan token dan biaya yang dihitung untuk interaksi terakhir.
        with st.expander("Usage & Price Estimate"):
            st.code(f'Input tokens (est): {resp["total_input_tokens"]}\nOutput tokens (est): {resp["total_output_tokens"]}\nEstimated price (IDR): {resp["price_idr"]:,.2f}')


# --- Main Execution Block ---
# Periksa apakah skrip dijalankan secara langsung.
if __name__ == "__main__":
    # Inisialisasi status autentikasi di status sesi jika belum diatur.
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Tampilkan aplikasi utama jika user sudah login, jika tidak tampilkan layar login.
    if st.session_state.authenticated:
        main_app()
    else:
        login_screen()
