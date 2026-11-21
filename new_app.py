import streamlit as st
import PyPDF2
import faiss
import numpy as np
import os
from PIL import Image
import io
from langchain_ollama import OllamaLLM
from langchain_community.embeddings.ollama import OllamaEmbeddings
import base64

# Initialize Ollama LLM
# Pick the model with the OLLAMA_MODEL environment variable (default: mistral; use phi3:mini on low-RAM laptops)
llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "mistral"), base_url="http://127.0.0.1:11434")
ollama_embeddings = OllamaEmbeddings(base_url="http://127.0.0.1:11434", model="nomic-embed-text")

# Page config with custom styling
st.set_page_config(
    page_title="AI-Powered PDF Q&A",
    page_icon="📚",
    layout="wide",
)

# Custom CSS with modern design and improved chat UI
st.markdown("""
<style>
    /* Base styles */
    html, body {
        background-color: #0b0f14;
        margin: 0;
        padding: 0;
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #0b0f14;
    }
    .stApp {
        max-width: 100%;
        margin: 0;   /* remove centering */
        padding-left: 40px; /* optional spacing */
        padding-right: 40px;
        background-color: #0b0f14;
    }

    /* Logo and header styling */
    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .logo-text {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-left: 10px;
    }
    .subtitle {
        color: #4b5563;
        font-size: 16px;
        margin-bottom: 25px;
        font-style: italic;
    }

    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 12px;
        padding: 15px 20px;
        border: 1px solid #e5e7eb;
        font-size: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
    }

    /* Containers */
    .result-box {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #e5e7eb;
        margin-top: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .upload-box {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 35px 25px;
        text-align: center;
        margin: 25px 0;
        background-color: #f8fafc;
        transition: all 0.3s ease;
    }
    .upload-box:hover {
        background-color: #f0f9ff;
        border-color: #93c5fd;
    }

    /* Chat UI styling */
    .chat-container {
        margin-top: 25px;
        border-radius: 12px;
        overflow: hidden;
        background-color: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .chat-message {
        padding: 16px 20px;
        margin: 4px 0;
        border-radius: 8px;
        line-height: 1.5;
        font-size: 15px;
    }
    .chat-message-user {
        background-color: #e9f2ff;
        margin-left: 30px;
        margin-right: 10px;
        border-bottom-right-radius: 2px;
        font-weight: 500;
        color: #1e3a8a;
    }
    .chat-message-ai {
        background-color: #f8fafc;
        margin-right: 30px;
        margin-left: 10px;
        border-bottom-left-radius: 2px;
        color: #333333;
    }
    .chat-header {
        padding: 15px 20px;
        background-color: #f1f5f9;
        border-bottom: 1px solid #e5e7eb;
        font-weight: 600;
        color: #334155;
    }
    .chat-input-container {
        padding: 15px;
        background-color: #f8fafc;
        border-top: 1px solid #e5e7eb;
    }
    img {
    margin-top: 40px;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    .summary-button button {
        background: linear-gradient(90deg, #047857 0%, #10b981 100%);
    }
    .facts-button button {
        background: linear-gradient(90deg, #b45309 0%, #f59e0b 100%);
    }
    .questions-button button {
        background: linear-gradient(90deg, #be185d 0%, #ec4899 100%);
    }
    .study-button button {
        background: linear-gradient(90deg, #4f46e5 0%, #818cf8 100%);
    }
    .think-button button {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    }

    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #64748b;
        font-size: 14px;
        padding: 15px;
        border-top: 1px solid #e2e8f0;
    }

    /* Radio button styling */
    .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
    .stRadio > div > div {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 10px 15px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .stRadio > div > div:hover {
        background-color: #f0f9ff;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 14px;
        font-weight: 600;
        color: #334155;
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 10px 15px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        background-color: #f1f5f9;
        color: #000000;
        font-weight: 700;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        border-color: #e2e8f0 !important;
        border-bottom: none !important;
        color: #000000 !important;
        font-weight: 700;
    }

    /* Success/Info/Error message styling */
    .stAlert {
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        # Limit PDF text size to prevent memory crash
        text = text[:20000]
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def generate_embeddings(text):
    try:
        # Use Ollama embeddings instead of OpenAI
        embedding = ollama_embeddings.embed_query(text)
        return embedding
    except Exception as e:
        # Fail silently to avoid duplicate error messages
        return None

def split_text_into_chunks(text, chunk_size=300):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def build_faiss_index(texts):
    try:
        # Fix embedding loop to avoid duplicate calls
        embeddings = []
        for chunk in texts:
            emb = generate_embeddings(chunk)
            if emb is not None:
                embeddings.append(emb)
        
        # Safeguard before FAISS index creation
        if not embeddings:
            return None, []
        
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index, embeddings
    except Exception as e:
        st.error(f"Error building FAISS index: {e}")
        return None, []

def search_faiss_index(index, query_embedding, texts, k=3):
    try:
        distances, indices = index.search(np.array([query_embedding]).astype('float32'), k=k)
        results = [texts[idx] for idx in indices[0] if idx < len(texts)]
        return results
    except Exception as e:
        st.error(f"Error searching FAISS index: {e}")
        return []

def create_llm_response(prompt, model='mistral'):
    try:
        # Use Ollama LLM instead of OpenAI
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        st.error(
            "Ollama request failed. Make sure the Ollama server is running "
            "(`ollama serve`) and reachable at http://127.0.0.1:11434.\n"
            f"Details: {e}"
        )
        return "Error: Ollama is not responding. Please start `ollama serve` and try again."



def get_ai_response(query, context=None, filename=None):
    if context:
        prompt = f"""
        Based on the following context from the document '{filename}',
        please answer the question thoroughly and accurately.

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        """
    else:
        prompt = query

    return create_llm_response(prompt)

# App state initialization
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'texts' not in st.session_state:
    st.session_state.texts = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "PDF Q&A"
if 'current_action' not in st.session_state:
    st.session_state.current_action = None
if 'current_level' not in st.session_state:
    st.session_state.current_level = "Level A"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'topic_input' not in st.session_state:
    st.session_state.topic_input = ""
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'last_pdf_name' not in st.session_state:
    st.session_state.last_pdf_name = ""
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "base"

left, right = st.columns([2, 1])
with left:
    tab1, tab2 = st.tabs(["🧠 Base Model", "📚 Trained Model"])

    with tab1:
        st.session_state.active_tab = "base"
        st.markdown("### Chat with Base Model")
        if "base_chat_history" not in st.session_state:
            st.session_state.base_chat_history = []

        # Display chat
        for chat in st.session_state.base_chat_history:
            if chat["role"] == "user":
                st.markdown(f'<div class="chat-message chat-message-user">{chat["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message chat-message-ai">{chat["content"]}</div>', unsafe_allow_html=True)

        # Input
        with st.form("base_chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask anything...")
            submit = st.form_submit_button("Send")

            if submit and user_input:
                st.session_state.base_chat_history.append({"role": "user", "content": user_input})

                with st.spinner("Thinking..."):
                    response = create_llm_response(user_input)

                st.session_state.base_chat_history.append({"role": "assistant", "content": response})
                st.rerun()


    with tab2:
        st.session_state.active_tab = "trained"
        st.markdown("### Trained Model")
        # KEEP your existing tab2 code here
        col_left, col_right = st.columns([1, 1])

        # -----------------------
        # RIGHT SIDE → PDF PREVIEW
        # -----------------------
        with col_right:
            st.markdown("#### 📄 PDF Preview")

            if st.session_state.pdf_bytes is not None:
                base64_pdf = base64.b64encode(st.session_state.pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700px" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                st.info("Upload a PDF to preview")

        # -----------------------
        # LEFT SIDE → UPLOAD + CHAT
        # -----------------------
        with col_left:

            uploaded_file = st.file_uploader("Upload PDF", type="pdf")

            if uploaded_file:
                # Load bytes immediately for preview
                if uploaded_file.name != st.session_state.last_pdf_name:
                    st.session_state.pdf_bytes = uploaded_file.read()
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.last_pdf_name = uploaded_file.name
                    st.session_state.processing_done = False
                    st.session_state.chat_history = []

            process_clicked = st.button("Process PDF")

            if process_clicked:
                if st.session_state.pdf_bytes is None:
                    st.warning("Please upload a PDF first.")
                else:
                    with st.spinner("Processing PDF..."):
                        pdf_file = io.BytesIO(st.session_state.pdf_bytes)
                        text = extract_text_from_pdf(pdf_file)
                        text = text[:20000]
                        chunks = split_text_into_chunks(text, chunk_size=300)
                        index, _ = build_faiss_index(chunks)
                        
                        st.session_state.pdf_text = text
                        st.session_state.texts = chunks
                        st.session_state.faiss_index = index
                        st.session_state.processing_done = True
                        st.session_state.chat_history = []

                    st.success("PDF processed!")

        # -----------------------
        # CHAT (ONLY AFTER PROCESS)
        # -----------------------
        if st.session_state.processing_done:

            st.markdown("### 💬 Chat with your PDF")

            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.markdown(f'<div class="chat-message chat-message-user">{chat["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message chat-message-ai">{chat["content"]}</div>', unsafe_allow_html=True)

            with st.form("trained_chat_form", clear_on_submit=True):
                user_input = st.text_input("Ask about your PDF")
                submit = st.form_submit_button("Send")

                if submit and user_input:
                    st.session_state.chat_history.append({"role": "user", "content": user_input})

                    with st.spinner("Thinking..."):
                        query_embedding = generate_embeddings(user_input)
                        
                        if query_embedding is None:
                            st.warning("Embedding failed. Please try again.")
                            st.stop()
                        
                        relevant_chunks = search_faiss_index(
                            st.session_state.faiss_index,
                            query_embedding,
                            st.session_state.texts
                        )
                        context = " ".join(relevant_chunks)
                        response = get_ai_response(user_input, context, st.session_state.file_name)

                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()


with right:
    if st.session_state.active_tab == "base":
        st.image("alpaca.png", width=320)

# ⬅️ your columns block ends here

if st.session_state.active_tab == "trained":
    st.markdown("""
        <style>
        .floating-img {
            position: fixed;
            top: 80px;
            right: 60px;
            z-index: 999;
        }
        </style>
        <div class="floating-img">
    """, unsafe_allow_html=True)

    st.image("alpaca.png", width=120)

    st.markdown("</div>", unsafe_allow_html=True)



# Footer
st.markdown('<div class="footer">© 2025 AI Scholar • Empowering education through AI</div>', unsafe_allow_html=True)
