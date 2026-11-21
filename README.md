# 📚 AI Scholar — Intelligent PDF Q&A System

AI Scholar is a modern, AI-powered web application built with **Streamlit** that enables users to interact with documents and large language models in an intuitive way. It allows users to upload PDFs, process them, and ask context-aware questions, or chat freely with a base AI model.

---

## 🚀 Features

### 🧠 Base Model Chat

* Chat directly with a local LLM (via Ollama)
* Continuous conversation with memory
* Fast and lightweight interaction

### 📚 Trained Model (PDF Q&A)

* Upload and process PDF documents
* Automatic text extraction and chunking
* Semantic search using embeddings + FAISS
* Context-aware Q&A based on document content
* Built-in PDF preview alongside chat

### 💬 Smart Chat Interface

* Clean, modern UI with chat bubbles
* Separate chat histories for base and trained modes
* Real-time responses with loading indicators

### ⚡ Performance Optimizations

* Text chunking for efficient retrieval
* FAISS vector indexing for fast similarity search
* Embedding caching safeguards
* PDF size limiting to prevent crashes

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit + Custom CSS
* **LLM Backend:** Ollama (Mistral model)
* **Embeddings:** Ollama (`nomic-embed-text`)
* **Vector Database:** FAISS
* **PDF Processing:** PyPDF2
* **Language:** Python

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8506
```

---

## 🧩 How It Works

### 📄 PDF Mode Workflow

1. Upload a PDF
2. Extract text using PyPDF2
3. Split into chunks
4. Generate embeddings using Ollama
5. Store vectors in FAISS
6. On query:

   * Convert query → embedding
   * Retrieve relevant chunks
   * Send context + question to LLM
   * Return intelligent answer

### 🧠 Base Model Workflow

* Direct prompt → Ollama LLM → Response

---

## 📁 Project Structure

```
ai-scholar/
│
├── app.py                 # Main Streamlit app
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── assets/               # (Optional) images or styling
```

---

## 🎯 Use Cases

* 📖 Study assistance from PDFs
* 🧑‍🏫 Training material exploration
* 🧠 Research paper summarization
* ❓ Context-aware Q&A systems
* 🏫 Educational tools

---

## ⚠️ Limitations

* PDF text extraction may fail for scanned/image PDFs
* Context limited to ~20,000 characters
* Requires local Ollama server running
* No persistent database (session-based)

---

## 🔮 Future Improvements

* Support for multiple PDFs
* Persistent vector database (e.g., Chroma, Pinecone)
* OCR support for scanned PDFs
* Streaming responses
* User authentication
* Cloud deployment

