# Automated Financial Data Analysis & RAG Chatbot Prototype

An end-to-end financial analytics, automated executive reporting, and Retrieval-Augmented Generation (RAG) Q&A system built with **Python, Pandas, Matplotlib, LangChain, ChromaDB, and FastAPI**.

---

## 🎯 Core Design Principle: Zero LLM Math

This system strictly separates deterministic computation from generative narration:

1. **100% Deterministic Analytics (`app/analyzer.py`)**: All schema validation, missing-value audits, statistical aggregations (sum, mean, median, min, max, quartiles, IQR), and $1.5 \times \text{IQR}$ outlier detection are computed exclusively in Python using **Pandas** and **NumPy**.

2. **Controlled Narrative Generation (`app/reporter.py`)**: The LLM receives *only* the pre-computed `financial_facts.json` payload at `temperature=0.0` to author an executive narrative and compile a professional multi-page PDF report (`final_report.pdf`) with embedded **Matplotlib** visualizations.

3. **Strictly Grounded RAG (`app/rag.py`)**: The generated report is chunked via LangChain's `RecursiveCharacterTextSplitter`, embedded into a persistent local **ChromaDB** vector store, and queried through a retrieval chain that returns grounded answers with source citations while politely declining unsupported or math-manipulation queries.

---

## 📂 Project Directory Structure

```text
financial-rag-prototype/
├── app/
│   ├── __init__.py
│   ├── analyzer.py        # CSV validation, data profiling, IQR outlier detection & Matplotlib charts
│   ├── config.py          # LLM & Embedding configuration (Groq OpenAI-compatible & OpenAI support)
│   ├── reporter.py        # Grounded LLM executive narrative & PDF report compiler (FPDF2)
│   ├── rag.py             # LangChain chunking, ChromaDB vector indexing & grounded Q&A engine
│   └── main.py            # FastAPI service exposing /analyze, /chat, /health, and embedded Chat UI
├── data/
│   └── fintech_transactions_training_dataset.csv
├── outputs/
│   └── runs/              # Timestamped production run artifacts (JSON, 4 PNG charts, MD & PDF reports)
│   └── test_run/          # Timestamped production test run artifacts (JSON, 4 PNG charts, MD & PDF reports)
├── tests/
│   └── test_analyzer.py   # Automated unit and API validation tests (pytest)
├── .env                   # Local environment variables (API keys)
├── .env.example           # Template for environment configuration
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation
```

---

## 🧰 Tools & Software Required

* **Python 3.10+ / 3.12** (Primary development and execution runtime)
* **pip** (Python package installer)
* **Git** (For version control)
* **API Key Provider**: 
  * **Groq API Key** (`GROQ_API_KEY`) using `openai/gpt-oss-20b` (Default free setup) — *OR* —
  * **OpenAI API Key** (`OPENAI_API_KEY`) using `gpt-4o-mini`.

---

## ⚙️ Installation & Setup Instructions

### 1. Clone or Open the Project
Open your terminal in the project root directory (`financial-rag-prototype`).

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
Install all required Python libraries specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (`.env`)
Create a `.env` file in the project root directory by copying `.env.example`:
```bash
cp .env.example .env
```
Open `.env` and paste your API credentials. The configuration supports both Groq and OpenAI:

```ini
# Default Setup (Groq OpenAI-Compatible + Free Local HuggingFace Embeddings)
GROQ_API_KEY=gsk_your_actual_groq_key_here
LLM_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional OpenAI Configuration (Uncomment if using OpenAI instead of Groq)
# OPENAI_API_KEY=sk-your_openai_api_key_here
# LLM_MODEL=gpt-4o-mini
# USE_OPENAI_EMBEDDINGS=true
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## 🚀 Running the Application

### 1. Start the FastAPI Server
Run the FastAPI application locally using Uvicorn with auto-reload:
```bash
uvicorn app.main:app --reload
```

### 2. Access the User Interfaces
Once the server is running (`Application startup complete`), you can access two interfaces:
* **Interactive Web Chat UI**: Open your browser and navigate to **`http://127.0.0.1:8000/`** for a clean chat interface with message bubbles, clickable presets, and an instant **"Re-Run Analysis Pipeline"** button.
* **Developer Swagger UI (`/docs`)**: Navigate to **`http://127.0.0.1:8000/docs`** to test the API endpoints (`POST /analyze`, `POST /chat`, and `GET /health`) interactively.

---

## 🧪 Running Unit Tests (`pytest`)

To execute the automated test suite verifying schema validation, data profiling, metric calculations, outlier detection, and API health:

```bash
python -m pytest -v
```
*(Tests route their temporary artifacts cleanly into `outputs/test_run/` without cluttering production runs).*