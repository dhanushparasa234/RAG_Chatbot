from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from app.analyzer import analyze_financial_data
from app.reporter import generate_executive_report
from app.rag import FinancialReportRAG

app = FastAPI(
    title="Automated Financial Data Analysis & RAG Chatbot API",
    description=(
        "End-to-end service for deterministic financial CSV analysis, "
        "executive report generation (Markdown & PDF), and strictly grounded RAG Q&A."
    ),
    version="1.0.0",
)

# In-memory cache of the active RAG engine for fast query responses
active_rag_engine: Optional[FinancialReportRAG] = None


# --- Response Schemas ---

class AnalyzeResponse(BaseModel):
    status: str
    run_timestamp: str
    run_directory: str
    markdown_report_path: str
    pdf_report_path: str
    indexed_chunks_count: int
    financial_facts: Dict[str, Any]


class SourceExcerpt(BaseModel):
    chunk_id: Optional[int] = None
    run_folder: Optional[str] = None
    source: Optional[str] = None
    excerpt: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    question: str
    answer: str
    run_folder: str
    report_used: str
    model_used: str
    embedding_model_used: str
    sources: List[SourceExcerpt]


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def chat_ui():
    """Serves a clean, self-contained web chat interface for interacting with the RAG chatbot."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Financial Intelligence RAG Chatbot</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans">
        <!-- Header -->
        <header class="bg-slate-900 border-b border-slate-800 px-6 py-4 flex justify-between items-center shadow-md">
            <div class="flex items-center space-x-3">
                <div class="bg-emerald-600 p-2 rounded-lg font-bold text-white">📊</div>
                <div>
                    <h1 class="text-lg font-bold text-white">Financial Intelligence RAG Chatbot</h1>
                    <p class="text-xs text-slate-400">Strictly grounded Q&A over verified audit reports</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <button onclick="triggerAnalyze()" id="analyze-btn" class="bg-slate-800 hover:bg-slate-700 text-xs px-3 py-2 rounded-md font-medium border border-slate-700 transition flex items-center space-x-1">
                    <span>🔄 Re-Run Analysis Pipeline</span>
                </button>
                <a href="/docs" target="_blank" class="bg-indigo-600 hover:bg-indigo-500 text-xs px-3 py-2 rounded-md font-medium transition">
                    API Docs (Swagger) ↗
                </a>
            </div>
        </header>

        <!-- Main Chat Container -->
        <main class="flex-1 max-w-4xl w-full mx-auto p-4 flex flex-col space-y-4 overflow-hidden">
            <!-- Chat Feed -->
            <div id="chat-feed" class="flex-1 bg-slate-900/50 border border-slate-800 rounded-xl p-4 overflow-y-auto space-y-4 shadow-inner min-h-[60vh] max-h-[70vh]">
                <!-- Welcome Message -->
                <div class="flex items-start space-x-3">
                    <div class="bg-indigo-600 text-white rounded-full h-8 w-8 flex items-center justify-center font-bold shrink-0">AI</div>
                    <div class="bg-slate-800 border border-slate-700 rounded-2xl px-4 py-3 max-w-[85%] text-sm leading-relaxed shadow">
                        <p class="font-semibold text-indigo-300 mb-1">Financial RAG Assistant</p>
                        <p>Hello! I am ready to answer questions based on your latest financial audit report. Ask me about core metrics, data quality, outliers, or trends.</p>
                        <div class="mt-3 flex flex-wrap gap-2">
                            <button onclick="askPreset(this)" class="text-xs bg-slate-700 hover:bg-slate-600 px-2.5 py-1 rounded-full text-slate-200 transition">What are the total, mean, and median transaction amounts?</button>
                            <button onclick="askPreset(this)" class="text-xs bg-slate-700 hover:bg-slate-600 px-2.5 py-1 rounded-full text-slate-200 transition">How many outliers were flagged using the IQR rule?</button>
                            <button onclick="askPreset(this)" class="text-xs bg-slate-700 hover:bg-slate-600 px-2.5 py-1 rounded-full text-slate-200 transition">What is the date range and total missing cell count?</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Input Form -->
            <form id="chat-form" onsubmit="handleSubmit(event)" class="flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-xl p-2 shadow-lg">
                <input 
                    type="text" 
                    id="user-input" 
                    placeholder="Ask a question about the financial report..." 
                    class="flex-1 bg-transparent px-4 py-2 text-sm focus:outline-none text-slate-100 placeholder-slate-500"
                    autocomplete="off"
                    required
                >
                <button 
                    type="submit" 
                    id="send-btn"
                    class="bg-indigo-600 hover:bg-indigo-500 text-white px-5 py-2 rounded-lg text-sm font-medium transition flex items-center space-x-1"
                >
                    <span>Send</span>
                    <span>→</span>
                </button>
            </form>
        </main>

        <!-- Footer status -->
        <footer class="text-center py-3 text-xs text-slate-500 border-t border-slate-900">
            Automated Financial RAG Prototype | Zero LLM Math & Strict Grounding
        </footer>

        <!-- JavaScript Handler -->
        <script>
            const feed = document.getElementById('chat-feed');
            const input = document.getElementById('user-input');
            const form = document.getElementById('chat-form');
            const sendBtn = document.getElementById('send-btn');
            const analyzeBtn = document.getElementById('analyze-btn');

            function askPreset(btn) {
                input.value = btn.innerText;
                form.requestSubmit();
            }

            async function triggerAnalyze() {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = "<span>⏳ Running Pipeline...</span>";
                appendSystemMessage("Running end-to-end analysis pipeline (Pandas calculations, report generation, and ChromaDB indexing)...");
                
                try {
                    const res = await fetch('/analyze', { method: 'POST' });
                    const data = await res.json();
                    if (res.ok) {
                        appendSystemMessage(`Pipeline successful! Indexed ${data.indexed_chunks_count} chunks from run: ${data.run_timestamp}`);
                    } else {
                        appendSystemMessage(`Pipeline error: ${data.detail || 'Unknown error'}`);
                    }
                } catch (err) {
                    appendSystemMessage(`Network error running pipeline: ${err.message}`);
                } finally {
                    analyzeBtn.disabled = false;
                    analyzeBtn.innerHTML = "<span>🔄 Re-Run Analysis Pipeline</span>";
                }
            }

            function appendSystemMessage(text) {
                const div = document.createElement('div');
                div.className = "text-center my-2 text-xs text-amber-400 bg-amber-950/30 border border-amber-900/50 py-1.5 px-3 rounded-lg mx-auto max-w-md";
                div.innerText = text;
                feed.appendChild(div);
                feed.scrollTop = feed.scrollHeight;
            }

            async function handleSubmit(e) {
                e.preventDefault();
                const question = input.value.trim();
                if (!question) return;

                appendMessage(question, 'user');
                input.value = '';
                input.disabled = true;
                sendBtn.disabled = true;

                const loadingId = 'loading-' + Date.now();
                appendLoadingMessage(loadingId);

                try {
                    const res = await fetch(`/chat?question=${encodeURIComponent(question)}`, {
                        method: 'POST'
                    });
                    const data = await res.json();
                    
                    const loader = document.getElementById(loadingId);
                    if (loader) loader.remove();

                    if (res.ok) {
                        appendMessage(data.answer, 'ai', data.sources, data.run_folder);
                    } else {
                        appendMessage(`Error: ${data.detail || 'Failed to fetch answer.'}`, 'ai');
                    }
                } catch (err) {
                    const loader = document.getElementById(loadingId);
                    if (loader) loader.remove();
                    appendMessage(`Network error: ${err.message}`, 'ai');
                } finally {
                    input.disabled = false;
                    sendBtn.disabled = false;
                    input.focus();
                }
            }

            function formatText(text) {
                // Safe native formatter for bolding and line breaks without external CDNs
                return text
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/\\*\\*(.*?)\\*\\*/g, '<strong class="text-indigo-200 font-semibold">$1</strong>')
                    .replace(/\\n/g, '<br>');
            }

            function appendMessage(text, sender, sources = [], runFolder = '') {
                const isUser = sender === 'user';
                const wrapper = document.createElement('div');
                wrapper.className = `flex items-start space-x-3 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`;

                const avatar = document.createElement('div');
                avatar.className = `rounded-full h-8 w-8 flex items-center justify-center font-bold shrink-0 text-xs ${isUser ? 'bg-emerald-600 text-white' : 'bg-indigo-600 text-white'}`;
                avatar.innerText = isUser ? 'YOU' : 'AI';

                const bubble = document.createElement('div');
                bubble.className = `rounded-2xl px-4 py-3 max-w-[85%] text-sm leading-relaxed shadow ${isUser ? 'bg-emerald-950/60 border border-emerald-800 text-emerald-100' : 'bg-slate-800 border border-slate-700 text-slate-100'}`;

                if (!isUser) {
                    bubble.innerHTML = `<div class="font-semibold text-indigo-300 mb-1">Financial RAG Assistant</div><div class="space-y-2">${formatText(text)}</div>`;
                    
                    if (sources && sources.length > 0) {
                        const sourcesDiv = document.createElement('div');
                        sourcesDiv.className = "mt-3 pt-2 border-t border-slate-700/60 text-xs text-slate-400";
                        const chunkIds = [...new Set(sources.map(s => s.chunk_id))].join(', ');
                        sourcesDiv.innerHTML = `<span class="font-medium text-slate-300">Sources Cited:</span> Chunks [${chunkIds}] <span class="text-slate-500 text-[10px] ml-1">(${runFolder})</span>`;
                        bubble.appendChild(sourcesDiv);
                    }
                } else {
                    bubble.innerText = text;
                }

                wrapper.appendChild(avatar);
                wrapper.appendChild(bubble);
                feed.appendChild(wrapper);
                feed.scrollTop = feed.scrollHeight;
            }

            function appendLoadingMessage(id) {
                const wrapper = document.createElement('div');
                wrapper.id = id;
                wrapper.className = "flex items-start space-x-3";
                wrapper.innerHTML = `
                    <div class="bg-indigo-600 text-white rounded-full h-8 w-8 flex items-center justify-center font-bold shrink-0 text-xs">AI</div>
                    <div class="bg-slate-800 border border-slate-700 rounded-2xl px-4 py-3 text-sm text-slate-400 flex items-center space-x-2">
                        <svg class="animate-spin h-4 w-4 text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                            <path class="opacity-75" fill="currentColor" d="M4 12a8 10 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Analyzing report chunks...</span>
                    </div>
                `;
                feed.appendChild(wrapper);
                feed.scrollTop = feed.scrollHeight;
            }
        </script>
    </body>
    </html>
    """


@app.get("/health", tags=["System"])
def health_check():
    """Returns API health status and the latest active report directory."""
    try:
        rag = active_rag_engine or FinancialReportRAG()
        return {
            "status": "healthy",
            "latest_run_folder": rag.run_folder.name,
            "latest_report_path": str(rag.report_path).replace("\\", "/"),
        }
    except FileNotFoundError:
        return {
            "status": "healthy",
            "latest_run_folder": None,
            "message": "No report generated yet. Run POST /analyze first.",
        }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis Pipeline"])
def run_analysis_pipeline(
    csv_path: str = Query(
        default="data/fintech_transactions_training_dataset.csv",
        description="Path to the financial CSV dataset to analyze.",
    ),
):
    """
    **End-to-End Financial Analysis & Report Generation Pipeline**:
    1. Validates the CSV schema and computes deterministic financial statistics, data quality metrics, IQR outliers, and PNG visualizations.
    2. Generates a comprehensive narrative report (`final_report.md` and `final_report.pdf`) strictly from the verified JSON metrics.
    3. Chunks, embeds, and indexes the compiled report into ChromaDB for immediate Q&A retrieval.
    """
    global active_rag_engine

    try:
        facts = analyze_financial_data(
            csv_path=csv_path.strip(),
            output_dir="outputs",
        )
        run_dir = facts["output_directory"]

        report_paths = generate_executive_report(run_dir=run_dir)
        md_report_path = report_paths["markdown_report_path"]
        pdf_report_path = report_paths["pdf_report_path"]

        active_rag_engine = FinancialReportRAG(report_path=md_report_path)
        chunks_count = active_rag_engine.ingest_report(force_rebuild=True)

        return AnalyzeResponse(
            status="success",
            run_timestamp=facts["run_timestamp"],
            run_directory=run_dir.replace("\\", "/"),
            markdown_report_path=md_report_path,
            pdf_report_path=pdf_report_path,
            indexed_chunks_count=chunks_count,
            financial_facts=facts,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution error: {str(e)}",
        )


@app.post("/chat", response_model=ChatResponse, tags=["RAG Chatbot"])
def chat_with_report(
    question: str = Query(
        default="What are the total, mean, and median transaction amounts reported?",
        description="Type your question directly here in plain English.",
    ),
):
    """
    **Grounded Financial Report Chatbot**:
    Type any question into the `question` text box and click Execute.
    Automatically retrieves the most recent timestamped report from `outputs/runs/`
    and returns a strictly grounded answer with source citations.
    """
    global active_rag_engine

    if not question or not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide a non-empty question.",
        )

    try:
        latest_rag = FinancialReportRAG()
        if (
            active_rag_engine is None
            or active_rag_engine.report_path != latest_rag.report_path
        ):
            latest_rag.ingest_report(force_rebuild=True)
            active_rag_engine = latest_rag
        rag = active_rag_engine

        result = rag.ask(question=question.strip(), k=8)
        return ChatResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"{str(e)} Please run POST /analyze first.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG chat error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)