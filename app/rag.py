import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_llm, get_embeddings, get_active_model_names

OUTPUTS_BASE_DIR = Path("outputs")
DEFAULT_CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "financial_report_rag"


def find_latest_report(base_dir: Path = OUTPUTS_BASE_DIR) -> Path:
    """
    Scans `outputs/` (preferring `outputs/runs/` first, then `outputs/test_runs/`)
    to locate the most recently generated report file.
    """
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Outputs directory '{base_dir}' does not exist. "
            "Run the analysis pipeline first."
        )

    runs_dir = base_dir / "runs"
    search_root = runs_dir if runs_dir.exists() else base_dir

    text_candidates: List[Path] = []
    for pattern in ("*report*.md", "*report*.txt", "*.md", "*.txt"):
        found = [
            p for p in search_root.rglob(pattern)
            if p.is_file() and p.name.lower() != "readme.md"
        ]
        if found:
            text_candidates.extend(found)
            break

    if text_candidates:
        text_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return text_candidates[0]

    pdf_candidates = [p for p in search_root.rglob("*.pdf") if p.is_file()]
    if pdf_candidates:
        pdf_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return pdf_candidates[0]

    raise FileNotFoundError(
        f"No generated report (.md, .txt, or .pdf) found inside '{search_root}'. "
        "Please run the analysis pipeline first."
    )


def read_report_content(report_path: Path) -> str:
    """Reads text from a .md/.txt report, or extracts text if the report is a .pdf."""
    if report_path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(report_path))
            return "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            ).strip()
        except ImportError:
            raise ImportError(
                f"Found PDF report at '{report_path}', but 'pypdf' is not installed. "
                "Run: pip install pypdf"
            )
    return report_path.read_text(encoding="utf-8").strip()


def build_json_appendix(run_folder: Path) -> str:
    """
    Reads the deterministic JSON facts file in the same run folder (if present)
    and creates a verified appendix section covering chart artifacts and exact outlier counts.
    """
    json_candidates = list(run_folder.glob("*.json"))
    if not json_candidates:
        return ""

    try:
        data = json.loads(json_candidates[0].read_text(encoding="utf-8"))
    except Exception:
        return ""

    png_files = sorted([p.name for p in run_folder.glob("*.png")])

    appendix_lines = [
        "\n\n## Verified Analytical Appendix (Deterministic JSON Facts & Chart Artifacts)",
        f"- **Run Folder**: `{run_folder.name}`",
        f"- **Generated Chart Visualizations (PNG Artifacts)**: {', '.join(png_files) if png_files else 'None found'}",
        "  - `amount_distribution.png` / Amount Distribution Chart: Displays the distribution and spread of transaction amounts.",
        "  - `volume_over_time.png` / Volume Over Time Chart: Shows transaction volume and activity trends across the date coverage period.",
        "  - `missing_values.png` / Missing Values Chart: Visualizes missing cell counts across dataset columns.",
        "  - `category_channel_breakdown.png` / Category & Channel Breakdown Chart: Breakdown of transactions by business category and channel.",
        f"- **Verified JSON Metrics Snapshot**:\n```json\n{json.dumps(data, indent=2)}\n```",
    ]
    return "\n".join(appendix_lines)


class FinancialReportRAG:
    """
    Retrieval-Augmented Generation (RAG) Engine for Financial Intelligence Reports.
    Automatically resolves the latest timestamped run report, chunks and embeds it
    into ChromaDB, and answers questions with strict grounding.
    """

    def __init__(
        self,
        report_path: Optional[str | Path] = None,
        persist_directory: str | Path = DEFAULT_CHROMA_DIR,
    ):
        if report_path is not None:
            candidate = Path(report_path)
            if candidate.is_dir():
                self.report_path = find_latest_report(candidate)
            else:
                self.report_path = candidate
        else:
            self.report_path = find_latest_report(OUTPUTS_BASE_DIR)

        self.run_folder = self.report_path.parent
        self.persist_directory = Path(persist_directory)
        self.embeddings = get_embeddings()
        self.vectorstore: Chroma | None = None

    def ingest_report(self, force_rebuild: bool = True) -> int:
        """
        Loads the report from the resolved timestamped run folder, appends deterministic
        chart/JSON metadata from the same run folder for complete recall, splits into chunks,
        and indexes them into ChromaDB.
        """
        if not self.report_path.exists():
            raise FileNotFoundError(f"Report not found at '{self.report_path}'.")

        report_text = read_report_content(self.report_path)
        if not report_text:
            raise ValueError(f"Report file at '{self.report_path}' is empty.")

        appendix = build_json_appendix(self.run_folder)
        full_text_to_index = report_text + appendix

        if force_rebuild and self.persist_directory.exists():
            self.vectorstore = None
            shutil.rmtree(self.persist_directory, ignore_errors=True)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=250,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(full_text_to_index)

        documents: List[Document] = [
            Document(
                page_content=chunk.strip(),
                metadata={
                    "source": self.report_path.name,
                    "run_folder": self.run_folder.name,
                    "report_path": str(self.report_path),
                    "chunk_id": idx,
                },
            )
            for idx, chunk in enumerate(chunks)
            if chunk.strip()
        ]

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory),
            collection_name=COLLECTION_NAME,
        )

        return len(documents)

    def load_vectorstore(self) -> Chroma:
        """Loads existing ChromaDB vector store or ingests the latest report if not yet indexed."""
        if self.vectorstore is not None:
            return self.vectorstore

        if not self.persist_directory.exists():
            self.ingest_report(force_rebuild=True)
            return self.vectorstore  # type: ignore

        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
        )
        return self.vectorstore

    def ask(self, question: str, k: int = 6) -> Dict[str, Any]:
        """
        Queries ChromaDB for the top-k relevant chunks and returns a strictly grounded answer.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        vectorstore = self.load_vectorstore()

        collection_count = vectorstore._collection.count()
        effective_k = min(k, max(1, collection_count))

        retriever = vectorstore.as_retriever(search_kwargs={"k": effective_k})
        retrieved_docs = retriever.invoke(question)

        model_info = get_active_model_names()

        if not retrieved_docs:
            return {
                "question": question,
                "answer": (
                    "I cannot answer this question based on the generated financial report, "
                    "as no relevant information was found."
                ),
                "run_folder": self.run_folder.name,
                "report_used": str(self.report_path),
                "model_used": model_info["llm_model"],
                "embedding_model_used": model_info["embedding_model"],
                "sources": [],
            }

        retrieved_docs_sorted = sorted(
            retrieved_docs, key=lambda d: d.metadata.get("chunk_id", 0)
        )

        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc in retrieved_docs_sorted]
        )

        system_prompt = (
            "You are a strictly grounded Financial Report QA Assistant.\n"
            "Follow these mandatory rules:\n"
            "1. GROUNDING ONLY: Answer the user's question using ONLY the facts, metrics, and observations stated in the Report Context below.\n"
            "2. NO MATH: Never perform new calculations, sums, averages, tax additions, or future growth extrapolations yourself. Report ONLY the exact pre-calculated numbers written in the context.\n"
            "3. CLEAN OUTPUT: Write a clear, concise answer without including any chunk tags, bracketed citations, or internal metadata labels.\n"
            "4. HELPFUL ON IN-SCOPE QUESTIONS: If the context contains relevant financial metrics, min/max values, data quality stats, chart descriptions, or outlier findings that address the question, state those exact facts clearly.\n"
            "5. REFUSE UNSUPPORTED OR MATH-MANIPULATION QUESTIONS: If the question asks you to compute a new formula (e.g., adding GST tax, dividing totals, or projecting future years like 2028/2029) OR asks about external info not in the report (e.g., CFO name, headquarters, stock ticker), reply with:\n"
            "'I cannot answer this question based on the generated financial report, as that information is not present in the report.'"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "Report Context:\n{context}\n\nUser Question: {question}\n\nProvide a concise, grounded answer:",
                ),
            ]
        )

        llm = get_llm(temperature=0.0, max_tokens=1024)
        chain = prompt | llm

        answer_text = ""
        for _ in range(3):
            response = chain.invoke({"context": context_text, "question": question})
            content = (response.content or "").strip()
            if content:
                answer_text = content
                break
            time.sleep(0.5)

        if not answer_text:
            answer_text = (
                "I cannot answer this question based on the generated financial report, "
                "as no valid response could be generated from the report context."
            )

        return {
            "question": question,
            "answer": answer_text,
            "run_folder": self.run_folder.name,
            "report_used": str(self.report_path),
            "model_used": model_info["llm_model"],
            "embedding_model_used": model_info["embedding_model"],
            "sources": [
                {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "run_folder": doc.metadata.get("run_folder"),
                    "source": doc.metadata.get("source"),
                    "excerpt": doc.page_content[:200]
                    + ("..." if len(doc.page_content) > 200 else ""),
                }
                for doc in retrieved_docs_sorted
            ],
        }


if __name__ == "__main__":
    models = get_active_model_names()
    print(f"Loaded config -> LLM: {models['llm_model']} | Embeddings: {models['embedding_model']}")
    rag = FinancialReportRAG()

    print(f"\n[Step 1] Auto-detected latest run folder: '{rag.run_folder}'")
    print(f"         Indexing report '{rag.report_path}' into ChromaDB...")
    count = rag.ingest_report(force_rebuild=True)
    print(f"         Indexed {count} chunks successfully.")