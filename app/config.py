import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Silence Windows HuggingFace symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Resolve absolute path to .env in project root
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def get_active_model_names() -> dict:
    """Returns the active LLM and Embedding model names for API metadata."""
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    use_openai_emb = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"

    if groq_key and groq_key != "gsk_your_actual_groq_key_here":
        llm_name = os.getenv("LLM_MODEL", "openai/gpt-oss-20b").strip()
    elif openai_key:
        llm_name = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    else:
        llm_name = "unconfigured"

    if (use_openai_emb or not groq_key) and openai_key and not openai_key.startswith("gsk_"):
        emb_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    else:
        emb_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()

    return {"llm_model": llm_name, "embedding_model": emb_name}


def get_llm(temperature: float = 0.0, max_tokens: Optional[int] = None) -> ChatOpenAI:
    """
    Returns a configured LangChain ChatOpenAI instance.
    Supports Groq (via OpenAI-compatible endpoint) and standard OpenAI API keys.
    """
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    # 1. Groq OpenAI-Compatible Endpoint (Default Free Setup)
    if groq_key and groq_key != "gsk_your_actual_groq_key_here":
        model_name = os.getenv("LLM_MODEL", "openai/gpt-oss-20b").strip()
        kwargs = {
            "api_key": groq_key,
            "base_url": "https://api.groq.com/openai/v1",
            "model": model_name,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    # 2. Standard OpenAI API Fallback (if evaluator provides OPENAI_API_KEY)
    if openai_key:
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        if model_name.startswith("openai/"):
            model_name = "gpt-4o-mini"
        kwargs = {
            "api_key": openai_key,
            "model": model_name,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    raise EnvironmentError(
        "Missing API Key! Please set GROQ_API_KEY or OPENAI_API_KEY in your .env file."
    )


def get_embeddings():
    """
    Returns local HuggingFace embeddings (default 100% free via sentence-transformers)
    or OpenAIEmbeddings if OPENAI_API_KEY is active with USE_OPENAI_EMBEDDINGS=true.
    """
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    use_openai_emb = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"

    if (use_openai_emb or not groq_key) and openai_key and not openai_key.startswith("gsk_"):
        emb_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        return OpenAIEmbeddings(api_key=openai_key, model=emb_model)

    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
    return HuggingFaceEmbeddings(model_name=model_name)