"""Shared configuration for Lab 18."""

import os
import threading
import time
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
_api_key = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_KEY = _api_key if _api_key.startswith("sk-") and len(_api_key) > 20 else ""
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else "deepseek" if DEEPSEEK_API_KEY else "openai")
LLM_MODEL = os.getenv("LLM_MODEL", {"gemini": "gemini-3.1-flash-lite", "deepseek": "deepseek-chat", "openai": "gpt-4o-mini"}[LLM_PROVIDER])
RAGAS_MODEL = os.getenv("RAGAS_MODEL", LLM_MODEL)
_gemini_lock = threading.Lock()
_gemini_next_request = 0.0
_gemini_clients = None


def _wait_for_gemini():
    global _gemini_next_request
    with _gemini_lock:
        time.sleep(max(0.0, _gemini_next_request - time.monotonic()))
        _gemini_next_request = time.monotonic() + 6.5


def gemini_http_clients():
    global _gemini_clients
    if _gemini_clients is None:
        import asyncio
        import httpx

        async def wait_for_gemini(request):
            await asyncio.to_thread(_wait_for_gemini)

        _gemini_clients = (
            httpx.Client(event_hooks={"request": [lambda request: _wait_for_gemini()]}),
            httpx.AsyncClient(event_hooks={"request": [wait_for_gemini]}),
        )
    return _gemini_clients


def get_llm_client():
    from openai import OpenAI

    if LLM_PROVIDER == "deepseek" and DEEPSEEK_API_KEY:
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", timeout=45, max_retries=1)
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        return OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=45,
            max_retries=2,
            http_client=gemini_http_clients()[0],
        )
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY, timeout=45, max_retries=1)
    return None

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
