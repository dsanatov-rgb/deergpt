import logging
import os
from dotenv import load_dotenv
load_dotenv()
import re
import time
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

# Логирование с управлением уровнем через переменную окружения
logging.basicConfig(
    level=os.getenv("RAG_LOG_LEVEL", "WARNING"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- Конфигурация Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL")  # например, http://localhost:6333
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # если включена аутентификация
COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
# Если коллекция использует named vectors, укажите имя (например, "text")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME")  # None или имя вектора

# Один клиент на процесс: предпочитаем URL, иначе host/port
if QDRANT_URL:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)  # [web:64]
else:
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)  # [web:64]

# --- Эмбеддинги и LLM ---
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.proxyapi.ru/openai/v1"),
    model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
)  # [attached_file:1]

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.proxyapi.ru/openai/v1"),
    model=os.getenv("CHAT_MODEL", "gpt-5.4-nano"),
    temperature=float(os.getenv("CHAT_TEMPERATURE", "0.2")),
)  # [attached_file:1]

def _clean_context(s: str) -> str:
    s = re.sub(r"(Ключевые слова|Keywords).*$", "", s, flags=re.IGNORECASE | re.S)  # [attached_file:1]
    s = re.sub(r"References?.*$", "", s, flags=re.IGNORECASE | re.S)  # [attached_file:1]
    return s.strip()  # [attached_file:1]

def _pluck_text(payload: Optional[Dict[str, Any]]) -> str:
    if not payload or not isinstance(payload, dict):
        return ""  # [attached_file:1]
    for k in ("text", "chunk", "content", "page_text", "page_content", "body"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v  # [attached_file:1]
    meta = payload.get("metadata") or payload.get("meta") or {}
    if isinstance(meta, dict):
        for k in ("text", "chunk", "content", "page_text", "page_content", "body"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v  # [attached_file:1]
    parts: List[str] = []
    for k, v in payload.items():
        if isinstance(v, str) and v.strip():
            parts.append(v)  # [attached_file:1]
    return "\n".join(parts)  # [attached_file:1]

def _build_filter(fname_filter: Optional[str]) -> Optional[Dict[str, Any]]:
    if not fname_filter:
        return None  # [attached_file:1]
    return {"must": [{"key": "fname", "match": {"value": fname_filter}}]}  # [attached_file:1]

def _qdrant_query_with_retry(
    embed_vec: List[float],
    top_k: int,
    fname_filter: Optional[str],
    retries: int = 5,
    delay: float = 0.3,
):
    """
    Совместимо с актуальным qdrant-client:
    - query_points ожидает query=list[float] и опционально using для named vectors.
    - Без dict-обёрток и без query_vector.
    """  # [web:93][web:91]
    query_filter = _build_filter(fname_filter)  # [attached_file:1]
    last_exc: Optional[Exception] = None  # [attached_file:1]
    for attempt in range(1, retries + 1):
        try:
            kwargs = {
                "collection_name": COLLECTION,
                "query": embed_vec,
                "limit": top_k,
                "query_filter": query_filter,
            }  # [web:93][web:91]
            if QDRANT_VECTOR_NAME:
                kwargs["using"] = QDRANT_VECTOR_NAME  # [web:100]
            return qdrant.query_points(**kwargs)  # [web:93]
        except Exception as e:
            last_exc = e  # [attached_file:1]
            logger.warning("[rag] Qdrant query_points failed (attempt %d/%d): %s", attempt, retries, e)  # [attached_file:1]
            time.sleep(delay * attempt)  # [attached_file:1]
    raise last_exc if last_exc else RuntimeError("Qdrant query failed without exception")  # [attached_file:1]

def get_contexts(question: str, top_k: int = 5, fname_filter: Optional[str] = None) -> List[str]:
    embed_vec = embeddings.embed_query(question)  # [attached_file:1]
    logger.debug("[rag] got embedding; dim=%d", len(embed_vec))  # [attached_file:1]

    res = _qdrant_query_with_retry(embed_vec=embed_vec, top_k=top_k, fname_filter=fname_filter)  # [attached_file:1]

    points = getattr(res, "points", None)  # [attached_file:1]
    if points is None:
        points = getattr(res, "hits", None)  # [attached_file:1]
    if points is None and isinstance(res, list):
        points = res  # [attached_file:1]
    points = points or []  # [attached_file:1]

    contexts: List[str] = []  # [attached_file:1]
    for p in points:
        payload = getattr(p, "payload", None)  # [attached_file:1]
        if payload is None and isinstance(p, dict):
            payload = p.get("payload")  # [attached_file:1]
        raw_text = _pluck_text(payload)  # [attached_file:1]
        if not raw_text:
            continue  # [attached_file:1]
        contexts.append(_clean_context(raw_text))  # [attached_file:1]

    logger.debug("[rag] contexts_found=%d", len(contexts))  # [attached_file:1]
    return contexts  # [attached_file:1]

def answer_with_rag(question: str, fname_filter: Optional[str] = None) -> str:
    contexts = get_contexts(question, top_k=3, fname_filter=fname_filter)  # [attached_file:1]
    context_str = "\n".join(contexts)  # [attached_file:1]

    logger.debug("[rag] context_len=%d", len(contexts))  # [attached_file:1]

    system_instr = (
        "Ты — помощник, отвечающий строго по предоставленному контексту. "
        "Если в контексте нет ответа, так и скажи, не выдумывай."
    )  # [attached_file:1]
    prompt = (
        f"{system_instr}\n\n"
        f"Контекст:\n{context_str}\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Ответ:"
    )  # [attached_file:1]

    try:
        resp = llm.invoke(prompt)  # [attached_file:1]
    except Exception as e:
        logger.error("[rag] LLM error: %s", e, exc_info=1)  # [attached_file:1]
        return "Упс! Что-то пошло не так! Попробуй еще!"  # [attached_file:1]

    text = getattr(resp, "content", None)  # [attached_file:1]
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))  # [attached_file:1]
    if not isinstance(text, str):
        text = getattr(resp, "text", "") or str(resp)  # [attached_file:1]
    text = (text or "").strip()  # [attached_file:1]

    logger.debug("[rag] answer_len_chars=%d", len(text))  # [attached_file:1]
    return text if text else "Упс! Что-то пошло не так! Попробуй еще!"  # [attached_file:1]

