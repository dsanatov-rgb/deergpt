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
    max_tokens=int(os.getenv("CHAT_MAX_TOKENS", "140")),
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
        "Если в контексте нет ответа, так и скажи, не выдумывай.\n"
        "Твой ответ будет озвучен вслух синтезатором речи, поэтому пиши так, "
        "как говорят люди:\n"
        "- никакой разметки: без звёздочек, решёток, дефисов-списков и заголовков;\n"
        "- без ссылок на источники, номеров в скобках, фамилий авторов и инициалов;\n"
        "- без сокращений: пиши «нашей эры», «то есть», «года» полностью;\n"
        "- века и числа записывай словами в нужном падеже: "
        "«в седьмом веке до нашей эры», «двадцать лет»;\n"
        "- не цитируй источники дословно, передавай смысл своими словами;\n"
        "- одно-два коротких предложения, не более 250 символов всего;\n"
        "- отвечай как экскурсовод: живо, конкретно, одна главная мысль;\n"
        "- сразу по сути, без вводных вроде «в тексте говорится» и «также упоминается»."
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


# --- Режим «интересный факт» ---
import random
from qdrant_client import models as qmodels

fact_log = logging.getLogger("uvicorn.error")

_FACT_RE = re.compile(r"интересн|любопытн|удивительн|удиви|необычн|факт", re.IGNORECASE)
_FACT_STOP = {
    "расскажи", "расскажите", "скажи", "скажите", "поведай", "давай", "дай", "назови",
    "мне", "нам", "пожалуйста", "что", "что-нибудь", "что-то", "нибудь", "какой",
    "какой-нибудь", "какой-то", "какую", "какое", "какие", "еще", "ещё", "один", "одну",
    "одно", "самый", "самое", "самую", "про", "обо", "это", "вот", "можешь", "можно",
    "хочу", "узнать", "меня", "нас", "бывает", "было", "есть", "знаешь", "новенькое", "новое", "олень",
}
_FACT_SEEDS = ["золото", "курган", "олень", "конь", "оружие", "пантера", "грифон",
               "гривна", "погребение", "царь", "птица", "звериный стиль"]


_FACT_STRONG = re.compile(
    r"\bфакт(?!ическ)|\bудиви(?:ть|те)?\b|что\s+(?:есть\s+|бывает\s+)?(?:интересн|любопытн|необычн|удивительн)",
    re.IGNORECASE,
)
_FACT_ASK = re.compile(r"расскаж|скаж|поведа|давай|нибудь|что-то|\bещ[её]\b", re.IGNORECASE)


def is_fact_request(question: str) -> bool:
    q = question or ""
    return bool(_FACT_STRONG.search(q) or (_FACT_RE.search(q) and _FACT_ASK.search(q)))


def _fact_topic(question: str) -> str:
    words = re.findall(r"[а-яёa-z0-9-]+", (question or "").lower())
    keep = [w for w in words
            if len(w) > 2 and w not in _FACT_STOP
            and not _FACT_RE.search(w) and not w.startswith("скиф")]
    return " ".join(keep)


def _points_to_texts(points) -> List[str]:
    out: List[str] = []
    for p in points or []:
        payload = getattr(p, "payload", None)
        if payload is None and isinstance(p, dict):
            payload = p.get("payload")
        raw = _pluck_text(payload)
        if raw:
            out.append(_clean_context(raw))
    return out


def _random_contexts(n: int = 10) -> List[str]:
    try:
        res = qdrant.query_points(
            collection_name=COLLECTION,
            query=qmodels.SampleQuery(sample=qmodels.Sample.RANDOM),
            limit=n,
            with_payload=True,
        )
        return _points_to_texts(getattr(res, "points", None))
    except Exception as e:
        fact_log.warning("[fact] random sample failed (%s), fallback to seed word", e)
        return get_contexts(random.choice(_FACT_SEEDS), top_k=n)


def _answer_fact_llm(question: str) -> str:
    topic = _fact_topic(question)
    pool = get_contexts(topic, top_k=12) if topic else _random_contexts(14)
    pool = [c for c in pool if len(c) >= 200] or pool
    contexts = random.sample(pool, min(6, len(pool)))
    fact_log.info("[fact] topic=%r pool=%d used=%d", topic, len(pool), len(contexts))
    if not contexts:
        return "Хм, сейчас ничего удивительного не нашлось. Спроси ещё разок!"

    instr = (
        "Перед тобой фрагменты научных текстов о скифах и их искусстве. "
        "Выбери из них ровно один факт, который удивит обычного слушателя, а не специалиста: "
        "необычный обычай, рекордная находка, неожиданный материал или размер, загадка, "
        "забавная деталь, конкретное число. Не бери сухие сведения о классификации, датировке, "
        "мастерских, орнаментах и технике изготовления, если в них нет ничего поразительного. "
        "Не склеивай факты из разных фрагментов и не добавляй «а ещё». "
        "Бери факт строго из фрагментов, ничего не выдумывай.\n"
        "Ответ будет озвучен вслух, поэтому:\n"
        "- начни с короткого интригующего зачина и меняй его: «А вы знали, что…», "
        "«Представьте себе:», «Мало кто знает, но…», «Удивительно, но…»;\n"
        "- никакой разметки, кавычек, ссылок, фамилий, инициалов и сокращений;\n"
        "- века и числа словами в нужном падеже;\n"
        "- одно-два предложения, не более 250 символов всего;\n"
        "- с азартом, как рассказчик, который сам поражён этим фактом."
    )
    prompt = (
        f"{instr}\n\n"
        f"Просьба слушателя:\n{question}\n\n"
        "Фрагменты:\n" + "\n---\n".join(contexts) + "\n\nФакт:"
    )
    try:
        resp = llm.invoke(prompt)
    except Exception as e:
        logger.error("[fact] LLM error: %s", e, exc_info=1)
        return "Упс! Что-то пошло не так! Попробуй еще!"
    text = getattr(resp, "content", None)
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    if not isinstance(text, str):
        text = getattr(resp, "text", "") or str(resp)
    text = (text or "").strip()
    return text if text else "Упс! Что-то пошло не так! Попробуй еще!"


# --- Факты из отобранной коллекции ---
from collections import deque

FACTS_COLLECTION = os.getenv("FACTS_COLLECTION", "facts")
FACT_MIN_SCORE = float(os.getenv("FACT_MIN_SCORE", "0.3"))
_FACT_OPENERS = [
    "А вот удивительный факт!",
    "Мало кто об этом знает.",
    "Представьте себе!",
    "Готовы удивиться?",
    "Вот история, в которую трудно поверить.",
    "Слушайте, это чистая правда.",
]
_NO_TOPIC_OPENER = "Про это у меня пока ничего нет, зато вот другое!"
_recent_facts = deque(maxlen=5)
_recent_openers = deque(maxlen=2)


def _pick_fact(cands):
    fresh = [c for c in cands if c[0] not in _recent_facts]
    fid, text = random.choice(fresh or cands)
    _recent_facts.append(fid)
    return text


def _pick_opener():
    o = random.choice([x for x in _FACT_OPENERS if x not in _recent_openers])
    _recent_openers.append(o)
    return o


def _topic_stems(topic: str):
    return [w[:max(3, len(w) - 2)] for w in topic.replace("ё", "е").split() if len(w) >= 3]


def answer_fact(question: str) -> str:
    topic = _fact_topic(question)
    try:
        pts, _ = qdrant.scroll(collection_name=FACTS_COLLECTION, limit=1000,
                               with_payload=True, with_vectors=False)
        all_facts = [(p.id, p.payload["fact"]) for p in pts if (p.payload or {}).get("fact")]
        if not all_facts:
            raise RuntimeError("facts collection is empty")
        opener, cands, how = None, all_facts, "random"
        if topic:
            stems = _topic_stems(topic)
            cands = [c for c in all_facts
                     if any(re.search(r"(?<![а-яa-z])" + re.escape(st), c[1].lower().replace("ё", "е"))
                            for st in stems)]
            how = f"words {stems}"
            if not cands:
                res = qdrant.query_points(collection_name=FACTS_COLLECTION,
                                          query=embeddings.embed_query(topic), limit=3,
                                          score_threshold=FACT_MIN_SCORE, with_payload=True)
                cands = [(p.id, p.payload["fact"]) for p in res.points if (p.payload or {}).get("fact")]
                how = f"vector {[round(p.score, 2) for p in res.points]}"
            if not cands:
                opener, cands, how = _NO_TOPIC_OPENER, all_facts, "no match"
        fact_log.info("[fact] topic=%r %s candidates=%d", topic, how, len(cands))
        return f"{opener or _pick_opener()} {_pick_fact(cands)}"
    except Exception as e:
        fact_log.warning("[fact] facts collection failed (%s), fallback to LLM", e)
        return _answer_fact_llm(question)
