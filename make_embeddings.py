import os
from dotenv import load_dotenv
load_dotenv()
import re
import time
from pathlib import Path
from typing import Iterable, List

import openai
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# -------- Конфигурация --------
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.proxyapi.ru/openai/v1"
EMB_MODEL = "text-embedding-3-small"   # размерность 1536
CHUNKS_DIR = "pdf_chunks"              # папка с *.txt чанками
COLLECTION = "pdf_chunks"              # имя коллекции в Qdrant
BATCH = 128                            # размер батча для upsert
MAX_CHARS = 3500
RETRIES = 4
BACKOFF = 2.0

# -------- Диагностика --------
print("DBG:file =", __file__)
print("DBG:openai ver =", getattr(openai, "__version__", "unknown"))
print("DBG:model =", EMB_MODEL)
print("DBG:base_url =", BASE_URL)
print("DBG:key_prefix =", API_KEY[:7], "len =", len(API_KEY))

# OpenAI клиент
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, organization=None, project=None)

# Qdrant клиент
qdrant = QdrantClient(host="localhost", port=6333)


def ensure_collection() -> None:
    # Создаём коллекцию, если её нет
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print("DBG: created collection", COLLECTION)
    else:
        # Проверим размерность
        info = qdrant.get_collection(COLLECTION)
        dim = info.vectors_count  # not dimension; fetch from config:
        # В новых версиях удобнее проверить через info.config.params.vectors.size
        try:
            dim = info.config.params.vectors.size
        except Exception:
            pass
        print("DBG: collection exists:", COLLECTION, "dim:", dim)


def sent_tokenize_simple(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def get_embedding(text: str) -> List[float]:
    text = text[:MAX_CHARS]
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = client.embeddings.create(model=EMB_MODEL, input=text)
            return r.data[0].embedding
        except Exception as e:
            last_err = e
            print(f"[emb] attempt {attempt} failed: {e}")
            time.sleep(BACKOFF * attempt)
    raise RuntimeError(f"embedding failed after {RETRIES} retries: {last_err}")


def iter_chunk_files() -> Iterable[Path]:
    yield from Path(CHUNKS_DIR).rglob("*.txt")


def parse_fname_and_idx(p: Path) -> tuple[str, int]:
    # Ожидаем имена вида <base>.<idx>.txt или <base>_chunk<idx>.txt
    stem = p.stem  # без .txt
    m = re.search(r"[._-](\d+)$", stem) or re.search(r"chunk(\d+)$", stem, re.I)
    idx = int(m.group(1)) if m else 0
    base = re.sub(r"([._-]\d+|chunk\d+)$", "", stem, flags=re.I)
    return base, idx


def main() -> None:
    ensure_collection()

    points: List[PointStruct] = []
    pid = 0
    total = 0

    for p in iter_chunk_files():
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            continue

        emb = get_embedding(txt)
        fname, idx = parse_fname_and_idx(p)

        payload = {
            "text": txt,
            "fname": fname,
            "file": p.name,
            "chunk_idx": idx,
        }

        points.append(PointStruct(id=pid, vector=emb, payload=payload))
        pid += 1
        total += 1

        if len(points) >= BATCH:
            qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)
            print(f"upsert: {total} (batch={BATCH})", flush=True)
            points = []

    if points:
        qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)
        print(f"upsert: {total} (final)", flush=True)

    print(f"ok: {total}")


if __name__ == "__main__":
    main()
