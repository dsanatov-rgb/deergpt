from __future__ import annotations

import glob
import hashlib
import os
import re
from typing import Iterable, Dict, Any, List

import numpy as np
from qdrant_client import QdrantClient, models

# --- Конфиг через переменные окружения ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1536"))
RECREATE = os.getenv("RECREATE_COLLECTION", "false").lower() == "true"

CHUNKS_DIR = "pdf_chunks"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# --- ID по хэшу пути чанка (детерминированно) ---
def stable_id(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:16], 16)


def ensure_collection() -> None:
    exists = client.collection_exists(COLLECTION_NAME)
    if RECREATE and exists:
        print(f"Drop collection {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
        exists = False
    if not exists:
        print(f"Create collection {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
        )


def parse_embeddings_tsv(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                fname, emb_str = line.split("\t", 1)
            elif " " in line:
                fname, emb_str = line.split(" ", 1)
            else:
                print(f"[skip] bad line {lineno}: {repr(line)}")
                continue

            vec = np.array([float(x) for x in re.split(r"[ \t]+", emb_str) if x], dtype=np.float32)
            if vec.size != VECTOR_SIZE:
                raise ValueError(f"Dim mismatch at line {lineno} for {fname}: {vec.size} != {VECTOR_SIZE}")

            base = os.path.basename(fname)
            file_name, idx = (base.rsplit("__", 1) + ["-1"])[:2]
            idx = int(idx.split(".")[0]) if idx != "-1" else -1

            # Определяем реальный путь к чанку
            chunk_path = fname
            if not os.path.isabs(chunk_path) and not os.path.isfile(chunk_path):
                chunk_path = os.path.join(CHUNKS_DIR, base)

            text = ""
            if os.path.isfile(chunk_path):
                try:
                    with open(chunk_path, encoding="utf-8", errors="ignore") as cf:
                        text = cf.read().strip()
                except Exception as e:
                    print(f"[warn] cannot read chunk {chunk_path}: {e}")
            else:
                print(f"[warn] chunk file not found for {fname} -> {chunk_path}")

            yield {
                "id": stable_id(fname),
                "vector": vec.tolist(),
                "payload": {
                    "fname": fname,        # исходный путь/имя из embeddings.tsv
                    "file": file_name,     # базовое имя PDF без индекса
                    "chunk_idx": idx,      # индекс чанка
                    "path": chunk_path,    # реальный путь к файлу чанка
                    "text": text,          # текст чанка (может быть пустым, если файла нет)
                },
            }


def upsert_batches(points_iter: Iterable[Dict[str, Any]], batch_size: int = 500) -> int:
    buf: List[Dict[str, Any]] = []
    total = 0
    for p in points_iter:
        buf.append(p)
        if len(buf) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=buf)
            total += len(buf)
            print(f"Upserted: {total}")
            buf = []
    if buf:
        client.upsert(collection_name=COLLECTION_NAME, points=buf)
        total += len(buf)
        print(f"Upserted: {total}")
    return total


def delete_orphans() -> int:
    # Список живых чанков по локальной файловой системе
    live = set(glob.glob(os.path.join(CHUNKS_DIR, "**/*.txt"), recursive=True))
    print(f"Live chunks on disk: {len(live)}")
    # Прокрутка коллекции батчами и поиск тех, чьи fname отсутствуют локально
    deleted = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            limit=1000,
            offset=offset,
        )
        if not points:
            break
        to_delete = [
            pt.id
            for pt in points
            if not pt.payload
            or not pt.payload.get("fname")
            or os.path.join("", pt.payload["fname"]) not in live
        ]
        if to_delete:
            client.delete(collection_name=COLLECTION_NAME, points_selector=models.PointIdsList(points=to_delete))
            deleted += len(to_delete)
            print(f"Deleted orphans: {deleted}")
        if offset is None:
            break
    if deleted:
        client.optimize(COLLECTION_NAME)
    return deleted


if __name__ == "__main__":
    ensure_collection()
    total = upsert_batches(parse_embeddings_tsv("embeddings.tsv"), batch_size=500)
    print(f"Upsert total: {total}")
    removed = delete_orphans()
    print(f"Orphans removed: {removed}")
    print("Done.")
