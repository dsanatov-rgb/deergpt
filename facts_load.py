import json, os, sys
from dotenv import load_dotenv
load_dotenv(".env")
from openai import OpenAI
from qdrant_client import QdrantClient, models as qm

SRC = sys.argv[1] if len(sys.argv) > 1 else "facts_curated.jsonl"
COLL = os.getenv("FACTS_COLLECTION", "facts")
rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE", "https://api.proxyapi.ru/openai/v1"))
emb = client.embeddings.create(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
                               input=[r["fact"] for r in rows])
vecs = [d.embedding for d in emb.data]
qd = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
if qd.collection_exists(COLL):
    qd.delete_collection(COLL)
qd.create_collection(COLL, vectors_config=qm.VectorParams(size=len(vecs[0]), distance=qm.Distance.COSINE))
qd.upsert(COLL, points=[qm.PointStruct(id=int(r["id"]), vector=v,
          payload={"fact": r["fact"], "src": r.get("src", "")}) for r, v in zip(rows, vecs)])
print(f"{COLL}: загружено {qd.count(COLL).count} фактов, вектор {len(vecs[0])}")
