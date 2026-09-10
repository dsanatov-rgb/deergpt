import os, re, uuid, statistics
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv(".env")
from openai import OpenAI
from qdrant_client import QdrantClient, models as qm

FILE, TARGET, MAXC = "herodotus4", 700, 900
CHAPTERS = {*range(5, 16), *range(23, 32), *range(59, 83), *range(103, 118)}
COLL = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
MODEL = os.getenv("FACT_MODEL", "gpt-5.4-mini")

text = open("herodotus4.txt", encoding="utf-8").read()
text = text[re.search(r"(?m)^1\.\s", text).start():]
end = text.rfind("\nПримечания")
text = text[:end] if end > 0 else text
parts = re.split(r"(?m)^(\d{1,3})\.\s", text)
chapters = [(int(parts[i]), parts[i + 1].strip()) for i in range(1, len(parts), 2)]
assert [n for n, _ in chapters] == list(range(1, 206)), "нумерация глав сбита"
chapters = [c for c in chapters if c[0] in CHAPTERS]

pieces = []
for n, body in chapters:
    if len(body) <= MAXC:
        pieces.append((n, n, body))
        continue
    buf = ""
    for sent in re.split(r"(?<=[.!?;])\s+", body):
        if buf and len(buf) + len(sent) + 1 > TARGET:
            pieces.append((n, n, buf))
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        pieces.append((n, n, buf))
chunks = []
for a, b, t in pieces:
    if chunks and chunks[-1][1] == a - 1 and len(chunks[-1][2]) + len(t) + 1 <= TARGET:
        pa, _, pt = chunks[-1]
        chunks[-1] = (pa, b, pt + "\n" + t)
    else:
        chunks.append((a, b, t))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE", "https://api.proxyapi.ru/openai/v1"),
                timeout=120, max_retries=2)
SUMMARY = ("Тебе дают фрагмент из «Истории» Геродота в старом переводе. Напиши одну строку современным языком, "
           "до 200 символов: о чём фрагмент и какие в нём существа, боги, животные, предметы и обычаи. "
           "Старые названия замени на принятые сегодня, например: полудева-полуехидна — змееногая дева, "
           "грифы — грифоны. Только то, что есть во фрагменте. Без вводных слов и без кавычек.")

def summarize(t):
    r = client.chat.completions.create(model=MODEL, max_completion_tokens=1500,
        messages=[{"role": "system", "content": SUMMARY}, {"role": "user", "content": t}])
    return re.sub(r"\s+", " ", r.choices[0].message.content or "").strip().strip("«»\"")[:250]

with ThreadPoolExecutor(4) as ex:
    summaries = list(ex.map(summarize, [t for _, _, t in chunks]))
docs = [f"Геродот, «История», книга четвёртая, глава {a}{'–' + str(b) if b != a else ''}. Кратко: {s}\n{t}"
        for (a, b, t), s in zip(chunks, summaries)]
lens = [len(d) for d in docs]
print(f"глав {len(chapters)} → фрагментов {len(docs)}, длина мин/сред/макс {min(lens)}/{int(statistics.mean(lens))}/{max(lens)}")
print("\n===== контроль: грифы, ехидна, плуг =====")
for d in docs:
    if re.search(r"гриф|ехидн|плуг", d, re.IGNORECASE):
        print("  " + d.split("\n", 1)[0])

vecs = [d.embedding for d in client.embeddings.create(
    model=os.getenv("EMBED_MODEL", "text-embedding-3-small"), input=docs).data]
qd = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
vp = qd.get_collection(COLL).config.params.vectors
assert getattr(vp, "size", None) == len(vecs[0]), f"вектор коллекции не совпадает: {vp}"
before = qd.count(COLL).count
qd.delete(COLL, points_selector=qm.FilterSelector(filter=qm.Filter(
    must=[qm.FieldCondition(key="file", match=qm.MatchValue(value=FILE))])), wait=True)
qd.upsert(COLL, wait=True, points=[qm.PointStruct(
    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"deergpt/{FILE}/{i}")), vector=v,
    payload={"fname": f"pdf_chunks/{FILE}__{i}.txt", "file": FILE, "chunk_idx": i,
             "path": f"pdf_chunks/{FILE}__{i}.txt", "text": d})
    for i, (d, v) in enumerate(zip(docs, vecs))])
print(f"\nколлекция {COLL}: было {before}, стало {qd.count(COLL).count}")
