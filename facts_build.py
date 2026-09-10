import json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from qdrant_client import QdrantClient

MODEL = os.getenv("FACT_MODEL", "gpt-5.4-mini")
COLL = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
BATCH = 5
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
OUT, DONE = Path("facts_raw.jsonl"), Path("facts_done.txt")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE", "https://api.proxyapi.ru/openai/v1"),
                timeout=180, max_retries=2)
qd = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

SCALE = """Оценка удивительности wow от 1 до 5 — для человека, который ничего не знает о скифах. Ориентиры уровня (это образцы шкалы, а не факты для выдачи):
5 — хочется пересказать друзьям: Пётр Первый запретил переплавлять золото из сибирских курганов и велел свозить его в Петербург;
4 — ярко и неожиданно: сарматских женщин нередко хоронили с оружием; в Кунсткамеру за один раз привезли двести пятьдесят золотых вещей из сибирских могил;
3 — любопытно: на серебряных бляхах для конской сбруи изобразили слона;
2 — описание изображения или устройства конкретной вещи: мускулы хищника на рукояти топора показаны кругами;
1 — классификация, доли, датировки, технология, способы крепления, подробности изучения.
Ресурс посвящён скифскому звериному стилю: связь с его образами — зверями и фантастическими существами на вещах, золотом, мифами и обычаями, которые объясняют эти образы, — поднимает оценку; сведения без связи с образами и искусством (войны, география, политика) — не выше 3. История находок, необычные обычаи, числа и рекорды тоже поднимают оценку; сухие искусствоведческие подробности её опускают."""

EXTRACT = """Ты редактор научно-популярной радиопередачи о скифах и их соседях. Тебе дают пронумерованные фрагменты научных текстов.
Выпиши только факты, которые сам оцениваешь на 3 и выше. Большинство фрагментов таких фактов не содержат — тогда верни [].
Правила:
- только то, что прямо и однозначно сказано во фрагменте; без оценок и домыслов вроде «внезапно», «очень неожиданно»; предположение автора не выдавай за факт; обрывочный или неясный фрагмент пропускай;
- понятно без фрагмента и без специальных знаний: назови вещь обычными словами и народ — скифы, сарматы, их соседи; не называй сарматов скифами; без фамилий учёных, названий музеев, курганов, сёл и типов находок — вместо них скажи, где примерно: в Крыму, в Сибири, в Поволжье; без терминов вроде фалар, наносник, налобник, тамга, псалий — скажи, что это за вещь;
- одна мысль на факт; из одного фрагмента не больше одного факта;
- если фрагмент — рассказ древнего автора или легенда, так и подай: Геродот пишет, что…; по скифской легенде…; не выдавай легенду или рассказ за установленный факт;
- числа и перечисления сверяй с фрагментом;
- разговорные слова, без цитат и кавычек, без сокращений, века и числа словами;
- одно-два предложения, до 230 символов, без зачина вроде «А вы знали».
""" + SCALE + """
Верни только JSON-массив: [{"n": номер фрагмента, "fact": "...", "wow": 4}]."""

VERIFY = """Ты строгий научный редактор. Тебе дают фрагменты научных текстов и факты, пересказанные из них для радиопередачи.
Проверь каждый факт. ok = true только если одновременно:
1) всё прямо подтверждается фрагментом: числа, перечисления, народ, материал, место; нет домыслов, преувеличений и оценок, которых нет во фрагменте; предположение автора не выдано за установленный факт; рассказ древнего автора или легенда поданы именно как рассказ или легенда;
2) понятно человеку без специальных знаний: нет фамилий, названий музеев, курганов, сёл и типов находок, нет терминов без пояснения; ясно, о какой вещи и о каком народе речь;
3) одна законченная мысль, понятая правильно — не перепутаны вещи, эпохи, народы.
Если хоть что-то не так — ok = false, в why коротко, что именно.
Заново оцени wow по шкале ниже — честно, не завышая и не занижая.
""" + SCALE + """
Верни только JSON-массив: [{"i": номер факта, "ok": true, "why": "", "wow": 3}]."""


def toint(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def ask(system, user):
    r = client.chat.completions.create(model=MODEL, max_completion_tokens=4000,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens


def parse_arr(raw):
    i = raw.find("[")
    if i < 0:
        return []
    val, _ = json.JSONDecoder().raw_decode(raw[i:])
    return [x for x in val if isinstance(x, dict)] if isinstance(val, list) else []


chunks, offset = [], None
while True:
    pts, offset = qd.scroll(collection_name=COLL, limit=256, offset=offset,
                            with_payload=True, with_vectors=False)
    for p in pts:
        pl = p.payload or {}
        if isinstance(pl.get("text"), str) and pl["text"].strip():
            chunks.append({"key": (str(pl.get("file", "")), toint(pl.get("chunk_idx"))),
                           "fname": pl.get("fname", str(p.id)), "text": re.sub(r" Кратко: [^\n]*", "", pl["text"], count=1)})
    if offset is None:
        break
chunks.sort(key=lambda c: c["key"])
FILE_FILTER = os.getenv("FACT_FILE")
if FILE_FILTER:
    chunks = [c for c in chunks if c["key"][0] == FILE_FILTER]
    OUT, DONE = Path(f"facts_raw_{FILE_FILTER}.jsonl"), Path(f"facts_done_{FILE_FILTER}.txt")
batches = [chunks[i:i + BATCH] for i in range(0, len(chunks), BATCH)]
done = set(DONE.read_text().split()) if DONE.exists() else set()
todo = [i for i in range(len(batches)) if str(i) not in done]
if LIMIT:
    todo = sorted(random.sample(todo, min(LIMIT, len(todo))))
print(f"модель={MODEL} фрагментов={len(chunks)} батчей={len(batches)} "
      f"готово={len(done)} в работе={len(todo)}", flush=True)


def work(bi):
    part = batches[bi]
    frag = "\n\n".join(f"[{k}]\n{c['text']}" for k, c in enumerate(part, 1))
    raw, pi, po = ask(EXTRACT, frag)
    facts = [f for f in parse_arr(raw)
             if isinstance(f.get("n"), int) and 1 <= f["n"] <= len(part) and f.get("fact") and toint(f.get("wow")) >= 3]
    recs = []
    if facts:
        listing = "\n".join(f"{k}. [фрагмент {f['n']}] {f['fact']}" for k, f in enumerate(facts, 1))
        raw2, pi2, po2 = ask(VERIFY, f"Фрагменты:\n{frag}\n\nФакты:\n{listing}")
        pi, po = pi + pi2, po + po2
        verdicts = {toint(v.get("i")): v for v in parse_arr(raw2)}
        for k, f in enumerate(facts, 1):
            v = verdicts.get(k, {})
            recs.append({"batch": bi, "src": part[f["n"] - 1]["fname"], "fact": str(f["fact"]).strip(),
                         "wow_extract": toint(f.get("wow")), "wow": toint(v.get("wow")),
                         "ok": v.get("ok") is True, "why": v.get("why") or ("" if v else "нет вердикта")})
    return recs, pi, po


tin = tout = 0
new = []
with ThreadPoolExecutor(4) as ex:
    futs = {ex.submit(work, bi): bi for bi in todo}
    for n, fut in enumerate(as_completed(futs), 1):
        bi = futs[fut]
        try:
            recs, pi, po = fut.result()
        except Exception as e:
            print(f"ОШИБКА батч {bi}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        with OUT.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with DONE.open("a") as f:
            f.write(f"{bi}\n")
        tin, tout = tin + pi, tout + po
        new += recs
        if n % 20 == 0:
            print(f"… {n}/{len(todo)} батчей, фактов {len(new)}, прошли {sum(r['ok'] for r in new)}", flush=True)

ok = [r for r in new if r["ok"]]
print(f"\nИтог: батчей {len(todo)}, фактов {len(new)}, прошли проверку {len(ok)}, "
      f"из них wow>=4: {sum(r['wow'] >= 4 for r in ok)}, токены in/out={tin}/{tout}")
if LIMIT or FILE_FILTER:
    for r in sorted(new, key=lambda r: (not r["ok"], -r["wow"])):
        ww = f"[{r['wow_extract']}→{r['wow']}]"
        mark = f"OK  {ww}" if r["ok"] else f"НЕТ {ww} {r['why']}"
        print(f"\n{mark}\n    {r['fact']}  ({r['src']})")
