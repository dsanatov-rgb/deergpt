import logging
import os
import re
import shutil
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import FileResponse
from gtts import gTTS
from openai import OpenAI

from rag import answer_with_rag, answer_fact, is_fact_request
import time
import activity

logging.basicConfig(level=logging.INFO)

app = FastAPI()
activity.init()

# --- STT через proxyapi.ru (вместо локального Whisper) ---
stt_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.proxyapi.ru/openai/v1"),
)
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "echo")
TTS_INSTRUCTIONS = os.getenv(
    "TTS_INSTRUCTIONS",
    "Говори по-русски живо и тепло, как увлечённый экскурсовод, влюблённый в свою тему. "
    "Улыбка в голосе, выразительная интонация, естественные паузы. Темп чуть бодрее обычного.",
)
TTS_FACT_INSTRUCTIONS = os.getenv(
    "TTS_FACT_INSTRUCTIONS",
    "Говори по-русски как азартный рассказчик, который делится поразительным фактом и сам от него в восторге. "
    "Первую фразу — заговорщицки, чуть тише, с интригующей паузой. Дальше — с нарастающим удивлением и энергией, "
    "ударение на самом неожиданном месте. Живо, с улыбкой, без театральной фальши.",
)
tts_client = stt_client.with_options(timeout=15.0, max_retries=0)
tts_log = logging.getLogger("uvicorn.error")


def _synthesize(text: str, path: str, instructions: str = "") -> None:
    """Озвучка через OpenAI TTS; при сбое — запасной gTTS."""
    import time
    t0 = time.monotonic()
    try:
        with tts_client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            instructions=instructions or TTS_INSTRUCTIONS,
            response_format="mp3",
        ) as speech:
            speech.stream_to_file(path)
        tts_log.info("TTS %s%s: %.2f s, %d chars", TTS_VOICE, " [fact]" if instructions else "", time.monotonic() - t0, len(text))
    except Exception as e:
        tts_log.warning("TTS failed after %.2f s (%s), fallback to gTTS", time.monotonic() - t0, e)
        gTTS(text, lang="ru").save(path)


@app.get("/")
def root():
    return {"status": "ok"}


ROMAN = {
    "I": "первого", "II": "второго", "III": "третьего", "IV": "четвёртого",
    "V": "пятого", "VI": "шестого", "VII": "седьмого", "VIII": "восьмого",
    "IX": "девятого", "X": "десятого", "XI": "одиннадцатого", "XII": "двенадцатого",
    "XIII": "тринадцатого", "XIV": "четырнадцатого", "XV": "пятнадцатого",
    "XVI": "шестнадцатого", "XVII": "семнадцатого", "XVIII": "восемнадцатого",
    "XIX": "девятнадцатого", "XX": "двадцатого", "XXI": "двадцать первого",
}


def _roman(match):
    return ROMAN.get(match.group(1).upper(), match.group(1)) + " " + match.group(2)

ROMAN = {
    "I": "первого", "II": "второго", "III": "третьего", "IV": "четвёртого",
    "V": "пятого", "VI": "шестого", "VII": "седьмого", "VIII": "восьмого",
    "IX": "девятого", "X": "десятого", "XI": "одиннадцатого", "XII": "двенадцатого",
    "XIII": "тринадцатого", "XIV": "четырнадцатого", "XV": "пятнадцатого",
    "XVI": "шестнадцатого", "XVII": "семнадцатого", "XVIII": "восемнадцатого",
    "XIX": "девятнадцатого", "XX": "двадцатого", "XXI": "двадцать первого",
}

def _roman(m):
    word = ROMAN.get(m.group(1).upper(), m.group(1))
    return word + " век"

ABBR = [
    (r"\bдо\s+н\.?\s?э\.?", "до нашей эры"),
    (r"\bн\.?\s?э\.?", "нашей эры"),
    (r"\bв\.\s?в\.", "веков"),
    (r"\bвв\.", "веков"),
    (r"\bв\.(?=\s|$)", "века"),
    (r"\bг\.(?=\s|$)", "года"),
    (r"\bгг\.", "годов"),
    (r"\bт\.\s?е\.", "то есть"),
    (r"\bт\.\s?д\.", "так далее"),
    (r"\bт\.\s?п\.", "тому подобное"),
    (r"\bс\.(?=\s?\d)", "страница"),
]


def _sanitize_tts(text: str) -> str:
    text = re.sub(r"(Ключевые слова|Keywords).*$", "", text, flags=re.IGNORECASE | re.S)

    # markdown-разметка
    text = re.sub(r"[*_`#>|]+", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # ссылки на источники: [1], (Иванов, 2020), (с. 15)
    text = re.sub(r"\[\s*\d+(\s*[,-]\s*\d+)*\s*\]", " ", text)
    text = re.sub(r"\([^)]{0,40}\d{4}[^)]{0,10}\)", " ", text)

    # инициалы: А. С. Пушкин -> Пушкин
    text = re.sub(r"\b[А-ЯЁA-Z]\.\s?(?=[А-ЯЁA-Z]\.|\s?[А-ЯЁA-Z][а-яёa-z])", "", text)

    # сокращения
    text = re.sub(r"\b([IVXLC]{1,6})\s*(?:вв?\.|век[а-я]*)", _roman, text)
    for pattern, repl in ABBR:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"(нашей эры|века|годов|года)\s+(?=[А-ЯЁ])", r"\1. ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", ". ", text)
    text = text.strip()
    if len(text) > 700:
        cut = text[:700]
        dot = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        text = cut[: dot + 1] if dot > 300 else cut
    return text


def _transcribe(path: str) -> str:
    with open(path, "rb") as f:
        result = stt_client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            language="ru",
        )
    return (result.text or "").strip()



# --- Авторизация по жетонам ---
import hashlib
import hmac
from datetime import datetime
from fastapi import Request, HTTPException

TOKENS_FILE = os.getenv("TOKENS_FILE", "tokens.txt")
AUTH_SECRET = os.getenv("AUTH_SECRET", "change-me")
ACCESS_LOG = os.getenv("ACCESS_LOG", "access_tokens.log")


def _load_tokens() -> set:
    try:
        with open(TOKENS_FILE, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except OSError:
        logging.exception("[auth] cannot read tokens file")
        return set()


def _sign(token: str) -> str:
    digest = hmac.new(AUTH_SECRET.encode(), token.encode(), hashlib.sha256).hexdigest()
    return digest


def _cookie_token(request: Request):
    """Жетон, которому соответствует кука, или None."""
    sig = request.cookies.get("deergpt_auth")
    if not sig:
        return None
    for t in _load_tokens():
        if hmac.compare_digest(sig, _sign(t)):
            return t
    return None


def _valid_cookie(request: Request) -> bool:
    return _cookie_token(request) is not None


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return (request.headers.get("x-real-ip") or fwd.split(",")[0].strip()
            or (request.client.host if request.client else ""))


def _mask(password: str) -> str:
    return f"{password[:2]}…({len(password)})" if password else ""


@app.post("/auth")
async def auth(request: Request):
    body = await request.json()
    password = (body.get("password") or "").strip()
    if password and password in _load_tokens():
        try:
            with open(ACCESS_LOG, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{password}\n")
        except OSError:
            pass
        activity.log_event("auth_ok", password, _client_ip(request), request.headers.get("user-agent"))
        resp = Response(content='{"ok":true}', media_type="application/json")
        resp.set_cookie(
            "deergpt_auth",
            _sign(password),
            max_age=60 * 60 * 24 * 30,
            httponly=True,
            samesite="lax",
        )
        return resp
    activity.log_event("auth_fail", _mask(password), _client_ip(request), request.headers.get("user-agent"))
    return Response(content='{"ok":false}', media_type="application/json", status_code=401)


@app.post("/voice")
async def voice(request: Request, file: UploadFile = File(...)):
    token = _cookie_token(request)
    if token is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    t_start = time.monotonic()
    rec = {"token": token, "ip": _client_ip(request), "status": "error"}

    uid = uuid.uuid4().hex
    safe_name = os.path.basename(file.filename or "audio.webm")
    input_path = f"/tmp/in_{uid}_{safe_name}"
    output_path = f"/tmp/tts_{uid}.mp3"

    with open(input_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        t = time.monotonic()
        query_text = _transcribe(input_path)
        rec["question"], rec["stt_s"] = query_text, round(time.monotonic() - t, 2)
        logging.info(f"[voice] STT text: {query_text!r}")

        fact_mode = is_fact_request(query_text)
        rec["mode"] = "fact" if fact_mode else "rag"
        rag_status = "ok"
        t = time.monotonic()
        try:
            rag_answer = answer_fact(query_text) if fact_mode else answer_with_rag(query_text)
        except Exception:
            logging.exception("[voice] RAG error")
            rag_answer = ""
            rag_status = "rag_error"
        rec["answer_s"] = round(time.monotonic() - t, 2)

        tts_text = _sanitize_tts(rag_answer or "Извините, не удалось получить ответ.")
        rec["answer"] = tts_text
        t = time.monotonic()
        _synthesize(tts_text, output_path, TTS_FACT_INSTRUCTIONS if fact_mode else "")
        rec["tts_s"] = round(time.monotonic() - t, 2)
        rec["status"] = rag_status

        return FileResponse(output_path, media_type="audio/mpeg", filename=f"{uid}.mp3")

    except Exception:
        logging.exception("[voice] processing error")
        return Response(content="processing error", media_type="text/plain", status_code=500)

    finally:
        rec["total_s"] = round(time.monotonic() - t_start, 2)
        activity.log_qa(**rec)
        try:
            os.remove(input_path)
        except OSError:
            pass


@app.get("/tts/{uid}.mp3")
async def get_tts(uid: str):
    path = f"/tmp/tts_{uid}.mp3"
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg", filename=f"{uid}.mp3")
    return {"status": "not_found"}


# --- Админка ---
import asyncio
from fastapi.responses import JSONResponse

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
ADMIN_COOKIE = "deergpt_admin"


def _admin_sig() -> str:
    return hmac.new(AUTH_SECRET.encode(), ("admin:" + ADMIN_PASSWORD).encode(), hashlib.sha256).hexdigest()


def _require_admin(request: Request) -> None:
    sig = request.cookies.get(ADMIN_COOKIE, "")
    if not ADMIN_PASSWORD or not sig or not hmac.compare_digest(sig, _admin_sig()):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.post("/admin/login")
async def admin_login(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    password = (body.get("password") or "").strip()
    ip, ua = _client_ip(request), request.headers.get("user-agent")
    if ADMIN_PASSWORD and hmac.compare_digest(password.encode(), ADMIN_PASSWORD.encode()):
        activity.log_event("admin_ok", None, ip, ua)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(ADMIN_COOKIE, _admin_sig(), max_age=60 * 60 * 12,
                        httponly=True, secure=True, samesite="strict", path="/api/admin")
        return resp
    activity.log_event("admin_fail", _mask(password), ip, ua)
    await asyncio.sleep(1.5)
    return JSONResponse({"ok": False}, status_code=401)


@app.post("/admin/logout")
async def admin_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(ADMIN_COOKIE, path="/api/admin")
    return resp


@app.get("/admin/me")
def admin_me(request: Request):
    _require_admin(request)
    return {"ok": True}


@app.get("/admin/summary")
def admin_summary(request: Request):
    _require_admin(request)
    rows = activity.summary()
    known = _load_tokens()
    seen = {r["token"] for r in rows}
    for r in rows:
        r["active"] = r["token"] in known
    for t in sorted(known - seen):
        rows.append({"token": t, "logins": 0, "questions": 0,
                     "first_seen": None, "last_seen": None, "active": True})
    return {"tokens": rows, "total_tokens": len(known)}


@app.get("/admin/qa")
def admin_qa(request: Request, token: str = "", limit: int = 200):
    _require_admin(request)
    return {"items": activity.recent_qa(token or None, max(1, min(limit, 1000)))}


@app.get("/admin/events")
def admin_events(request: Request, limit: int = 200):
    _require_admin(request)
    return {"items": activity.recent_events(max(1, min(limit, 1000)))}


# --- Счётчик заходов ---
import json


@app.post("/hit")
async def hit(request: Request):
    ref = None
    try:
        raw = await request.body()
        if raw and len(raw) <= 2048:
            ref = str(json.loads(raw).get("ref") or "")[:300] or None
    except Exception:
        pass
    activity.log_hit(_client_ip(request), request.headers.get("user-agent"), ref, _cookie_token(request))
    return Response(status_code=204)


@app.get("/admin/hits")
def admin_hits(request: Request, tz: int = 0):
    _require_admin(request)
    return activity.hits_stats(-max(-840, min(tz, 840)))

