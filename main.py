import logging
import os
import re
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import FileResponse
from whisper import load_model
from gtts import gTTS
from rag import answer_with_rag

logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

# --- Whisper модель ---
whisper_model = load_model("small")

def _sanitize_tts(text: str) -> str:
    text = re.sub(r"(Ключевые слова|Keywords).*$", "", text, flags=re.IGNORECASE | re.S)
    text = text.strip()
    return text[:800]

@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    input_path = f"/tmp/in_{uid}_{os.path.basename(file.filename)}"
    output_path = f"/tmp/tts_{uid}.mp3"

    with open(input_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        # STT
        result = whisper_model.transcribe(input_path, language="ru")
        query_text = (result.get("text") or "").strip()
        logging.info(f"[voice] STT text: {query_text!r}")

        # RAG
        try:
            rag_answer = answer_with_rag(query_text)
        except Exception:
            logging.exception("[voice] RAG error")
            rag_answer = ""

        # TTS
        tts_text = _sanitize_tts(rag_answer or "Извините, не удалось получить ответ.")
        tts = gTTS(tts_text, lang="ru")
        tts.save(output_path)

        # Отдать сразу MP3
        return FileResponse(output_path, media_type="audio/mpeg", filename=f"{uid}.mp3")

    except Exception:
        logging.exception("[voice] processing error")
        # Можно вернуть 500 JSON или простой текст
        return Response(content="processing error", media_type="text/plain", status_code=500)

    finally:
        # Убираем входной временный файл
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
