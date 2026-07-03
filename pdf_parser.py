from __future__ import annotations

import os
import re
from typing import List

import fitz  # PyMuPDF
import regex  # pip install regex
from tqdm import tqdm

PDFDIR = "pdf_files"
CHUNKSDIR = "pdf_chunks"

# ========== Параметры ==========
SENTS_PER_CHUNK = 5        # базовый размер чанка по предложениям
MIN_CHUNK_LEN = 120        # минимальная длина чанка (символы) до склейки
ALNUM_RATIO_THR = 0.60     # минимальная доля буквенно-цифровых символов
CYR_THR = 0.40             # минимальная доля кириллицы в чанке
LAT_MAX = 0.50             # если латиницы больше порога и кириллицы мало — дроп
MIN_FULL_SENT = 2          # минимум полных предложений в чанке
BIBLIO_THR = 0.40          # порог «библиографичности»

os.makedirs(CHUNKSDIR, exist_ok=True)


# ---------- Нормализация текста ----------

def _fix_hyphen_breaks(text: str) -> str:
    # "символи- зирует" -> "символизирует"
    return re.sub(r'(\w+)[-‐–—]\s+(\w+)', r'\1\2', text)


def _keep_surname_drop_initials(text: str) -> str:
    # "И.И. Иванов" -> "Иванов", "A.B. Smith" -> "Smith"
    return re.sub(r'\b[А-ЯA-Z]\.[А-ЯA-Z]\.?\s*([А-Яа-яA-Za-zЁё-]+)', r'\1', text)


def _expand_abbrev(text: str) -> str:
    text = re.sub(r'н\.?\s?э\.', 'нашей эры', text, flags=re.IGNORECASE)
    text = re.sub(r'т\.?\s?е\.', 'то есть', text, flags=re.IGNORECASE)
    return text


def _replace_century_safe(text: str) -> str:
    text = re.sub(r'(\d+|[IVXLCDM]+)\s*[-–—]\s*(\d+|[IVXLCDM]+)\s*вв\.', r'\1–\2 века', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+|[IVXLCDM]+)\s*в\.', r'\1 век', text, flags=re.IGNORECASE)
    text = re.sub(r'\bв\.(?=\s|$)', 'веке', text, flags=re.IGNORECASE)
    return text


def _remove_publishers_and_pages(text: str) -> str:
    # Убирать хвосты типа "/ М., Изд-во..., 1999" и "стр. 123"
    text = re.sub(r'/.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(стр\.|pages|p\.|c\.)\s*\d+', '', text, flags=re.IGNORECASE)
    return text


def _drop_english_noise_soft(text: str) -> str:
    # Оставлять строки, где >=25% символов — буквы (минимизирует «с. 93», «№», «табл.» и т.п.)
    lines = text.split('\n')
    kept = []
    for line in lines:
        letters = sum(c.isalpha() for c in line)
        if (letters / max(len(line), 1)) >= 0.25:
            kept.append(line)
    return '\n'.join(kept)


def normalize_text(t: str) -> str:
    t = t.replace('\r', '\n')
    t = re.sub(r'\f', '', t)
    t = re.sub(r'[‐–—]', '-', t)
    t = re.sub(r'…', '...', t)
    t = re.sub(r'([.,;!?]){2,}', r'\1', t)

    t = _keep_surname_drop_initials(t)
    t = _remove_publishers_and_pages(t)
    t = _fix_hyphen_breaks(t)
    t = _expand_abbrev(t)
    t = _replace_century_safe(t)
    t = _drop_english_noise_soft(t)

    # Финальная чистка пробелов/переносов
    t = re.sub(r'[ \t\u00A0]+', ' ', t)
    t = re.sub(r'\s*\n\s*', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


# ---------- Метрики и фильтры ----------

def alnum_ratio(t: str) -> float:
    if not t:
        return 0.0
    alnum = sum(ch.isalnum() for ch in t)
    return alnum / max(1, len(t))


def cyr_ratio(text: str) -> float:
    cyr = len(regex.findall(r"\p{Cyrillic}", text))
    return cyr / max(1, len(text))


def lat_ratio(text: str) -> float:
    lat = len(regex.findall(r"\p{Latin}", text))
    return lat / max(1, len(text))


def full_sentence_count(text: str) -> int:
    # Предложение = фраза, заканчивающаяся . ! ? и длиннее 40 символов
    sents = re.findall(r"[^\n]+?[.!?](?:\s|$)", text)
    return len([s for s in sents if len(s.strip()) >= 40])


BIB_PAT = re.compile(r"(?:\bс\.|\bтабл\.|\bрис\.|\bкат\.|\b№|\bfig\.|\btab\.|\bcat\.)", re.IGNORECASE)
YEAR_PAT = re.compile(r"\b(19[0-9]{2}|20[0-4][0-9])\b")


def biblio_ratio(text: str) -> float:
    tokens = re.findall(r"\b[\w\-]+\b", text)
    if not tokens:
        return 1.0
    short = sum(1 for t in tokens if len(t) <= 3)
    marks = len(BIB_PAT.findall(text)) + len(YEAR_PAT.findall(text))
    return (short + marks) / max(1, len(tokens))


def accept_chunk(text: str) -> bool:
    # 1) Кириллица должна быть заметной, а латиница не доминировать
    if cyr_ratio(text) < CYR_THR and lat_ratio(text) > LAT_MAX:
        return False
    # 2) Достаточно «полных» предложений
    if full_sentence_count(text) < MIN_FULL_SENT:
        return False
    # 3) Не «библиография/списки»
    if biblio_ratio(text) >= BIBLIO_THR:
        return False
    # 4) Общая «осмысленность»
    if alnum_ratio(text) < ALNUM_RATIO_THR:
        return False
    return True


# ---------- Чанкинг ----------

def chunk_by_sentences(text: str, sents_per_chunk: int = SENTS_PER_CHUNK) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: List[str] = []
    for i in range(0, len(sentences), sents_per_chunk):
        chunks.append(" ".join(sentences[i:i + sents_per_chunk]))
    return chunks


def merge_short_chunks(chunks: List[str], min_len: int = MIN_CHUNK_LEN) -> List[str]:
    out: List[str] = []
    buf = ''
    for c in chunks:
        c = c.strip()
        if len(c) < min_len:
            buf = (buf + ' ' + c).strip()
            continue
        if buf:
            out.append((buf + ' ' + c).strip())
            buf = ''
        else:
            out.append(c)
    if buf:
        if out:
            out[-1] = (out[-1] + ' ' + buf).strip()
        else:
            out.append(buf)
    return out


# ---------- Извлечение и обработка ----------

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    return "\n".join(text_parts)


def process_all_pdfs() -> None:
    pdfs = [f for f in os.listdir(PDFDIR) if f.lower().endswith('.pdf')]
    for fname in tqdm(sorted(pdfs)):
        path = os.path.join(PDFDIR, fname)
        raw = extract_text_from_pdf(path)
        cleaned = normalize_text(raw)

        chunks = chunk_by_sentences(cleaned, sents_per_chunk=SENTS_PER_CHUNK)
        chunks = merge_short_chunks(chunks, min_len=MIN_CHUNK_LEN)
        # Композитный фильтр качества
        chunks = [c for c in chunks if accept_chunk(c)]

        base = os.path.splitext(fname)[0]
        for i, ch in enumerate(chunks):
            if not ch.strip():
                continue
            out = os.path.join(CHUNKSDIR, f"{base}__{i}.txt")
            with open(out, "w", encoding="utf-8") as f:
                f.write(ch)


if __name__ == "__main__":
    os.makedirs(CHUNKSDIR, exist_ok=True)
    process_all_pdfs()
