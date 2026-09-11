"""Журнал активности DeerGPT: входы и пары «вопрос — ответ» в SQLite."""
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timezone

DB_PATH = os.getenv(
    "ACTIVITY_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "activity.db"),
)
_lock = threading.Lock()
_log = logging.getLogger("uvicorn.error")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id    INTEGER PRIMARY KEY,
    ts    TEXT NOT NULL,
    kind  TEXT NOT NULL,
    token TEXT,
    ip    TEXT,
    ua    TEXT
);
CREATE TABLE IF NOT EXISTS qa (
    id       INTEGER PRIMARY KEY,
    ts       TEXT NOT NULL,
    token    TEXT,
    ip       TEXT,
    mode     TEXT,
    question TEXT,
    answer   TEXT,
    status   TEXT,
    stt_s    REAL,
    answer_s REAL,
    tts_s    REAL,
    total_s  REAL
);
CREATE INDEX IF NOT EXISTS idx_qa_token_ts ON qa(token, ts);
CREATE INDEX IF NOT EXISTS idx_qa_ts ON qa(ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS hits (
    id    INTEGER PRIMARY KEY,
    ts    TEXT NOT NULL,
    ip    TEXT,
    ua    TEXT,
    ref   TEXT,
    token TEXT
);
CREATE INDEX IF NOT EXISTS idx_hits_ts ON hits(ts);
"""

_QA_FIELDS = ("token", "ip", "mode", "question", "answer", "status",
              "stt_s", "answer_s", "tts_s", "total_s")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _write(sql: str, params: tuple) -> None:
    with _lock, closing(_connect()) as conn:
        with conn:
            conn.execute(sql, params)


def _read(sql: str, params: tuple = ()) -> list:
    with closing(_connect()) as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def init() -> None:
    with _lock, closing(_connect()) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
    try:
        os.chmod(DB_PATH, 0o600)
    except OSError:
        pass


def log_event(kind: str, token=None, ip=None, ua=None) -> None:
    """kind: auth_ok | auth_fail"""
    try:
        _write("INSERT INTO events (ts, kind, token, ip, ua) VALUES (?, ?, ?, ?, ?)",
               (_now(), kind, (token or "")[:64] or None, ip, (ua or "")[:200]))
    except Exception:
        _log.exception("[activity] log_event failed")


def log_qa(**fields) -> None:
    """mode: rag | fact; status: ok | rag_error | error"""
    try:
        values = tuple(fields.get(k) for k in _QA_FIELDS)
        cols = ", ".join(_QA_FIELDS)
        marks = ", ".join("?" * (len(_QA_FIELDS) + 1))
        _write(f"INSERT INTO qa (ts, {cols}) VALUES ({marks})", (_now(), *values))
    except Exception:
        _log.exception("[activity] log_qa failed")


def summary() -> list:
    """По каждому жетону: входы, вопросы, первый и последний визит."""
    return _read("""
        SELECT token,
               SUM(logins)    AS logins,
               SUM(questions) AS questions,
               MIN(first_ts)  AS first_seen,
               MAX(last_ts)   AS last_seen
        FROM (
            SELECT token, COUNT(*) AS logins, 0 AS questions,
                   MIN(ts) AS first_ts, MAX(ts) AS last_ts
            FROM events WHERE kind = 'auth_ok' GROUP BY token
            UNION ALL
            SELECT token, 0, COUNT(*), MIN(ts), MAX(ts)
            FROM qa GROUP BY token
        )
        GROUP BY token
        ORDER BY last_seen DESC
    """)


def recent_qa(token=None, limit: int = 200) -> list:
    if token:
        return _read("SELECT * FROM qa WHERE token = ? ORDER BY id DESC LIMIT ?", (token, limit))
    return _read("SELECT * FROM qa ORDER BY id DESC LIMIT ?", (limit,))


def recent_events(limit: int = 200) -> list:
    return _read("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))


# --- Заходы на страницу ---
HIT_MIN_INTERVAL = 10.0
_hit_lock = threading.Lock()
_last_hit = {}


def log_hit(ip, ua=None, ref=None, token=None) -> bool:
    """Заход на страницу; с одного IP не чаще раза в HIT_MIN_INTERVAL секунд."""
    now = time.monotonic()
    key = ip or "?"
    with _hit_lock:
        last = _last_hit.get(key)
        if last is not None and now - last < HIT_MIN_INTERVAL:
            return False
        _last_hit[key] = now
        if len(_last_hit) > 10000:
            cutoff = now - HIT_MIN_INTERVAL
            for k in [k for k, v in _last_hit.items() if v < cutoff]:
                del _last_hit[k]
    try:
        _write("INSERT INTO hits (ts, ip, ua, ref, token) VALUES (?, ?, ?, ?, ?)",
               (_now(), ip, (ua or "")[:200], (ref or "")[:300] or None, token))
        return True
    except Exception:
        _log.exception("[activity] log_hit failed")
        return False


def hits_stats(offset_min: int = 0) -> dict:
    """Статистика заходов; offset_min — сдвиг местного времени от UTC в минутах."""
    mod = f"{int(offset_min):+d} minutes"
    total = _read("""SELECT COUNT(*) AS n, COUNT(DISTINCT ip) AS ips,
                            COALESCE(SUM(token IS NOT NULL), 0) AS with_token, MIN(ts) AS since
                     FROM hits""")[0]
    today = _read("""SELECT COUNT(*) AS n, COUNT(DISTINCT ip) AS ips,
                            COALESCE(SUM(token IS NOT NULL), 0) AS with_token
                     FROM hits WHERE date(ts, ?) = date('now', ?)""", (mod, mod))[0]
    days = _read("""SELECT date(ts, ?) AS day, COUNT(*) AS n, COUNT(DISTINCT ip) AS ips
                    FROM hits GROUP BY day ORDER BY day DESC LIMIT 14""", (mod,))
    funnel = {"logins": 0, "questions": 0, "askers": 0}
    if total["since"]:
        s = total["since"]
        funnel = _read("""SELECT
            (SELECT COUNT(*) FROM events WHERE kind = 'auth_ok' AND ts >= ?) AS logins,
            (SELECT COUNT(*) FROM qa WHERE ts >= ?) AS questions,
            (SELECT COUNT(DISTINCT token) FROM qa WHERE ts >= ?) AS askers""", (s, s, s))[0]
    return {"total": total, "today": today, "days": days, "funnel": funnel}

