#!/usr/bin/env bash
# Ночная копия журнала DeerGPT (activity.db) на той же ВМ; хранится KEEP_DAYS дней.
set -euo pipefail
SRC=/home/user1/rag2_project/activity.db
DST=/home/user1/deergpt_backups/activity
KEEP_DAYS=30

mkdir -p "$DST"
chmod 700 "$DST"
OUT="$DST/activity-$(date +%F).db"

/usr/bin/python3 - "$SRC" "$OUT" << 'PY'
import sqlite3, sys
src, out = sys.argv[1], sys.argv[2]
s = sqlite3.connect(src, timeout=10)
d = sqlite3.connect(out)
s.backup(d)
check = d.execute("PRAGMA integrity_check").fetchone()[0]
counts = {t: d.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("events", "qa", "hits")}
d.close()
s.close()
if check != "ok":
    sys.exit(f"integrity_check failed: {check}")
print("backup ok:", out, counts)
PY

chmod 600 "$OUT"
gzip -f "$OUT"
find "$DST" -name 'activity-*.db.gz' -mtime +"$KEEP_DAYS" -delete
