# Развёртывание DeerGPT

Секреты в репозитории не хранятся: `.env` и `tokens.txt` берутся из резервной копии.
Переменные `.env`: OPENAI_API_KEY, OPENAI_API_BASE, QDRANT_URL, QDRANT_COLLECTION, CHAT_MODEL,
EMBED_MODEL, STT_MODEL, AUTH_SECRET, CHAT_MAX_TOKENS; необязательные (есть значения по умолчанию):
TTS_MODEL, TTS_VOICE, TTS_INSTRUCTIONS, TTS_FACT_INSTRUCTIONS, FACTS_COLLECTION, FACT_MIN_SCORE.

1. Сервис: `deergpt.service` → `/etc/systemd/system/`, затем `sudo systemctl daemon-reload && sudo systemctl enable --now deergpt`.
2. Nginx: `nginx-deergpt.conf` → `/etc/nginx/sites-available/deergpt`, ссылка в `sites-enabled`, затем `sudo certbot --nginx -d deergpt.ru -d www.deergpt.ru`.
3. Фронтенд: `sudo cp -a frontend/. /var/www/deergpt/`. Источник фронтенда — только папка `frontend/` репозитория.
4. Факты: `venv/bin/python facts_load.py` пересоздаёт коллекцию `facts` из `facts_curated.jsonl`.
5. Геродот: `venv/bin/python herodotus_load.py` перезаписывает фрагменты `herodotus4` в основной коллекции.
6. Qdrant (снято с работающего контейнера): образ qdrant/qdrant:v1.15.4, политика перезапуска always, тома: /home/user1/rag2_project/qdrant_data → /qdrant/storage;  порты: 6333/tcp -> 127.0.0.1:6333 
