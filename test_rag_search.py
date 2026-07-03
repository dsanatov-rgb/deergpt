import openai
from qdrant_client import QdrantClient

# Настроить OpenAI через proxyAPI.ru (как в исходнике)
client_openai = openai.OpenAI(
    api_key="sk-yJpP8DJ6lMP9XphS64ElhiGcE5LK9wYk",
    base_url="https://api.proxyapi.ru/openai/v1",
)

def get_embedding(text):
    # Главное изменение: модель под размерность 1536 вместо 3072
    response = client_openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

query = "Перечисли признаки звериного стиля"
query_vector = get_embedding(query)

client_qdrant = QdrantClient(host="localhost", port=6333)

results = client_qdrant.query_points(
    collection_name="pdf_chunks",
    query=query_vector,
    limit=5
)

# Показываем отладочную структуру результатов (как в исходнике)
print(f"Тип results: {type(results)}")
if hasattr(results, "hits"):
    print(f"results.hits: {results.hits}")
elif hasattr(results, "points"):
    print(f"results.points: {results.points}")
else:
    print(f"results: {results}")

# Извлекаем найденные пункты для дальнейшей обработки
points = getattr(results, "hits", None) or getattr(results, "points", None) or []

for point in points:
    payload = getattr(point, "payload", {}) or {}
    fname = payload.get("fname")
    score = getattr(point, "score", None)
    print(f"{fname} | score: {score}")
    try:
        with open(fname, encoding="utf-8") as f:
            print(f.read())
    except Exception as e:
        print(f"Не удалось открыть {fname}: {e}")
    print("=" * 80)
