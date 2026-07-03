from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
client.recreate_collection(
    collection_name="pdf_chunks",
    vectors_config={"size": 3072, "distance": "Cosine"}
)
print("Коллекция pdf_chunks создана.")
