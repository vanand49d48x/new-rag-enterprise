from sentence_transformers import SentenceTransformer
from typing import List


# Load embedding model (adjust as needed)
model = SentenceTransformer("BAAI/bge-small-en")

def embed(text: str) -> List[float]:
    return model.encode(text, normalize_embeddings=True).tolist()

# 4. ✅ Vector DB Integration (Qdrant)
# File: backend/rag/vector_store.py

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from backend.rag.embedder import embed

client = QdrantClient(host="localhost", port=6333)


def index_chunks(chunks_with_metadata: List[dict]):
    points = []
    for chunk in chunks_with_metadata:
        vector = embed(chunk["text"])
        points.append(PointStruct(
            id=chunk["id"],
            vector=vector,
            payload=chunk["metadata"]
        ))
    client.upsert(collection_name="docs", points=points)