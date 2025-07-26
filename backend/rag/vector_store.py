# File: backend/rag/vector_store.py

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from backend.rag.embedder import embed
from typing import List
import uuid

client = QdrantClient(host="qdrant", port=6333)
VECTOR_SIZE = 384  # Set to match your embed() output

def ensure_collection_exists():
    if not client.collection_exists("docs"):
        client.recreate_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

def index_chunks(chunks_with_metadata: List[dict]):
    ensure_collection_exists()  # ✅ Ensure collection exists before upsert

    points = []
    for chunk in chunks_with_metadata:
        vector = embed(chunk["text"])
        payload = chunk["metadata"].copy()
        payload["text"] = chunk["text"]
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload
        ))
    client.upsert(collection_name="docs", points=points)

# Add this function below index_chunks in backend/rag/vector_store.py

from qdrant_client.models import Filter, SearchRequest
from backend.rag.embedder import embed

def search_chunks(query: str, top_k: int = 5):
    vector = embed(query)
    results = client.search(
        collection_name="docs",
        query_vector=vector,
        limit=top_k,
    )
    return [
        {
            "text": hit.payload.get("text", ""),
            "score": hit.score,
            "metadata": hit.payload
        }
        for hit in results
    ]

