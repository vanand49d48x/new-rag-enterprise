# backend/api/search_api.py

from fastapi import APIRouter, Query, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchRequest, PointStruct, ScoredPoint
from backend.rag.embedder import embed
from typing import List

router = APIRouter()

# Connect to Qdrant
client = QdrantClient(host="qdrant", port=6333)

@router.get("")
def search(query: str = Query(..., description="Search query"), top_k: int = 5):
    try:
        query_vector = embed(query)
        response = client.search(
            collection_name="docs",
            query_vector=query_vector,
            limit=top_k
        )
        results = [
            {
                "score": point.score,
                "text": point.payload.get("text", ""),
                "metadata": point.payload
            }
            for point in response
        ]
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
