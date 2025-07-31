# Enterprise RAG Chat API
# File: backend/api/chat_api.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import asyncio
import logging
from backend.rag.enhanced_vector_store import EnterpriseVectorStore
from backend.rag.llm import generate_answer, generate_answer_stream


logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize vector store
vector_store = EnterpriseVectorStore()

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    history: Optional[List[dict]] = []
    filters: Optional[Dict[str, Any]] = None
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class StreamingChatResponse(BaseModel):
    type: str  # "token", "source", "done"
    content: str
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Enterprise-grade chat endpoint with intelligent retrieval
    """
    try:
        logger.info(f"Processing chat request: {request.query[:50]}...")
        
        # Perform enterprise search (enhanced hybrid search)
        search_results = vector_store.enterprise_search(
            query=request.query,
            top_k=request.top_k,
            semantic_weight=request.semantic_weight,
            bm25_weight=request.bm25_weight,
            filters=request.filters
        )
        
        if not search_results:
            return ChatResponse(
                answer="I couldn't find any relevant information in the available documents. Please try rephrasing your question or upload more documents.",
                sources=[],
                metadata={"retrieved_chunks": 0, "search_method": "enterprise"}
            )
        
        # Prepare context from search results
        context = _prepare_context(search_results)
        sources = _prepare_sources(search_results)
        
        # Generate answer using LLM
        answer = generate_answer(
            query=request.query,
            context=context,
            history=request.history
        )
        
        logger.info(f"Generated answer with {len(search_results)} sources")
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            metadata={
                "retrieved_chunks": len(search_results),
                "query_length": len(request.query),
                "context_length": len(context),
                "answer_length": len(answer),
                "search_method": "enterprise",
                "semantic_weight": request.semantic_weight,
                "bm25_weight": request.bm25_weight
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint with enterprise search
    """
    try:
        logger.info(f"Processing streaming chat request: {request.query[:50]}...")
        
        # Perform enterprise search
        search_results = vector_store.enterprise_search(
            query=request.query,
            top_k=request.top_k,
            semantic_weight=request.semantic_weight,
            bm25_weight=request.bm25_weight,
            filters=request.filters
        )
        
        if not search_results:
            # Return immediate response for no results
            return StreamingResponse(
                _stream_no_results(),
                media_type="text/plain"
            )
        
        # Prepare context and sources
        context = _prepare_context(search_results)
        sources = _prepare_sources(search_results)
        
        # Stream the response
        return StreamingResponse(
            _stream_response(request.query, context, sources, search_results),
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _stream_no_results():
    """Stream response when no results are found"""
    yield "data: " + json.dumps({
        "type": "token",
        "content": "I couldn't find any relevant information in the available documents. Please try rephrasing your question or upload more documents."
    }) + "\n\n"
    yield "data: " + json.dumps({"type": "done"}) + "\n\n"

async def _stream_response(query: str, context: str, sources: List[Dict], search_results: List[Dict]):
    """Stream the LLM response with sources"""
    
    # First, send sources information
    yield "data: " + json.dumps({
        "type": "sources",
        "sources": sources,
        "metadata": {
            "retrieved_chunks": len(search_results),
            "search_method": "hybrid"
        }
    }) + "\n\n"
    
    # Then stream the LLM response
    async for token in generate_answer_stream(query, context):
        yield "data: " + json.dumps({
            "type": "token",
            "content": token
        }) + "\n\n"
    
    # Send completion signal
    yield "data: " + json.dumps({"type": "done"}) + "\n\n"

@router.post("/search")
async def search_documents(request: ChatRequest):
    """
    Search documents without generating LLM response
    """
    try:
        search_results = vector_store.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            semantic_weight=request.semantic_weight,
            bm25_weight=request.bm25_weight,
            filters=request.filters
        )
        
        sources = _prepare_sources(search_results)
        
        return {
            "query": request.query,
            "sources": sources,
            "total_results": len(search_results),
            "search_method": "hybrid"
        }
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _prepare_context(search_results: List[Dict]) -> str:
    """Prepare context from search results"""
    context_parts = []
    
    for i, result in enumerate(search_results, 1):
        source_info = f"Document {i}: {result.get('metadata', {}).get('title', 'Unknown')}"
        content = result.get('text', '')
        context_parts.append(f"{source_info}\nContent: {content}\n")
    
    return "\n".join(context_parts)

def _prepare_sources(search_results: List[Dict]) -> List[Dict[str, Any]]:
    """Prepare sources information for response"""
    sources = []
    
    for result in search_results:
        metadata = result.get('metadata', {})
        sources.append({
            "id": result.get('id'),
            "title": metadata.get('title', 'Unknown'),
            "file_type": metadata.get('file_type', 'unknown'),
            "chunk_type": metadata.get('chunk_type', 'unknown'),
            "chunk_level": metadata.get('chunk_level', 'unknown'),
            "score": result.get('score', 0.0),
            "text_preview": result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', ''),
            "created": metadata.get('created', '')
        })
    
    return sources
