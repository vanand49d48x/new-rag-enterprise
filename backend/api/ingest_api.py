# Enterprise Multi-Modal Ingestion API
# File: backend/api/ingest_api.py

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from backend.ingest.multimodal_processor import EnterpriseMultimodalProcessor
from backend.rag.enhanced_vector_store import EnterpriseVectorStore
from typing import List, Dict, Any
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize processors
multimodal_processor = EnterpriseMultimodalProcessor()
vector_store = EnterpriseVectorStore()

@router.post("/ingest")
async def ingest_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Enterprise-grade multi-modal document ingestion
    Supports: PDF, DOCX, TXT, CSV, XLSX, HTML, RTF, EML, Images, Audio, Video
    """
    logger.info(f"Received ingest request with {len(files)} files")
    try:
        total_chunks = 0
        results = []
        
        # Process files sequentially for now (can be parallelized)
        for file in files:
            try:
                logger.info(f"Processing file: {file.filename}")
                
                # Read file content
                file_content = await file.read()
                
                # Process with multi-modal processor
                chunks = multimodal_processor.process_file(
                    file_bytes=file_content,
                    filename=file.filename,
                    doc_type="uploaded"
                )
                
                # Index chunks in vector store
                vector_store.index_chunks(chunks)
                
                total_chunks += len(chunks)
                results.append({
                    "file": file.filename,
                    "chunks": len(chunks),
                    "file_size": len(file_content),
                    "status": "success"
                })
                
                logger.info(f"Successfully processed {file.filename}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    "file": file.filename,
                    "chunks": 0,
                    "error": str(e),
                    "status": "error"
                })
        
        return {
            "message": f"Processed {len(files)} files, created {total_chunks} total chunks",
            "details": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/batch")
async def ingest_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Batch ingestion with background processing
    """
    try:
        # Store files temporarily and process in background
        file_data = []
        for file in files:
            content = await file.read()
            file_data.append({
                "filename": file.filename,
                "content": content
            })
        
        # Process in background
        if background_tasks:
            background_tasks.add_task(process_batch_files, file_data)
            return {
                "message": f"Started background processing of {len(files)} files",
                "status": "processing"
            }
        else:
            # Process immediately
            return await process_batch_files(file_data)
            
    except Exception as e:
        logger.error(f"Error in batch ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_batch_files(file_data: List[Dict[str, Any]]):
    """Background task for processing batch files"""
    try:
        total_chunks = 0
        results = []
        
        for file_info in file_data:
            try:
                chunks = multimodal_processor.process_file(
                    file_bytes=file_info["content"],
                    filename=file_info["filename"],
                    doc_type="batch_uploaded"
                )
                
                vector_store.index_chunks(chunks)
                total_chunks += len(chunks)
                
                results.append({
                    "file": file_info["filename"],
                    "chunks": len(chunks),
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Error processing {file_info['filename']}: {str(e)}")
                results.append({
                    "file": file_info["filename"],
                    "error": str(e),
                    "status": "error"
                })
        
        logger.info(f"Batch processing completed: {total_chunks} total chunks")
        return {
            "message": f"Batch processing completed: {total_chunks} total chunks",
            "details": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise

@router.get("/stats")
async def get_ingestion_stats():
    """Get ingestion and collection statistics"""
    try:
        collection_stats = vector_store.get_collection_stats()
        
        return {
            "collection_stats": collection_stats,
            "processor_info": {
                "supports_file_types": [
                    "pdf", "docx", "txt", "csv", "xlsx", "html", "rtf", "eml",
                    "png", "jpg", "jpeg", "bmp", "tiff",
                    "mp3", "wav", "flac", "m4a",
                    "mp4", "mov", "avi", "mkv"
                ],
                "chunking_strategy": "hierarchical",
                "retrieval_method": "hybrid_semantic_bm25"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear")
async def clear_collection():
    """Clear all documents from the collection"""
    try:
        # This would need to be implemented in the vector store
        # For now, return a message
        return {
            "message": "Collection clear functionality not implemented yet",
            "note": "Use Qdrant admin interface to clear collection"
        }
        
    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
