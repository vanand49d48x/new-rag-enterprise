# Enterprise Multi-Modal Ingestion API
# File: backend/api/ingest_api.py

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from backend.ingest.multimodal_processor import MultimodalProcessor
from backend.rag.enhanced_vector_store import EnterpriseVectorStore
from typing import List, Dict, Any
import logging
import asyncio
import json
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize processors
multimodal_processor = MultimodalProcessor()
vector_store = EnterpriseVectorStore()

# Progress tracking
progress_sessions = {}

class ProgressTracker:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.total_files = 0
        self.processed_files = 0
        self.current_file = ""
        self.current_file_progress = 0
        self.total_chunks = 0
        self.status = "starting"
        self.start_time = datetime.utcnow()
        
    def update_file_progress(self, filename: str, progress: int):
        self.current_file = filename
        self.current_file_progress = progress
        
    def complete_file(self, filename: str, chunks: int):
        self.processed_files += 1
        self.total_chunks += chunks
        self.current_file_progress = 100
        
    def get_progress_data(self) -> Dict[str, Any]:
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        return {
            "session_id": self.session_id,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "current_file": self.current_file,
            "current_file_progress": self.current_file_progress,
            "total_chunks": self.total_chunks,
            "status": self.status,
            "elapsed_time": elapsed,
            "timestamp": datetime.utcnow().isoformat()
        }

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

@router.post("/ingest/realtime")
async def ingest_files_realtime(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Real-time document ingestion with progress tracking
    """
    # Create progress session
    session_id = str(uuid.uuid4())
    tracker = ProgressTracker(session_id)
    tracker.total_files = len(files)
    tracker.status = "processing"
    progress_sessions[session_id] = tracker
    
    logger.info(f"Started real-time ingest session {session_id} with {len(files)} files")
    
    try:
        total_chunks = 0
        results = []
        
        # Process files with real-time progress updates
        for i, file in enumerate(files):
            try:
                logger.info(f"Processing file: {file.filename}")
                
                # Update progress for current file
                tracker.update_file_progress(file.filename, 0)
                
                # Read file content
                file_content = await file.read()
                tracker.update_file_progress(file.filename, 25)
                
                # Process with multi-modal processor
                chunks = multimodal_processor.process_file(
                    file_bytes=file_content,
                    filename=file.filename,
                    doc_type="uploaded"
                )
                tracker.update_file_progress(file.filename, 75)
                
                # Index chunks in vector store
                vector_store.index_chunks(chunks)
                tracker.update_file_progress(file.filename, 90)
                
                total_chunks += len(chunks)
                tracker.complete_file(file.filename, len(chunks))
                
                results.append({
                    "file": file.filename,
                    "chunks": len(chunks),
                    "file_size": len(file_content),
                    "status": "success"
                })
                
                logger.info(f"Successfully processed {file.filename}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                tracker.status = "error"
                results.append({
                    "file": file.filename,
                    "chunks": 0,
                    "error": str(e),
                    "status": "error"
                })
        
        # Mark as completed
        tracker.status = "completed"
        
        return {
            "session_id": session_id,
            "message": f"Processed {len(files)} files, created {total_chunks} total chunks",
            "details": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in real-time ingestion: {str(e)}")
        tracker.status = "error"
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/progress/{session_id}")
async def get_progress_stream(session_id: str):
    """
    Server-Sent Events endpoint for real-time progress tracking
    """
    async def event_generator():
        while True:
            if session_id in progress_sessions:
                tracker = progress_sessions[session_id]
                progress_data = tracker.get_progress_data()
                
                # Send progress data
                yield f"data: {json.dumps(progress_data)}\n\n"
                
                # If processing is complete, send final update and close
                if tracker.status == "completed" or tracker.status == "error":
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    break
            else:
                # Session not found
                yield f"data: {json.dumps({'error': 'Session not found', 'session_id': session_id})}\n\n"
                break
                
            await asyncio.sleep(1)  # Update every second
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

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
                    "pdf", "docx", "txt", "csv", "xlsx", "html", "rtf", "eml", "json",
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

@router.post("/ingest/folder")
async def ingest_folder(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process a folder of files recursively
    """
    logger.info(f"Received folder ingest request with {len(files)} files")
    try:
        total_chunks = 0
        results = []
        
        # Sort files by path for better organization
        sorted_files = sorted(files, key=lambda x: x.filename)
        
        for file in sorted_files:
            try:
                logger.info(f"Processing folder file: {file.filename}")
                
                # Read file content
                file_content = await file.read()
                
                # Process with multi-modal processor
                chunks = multimodal_processor.process_file(
                    file_bytes=file_content,
                    filename=file.filename,
                    doc_type="folder_uploaded"
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
                
                logger.info(f"Successfully processed folder file {file.filename}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing folder file {file.filename}: {str(e)}")
                results.append({
                    "file": file.filename,
                    "chunks": 0,
                    "error": str(e),
                    "status": "error"
                })
        
        return {
            "message": f"Processed {len(files)} files from folder, created {total_chunks} total chunks",
            "details": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in folder ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
