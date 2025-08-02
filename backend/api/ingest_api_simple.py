# Enterprise Multi-Modal Ingestion API with Real-Time Progress
# File: backend/api/ingest_api_simple.py

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse
from backend.ingest.multimodal_processor import MultimodalProcessor
from backend.rag.enhanced_vector_store import EnterpriseVectorStore
from typing import List, Dict, Any, Tuple
import asyncio
import json
import uuid
from datetime import datetime
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)
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
    
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Started real-time ingest session {session_id} with {len(files)} files")
    for file in files:
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] File: {file.filename} ({file.size} bytes)")
    
    # Store file contents for background processing
    file_contents = []
    for file in files:
        content = await file.read()
        file_contents.append((file.filename, content))
    
    # Start background processing
    asyncio.create_task(process_files_background(session_id, file_contents))
    
    return {
        "session_id": session_id,
        "message": f"Started processing {len(files)} files",
        "status": "processing",
        "timestamp": datetime.utcnow().isoformat()
    }

async def process_files_background(session_id: str, file_contents: List[Tuple[str, bytes]]):
    """
    Background task to process files with real-time progress updates
    """
    tracker = progress_sessions.get(session_id)
    if not tracker:
        logger.error(f"Session {session_id} not found for background processing")
        return
    
    try:
        total_chunks = 0
        results = []
        
        # Process files with real-time progress updates
        for i, (filename, file_content) in enumerate(file_contents):
            try:
                logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Processing file: {filename} ({len(file_content)} bytes)")
                
                # Update progress for current file
                tracker.update_file_progress(filename, 0)
                await asyncio.sleep(0.5)  # Small delay to make progress visible
                
                # Read file content (already done)
                tracker.update_file_progress(filename, 25)
                await asyncio.sleep(0.5)  # Small delay to make progress visible
                
                # Process with multi-modal processor using enhanced progress tracking
                def progress_callback(progress: int, message: str):
                    tracker.update_file_progress(filename, progress)
                    logger.info(f"Progress for {filename}: {progress}% - {message}")
                
                chunks = multimodal_processor.process_file(
                    file_bytes=file_content,
                    filename=filename,
                    doc_type="uploaded",
                    progress_callback=progress_callback
                )
                tracker.update_file_progress(filename, 75)
                await asyncio.sleep(0.5)  # Small delay to make progress visible
                
                # Index chunks in vector store
                vector_store.index_chunks(chunks)
                tracker.update_file_progress(filename, 90)
                await asyncio.sleep(0.5)  # Small delay to make progress visible
                
                total_chunks += len(chunks)
                tracker.complete_file(filename, len(chunks))
                
                results.append({
                    "file": filename,
                    "chunks": len(chunks),
                    "file_size": len(file_content),
                    "status": "success"
                })
                
                logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Successfully processed {filename}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Error processing {filename}: {str(e)}")
                tracker.status = "error"
                results.append({
                    "file": filename,
                    "chunks": 0,
                    "error": str(e),
                    "status": "error"
                })
        
        # Mark as completed
        tracker.status = "completed"
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Completed background processing for session {session_id} - Total chunks: {total_chunks}")
        
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        tracker.status = "error"

@router.options("/progress/{session_id}")
async def options_progress_stream(session_id: str):
    """Handle preflight OPTIONS request for SSE endpoint"""
    from fastapi.responses import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Cache-Control",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@router.get("/progress/{session_id}")
async def get_progress_stream(session_id: str):
    """
    Server-Sent Events endpoint for real-time progress tracking
    """
    async def event_generator():
        logger.info(f"Starting SSE stream for session {session_id}")
        while True:
            if session_id in progress_sessions:
                tracker = progress_sessions[session_id]
                progress_data = tracker.get_progress_data()
                
                logger.info(f"Progress update for session {session_id}: {progress_data}")
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
            "Access-Control-Allow-Headers": "Cache-Control",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Credentials": "true"
        }
    )

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