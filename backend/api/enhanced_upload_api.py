import os
import uuid
import mimetypes
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import sqlite3
from dataclasses import dataclass, asdict
import threading
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import qdrant_client
from qdrant_client.models import PointStruct, Distance, VectorParams

# Import processing modules
from ..ingest.processor import DocumentProcessor
from ..ingest.multimodal_processor import MultimodalProcessor
from ..rag.embedder import Embedder
from ..utils.config import get_config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/enhanced-upload", tags=["enhanced-upload"])

@dataclass
class FileMetadata:
    id: str
    filename: str
    file_path: str
    file_type: str
    file_size: int
    upload_time: datetime
    status: str
    processing_steps: List[str]
    chunks_created: int
    vectors_created: int
    error_message: Optional[str] = None

class FileProcessor:
    def __init__(self):
        self.config = get_config()
        self.upload_dir = os.getenv('UPLOAD_DIR', '/app/uploads')
        self.processed_dir = os.getenv('PROCESSED_DIR', '/app/processed')
        self.qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        self.qdrant_port = int(os.getenv('QDRANT_PORT', '6334'))
        
        # Initialize components
        self.qdrant_client = qdrant_client.QdrantClient(
            host=self.qdrant_host, 
            port=self.qdrant_port
        )
        self.embedder = Embedder()
        self.document_processor = DocumentProcessor()
        self.multimodal_processor = MultimodalProcessor()
        
        # Ensure directories exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Processing queue
        self.processing_queue = []
        self.processing_lock = threading.Lock()
        
        # Start background processor
        self.start_background_processor()

    def init_database(self):
        """Initialize SQLite database for file metadata"""
        db_path = os.path.join(self.upload_dir, 'file_metadata.db')
        self.db_path = db_path
        
        with sqlite3.connect(db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    upload_time TEXT NOT NULL,
                    status TEXT NOT NULL,
                    processing_steps TEXT,
                    chunks_created INTEGER DEFAULT 0,
                    vectors_created INTEGER DEFAULT 0,
                    error_message TEXT
                )
            ''')
            conn.commit()

    def start_background_processor(self):
        """Start background thread for processing files"""
        def process_queue():
            while True:
                with self.processing_lock:
                    if self.processing_queue:
                        file_metadata = self.processing_queue.pop(0)
                    else:
                        file_metadata = None
                
                if file_metadata:
                    try:
                        self.process_file(file_metadata)
                    except Exception as e:
                        logger.error(f"Error processing file {file_metadata.filename}: {e}")
                        self.update_file_status(file_metadata.id, "error", error_message=str(e))
                
                time.sleep(1)  # Check queue every second
        
        thread = threading.Thread(target=process_queue, daemon=True)
        thread.start()

    def detect_file_type(self, file_path: str) -> str:
        """Detect file type and return processing route"""
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if not mime_type:
            # Try to detect by file extension
            ext = Path(file_path).suffix.lower()
            if ext in ['.pdf', '.docx', '.doc', '.txt', '.md']:
                return "document"
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                return "image"
            elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                return "audio"
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                return "video"
            else:
                # Try to detect by content for files without extensions
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(512)  # Read first 512 bytes
                        
                    # Check for common file signatures
                    if header.startswith(b'%PDF'):
                        return "document"
                    elif header.startswith(b'\xff\xd8\xff'):  # JPEG
                        return "image"
                    elif header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                        return "image"
                    elif header.startswith(b'GIF8'):  # GIF
                        return "image"
                    elif header.startswith(b'BM'):  # BMP
                        return "image"
                    elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):  # MP3
                        return "audio"
                    elif header.startswith(b'RIFF') and b'AVI ' in header:  # AVI
                        return "video"
                    elif header.startswith(b'\x00\x00\x00') and b'ftyp' in header:  # MP4
                        return "video"
                    else:
                        # Default to document for text-like content
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                f.read(100)  # Try to read as text
                            return "document"
                        except:
                            return "unknown"
                except:
                    return "unknown"
        
        if mime_type.startswith("application/pdf"):
            return "document"
        elif mime_type.startswith("application/vnd.openxmlformats-officedocument"):
            return "document"
        elif mime_type.startswith("text/"):
            return "document"
        elif mime_type.startswith("image/"):
            return "image"
        elif mime_type.startswith("audio/"):
            return "audio"
        elif mime_type.startswith("video/"):
            return "video"
        else:
            return "unknown"

    def route_file(self, file_path: str) -> str:
        """Route file to appropriate processor"""
        file_type = self.detect_file_type(file_path)
        logger.info(f"Routing file {file_path} to {file_type} processor")
        return file_type

    def save_file_metadata(self, metadata: FileMetadata):
        """Save file metadata to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO file_metadata 
                (id, filename, file_path, file_type, file_size, upload_time, status, 
                 processing_steps, chunks_created, vectors_created, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.id, metadata.filename, metadata.file_path, metadata.file_type,
                metadata.file_size, metadata.upload_time.isoformat(), metadata.status,
                json.dumps(metadata.processing_steps), metadata.chunks_created,
                metadata.vectors_created, metadata.error_message
            ))
            conn.commit()

    def update_file_status(self, file_id: str, status: str, error_message: str = None):
        """Update file processing status"""
        with sqlite3.connect(self.db_path) as conn:
            if error_message:
                conn.execute('''
                    UPDATE file_metadata 
                    SET status = ?, error_message = ?
                    WHERE id = ?
                ''', (status, error_message, file_id))
            else:
                conn.execute('''
                    UPDATE file_metadata 
                    SET status = ?
                    WHERE id = ?
                ''', (status, file_id))
            conn.commit()

    def process_file(self, metadata: FileMetadata):
        """Process file based on its type"""
        try:
            self.update_file_status(metadata.id, "processing")
            
            file_type = self.route_file(metadata.file_path)
            metadata.file_type = file_type
            
            if file_type == "document":
                self.process_document(metadata)
            elif file_type == "image":
                self.process_image(metadata)
            elif file_type == "audio":
                self.process_audio(metadata)
            elif file_type == "video":
                self.process_video(metadata)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Copy to processed directory
            processed_path = os.path.join(self.processed_dir, metadata.filename)
            import shutil
            shutil.copy2(metadata.file_path, processed_path)
            os.remove(metadata.file_path)  # Remove original file
            metadata.file_path = processed_path
            
            self.update_file_status(metadata.id, "completed")
            self.save_file_metadata(metadata)
            
        except Exception as e:
            logger.error(f"Error processing file {metadata.filename}: {e}")
            self.update_file_status(metadata.id, "error", str(e))

    def process_document(self, metadata: FileMetadata):
        """Process document files (PDF, DOCX, TXT)"""
        logger.info(f"Processing document: {metadata.filename}")
        
        # Extract text
        metadata.processing_steps.append("text_extraction")
        text = self.document_processor.extract_text(metadata.file_path)
        
        # Chunk text
        metadata.processing_steps.append("chunking")
        chunks = self.document_processor.chunk_text(text, chunk_size=500)
        metadata.chunks_created = len(chunks)
        
        # Clean chunks in parallel for better CPU performance
        if len(chunks) > 1000:
            logger.info(f"Large file detected ({len(chunks)} chunks), using parallel cleaning")
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                cleaned_chunks = list(executor.map(clean_chunk_parallel, chunks))
            # Filter out empty chunks
            chunks = [chunk for chunk in cleaned_chunks if chunk]
            logger.info(f"Cleaned chunks: {len(chunks)} valid chunks remaining")
        
        # Create embeddings with batching for large files
        metadata.processing_steps.append("embedding")
        if len(chunks) > 1000:  # Use batching for large files
            logger.info(f"Large file detected ({len(chunks)} chunks), using batch processing")
            embeddings = self.embedder.embed_texts_batch(chunks, batch_size=1000)
        else:
            embeddings = self.embedder.embed_texts(chunks)
        metadata.vectors_created = len(embeddings)
        
        # Store in Qdrant
        metadata.processing_steps.append("vector_storage")
        self.store_vectors_in_qdrant(chunks, embeddings, metadata.filename)

    def process_image(self, metadata: FileMetadata):
        """Process image files"""
        logger.info(f"Processing image: {metadata.filename}")
        
        # Extract image features
        metadata.processing_steps.append("image_analysis")
        image_features = self.multimodal_processor.process_image(metadata.file_path)
        
        # Generate caption if possible
        metadata.processing_steps.append("caption_generation")
        caption = self.multimodal_processor.generate_caption(metadata.file_path)
        
        # Create text embedding for caption
        metadata.processing_steps.append("embedding")
        caption_embedding = self.embedder.embed_texts([caption])[0]
        
        # Store in Qdrant
        metadata.processing_steps.append("vector_storage")
        self.store_vectors_in_qdrant([caption], [caption_embedding], metadata.filename)
        
        metadata.chunks_created = 1
        metadata.vectors_created = 1

    def process_audio(self, metadata: FileMetadata):
        """Process audio files"""
        logger.info(f"Processing audio: {metadata.filename}")
        
        # Transcribe audio
        metadata.processing_steps.append("transcription")
        transcript = self.multimodal_processor.transcribe_audio(metadata.file_path)
        
        # Chunk transcript
        metadata.processing_steps.append("chunking")
        chunks = self.document_processor.chunk_text(transcript, chunk_size=500)
        metadata.chunks_created = len(chunks)
        
        # Create embeddings with batching for large files
        metadata.processing_steps.append("embedding")
        if len(chunks) > 1000:  # Use batching for large files
            logger.info(f"Large audio file detected ({len(chunks)} chunks), using batch processing")
            embeddings = self.embedder.embed_texts_batch(chunks, batch_size=1000)
        else:
            embeddings = self.embedder.embed_texts(chunks)
        metadata.vectors_created = len(embeddings)
        
        # Store in Qdrant
        metadata.processing_steps.append("vector_storage")
        self.store_vectors_in_qdrant(chunks, embeddings, metadata.filename)

    def process_video(self, metadata: FileMetadata):
        """Process video files"""
        logger.info(f"Processing video: {metadata.filename}")
        
        # Extract audio from video
        metadata.processing_steps.append("audio_extraction")
        audio_path = self.multimodal_processor.extract_audio_from_video(metadata.file_path)
        
        # Transcribe audio
        metadata.processing_steps.append("transcription")
        transcript = self.multimodal_processor.transcribe_audio(audio_path)
        
        # Clean up temporary audio file
        os.remove(audio_path)
        
        # Chunk transcript
        metadata.processing_steps.append("chunking")
        chunks = self.document_processor.chunk_text(transcript, chunk_size=500)
        metadata.chunks_created = len(chunks)
        
        # Create embeddings with batching for large files
        metadata.processing_steps.append("embedding")
        if len(chunks) > 1000:  # Use batching for large files
            logger.info(f"Large video file detected ({len(chunks)} chunks), using batch processing")
            embeddings = self.embedder.embed_texts_batch(chunks, batch_size=1000)
        else:
            embeddings = self.embedder.embed_texts(chunks)
        metadata.vectors_created = len(embeddings)
        
        # Store in Qdrant
        metadata.processing_steps.append("vector_storage")
        self.store_vectors_in_qdrant(chunks, embeddings, metadata.filename)

    def store_vectors_in_qdrant(self, chunks: List[str], embeddings: List[List[float]], source: str):
        """Store vectors in Qdrant"""
        collection_name = "rag_chunks"
        
        # Ensure collection exists
        try:
            self.qdrant_client.get_collection(collection_name)
        except Exception as e:
            # Collection doesn't exist, create it
            try:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
                )
            except Exception as create_error:
                # If creation fails, check if it's because collection already exists
                if "already exists" in str(create_error):
                    # Collection already exists, that's fine
                    pass
                else:
                    # Some other error, raise it
                    raise create_error
        
        # Create points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": source,
                    "chunk_index": i,
                    "timestamp": datetime.now().isoformat()
                }
            )
            points.append(point)
        
        # Upsert points
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

def clean_chunk_parallel(chunk: str) -> str:
    """Clean a single chunk in parallel"""
    # Basic cleaning operations
    chunk = chunk.strip()
    if len(chunk) < 10:  # Skip very short chunks
        return ""
    return chunk

class EnhancedUploadAPI:
    def __init__(self):
        self.config = get_config()
        self.upload_dir = os.getenv('UPLOAD_DIR', '/app/uploads')
        self.processed_dir = os.getenv('PROCESSED_DIR', '/app/processed')
        self.qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        self.qdrant_port = int(os.getenv('QDRANT_PORT', '6334'))
        
        # Initialize components
        self.qdrant_client = qdrant_client.QdrantClient(
            host=self.qdrant_host, 
            port=self.qdrant_port
        )
        self.embedder = Embedder()
        self.document_processor = DocumentProcessor()
        self.multimodal_processor = MultimodalProcessor()
        
        # Ensure directories exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Processing queue
        self.processing_queue = []
        self.processing_lock = threading.Lock()
        
        # Start background processor
        self.start_background_processor()

    def init_database(self):
        """Initialize SQLite database for file metadata"""
        db_path = os.path.join(self.upload_dir, 'file_metadata.db')
        self.db_path = db_path
        
        with sqlite3.connect(db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    upload_time TEXT NOT NULL,
                    status TEXT NOT NULL,
                    processing_steps TEXT,
                    chunks_created INTEGER DEFAULT 0,
                    vectors_created INTEGER DEFAULT 0,
                    error_message TEXT
                )
            ''')
            conn.commit()

    def start_background_processor(self):
        """Start background thread for processing files"""
        def process_queue():
            while True:
                with self.processing_lock:
                    if self.processing_queue:
                        file_metadata = self.processing_queue.pop(0)
                    else:
                        file_metadata = None
                
                if file_metadata:
                    try:
                        self.process_file(file_metadata)
                    except Exception as e:
                        logger.error(f"Error processing file {file_metadata.filename}: {e}")
                        self.update_file_status(file_metadata.id, "error", error_message=str(e))
                
                time.sleep(1)  # Check queue every second
        
        thread = threading.Thread(target=process_queue, daemon=True)
        thread.start()

    def detect_file_type(self, file_path: str) -> str:
        """Detect file type and return processing route"""
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if not mime_type:
            # Try to detect by file extension
            ext = Path(file_path).suffix.lower()
            if ext in ['.pdf', '.docx', '.doc', '.txt', '.md']:
                return "document"
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                return "image"
            elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                return "audio"
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                return "video"
            else:
                # Try to detect by content for files without extensions
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(512)  # Read first 512 bytes
                        
                    # Check for common file signatures
                    if header.startswith(b'%PDF'):
                        return "document"
                    elif header.startswith(b'\xff\xd8\xff'):  # JPEG
                        return "image"
                    elif header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                        return "image"
                    elif header.startswith(b'GIF8'):  # GIF
                        return "image"
                    elif header.startswith(b'BM'):  # BMP
                        return "image"
                    elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):  # MP3
                        return "audio"
                    elif header.startswith(b'RIFF') and b'AVI ' in header:  # AVI
                        return "video"
                    elif header.startswith(b'\x00\x00\x00') and b'ftyp' in header:  # MP4
                        return "video"
                    else:
                        # Default to document for text-like content
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                f.read(100)  # Try to read as text
                            return "document"
                        except:
                            return "unknown"
                except:
                    return "unknown"
        
        if mime_type.startswith("application/pdf"):
            return "document"
        elif mime_type.startswith("application/vnd.openxmlformats-officedocument"):
            return "document"
        elif mime_type.startswith("text/"):
            return "document"
        elif mime_type.startswith("image/"):
            return "image"
        elif mime_type.startswith("audio/"):
            return "audio"
        elif mime_type.startswith("video/"):
            return "video"
        else:
            return "unknown"

    def route_file(self, file_path: str) -> str:
        """Route file to appropriate processor"""
        file_type = self.detect_file_type(file_path)
        logger.info(f"Routing file {file_path} to {file_type} processor")
        return file_type

    def save_file_metadata(self, metadata: FileMetadata):
        """Save file metadata to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO file_metadata 
                (id, filename, file_path, file_type, file_size, upload_time, status, 
                 processing_steps, chunks_created, vectors_created, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.id, metadata.filename, metadata.file_path, metadata.file_type,
                metadata.file_size, metadata.upload_time.isoformat(), metadata.status,
                json.dumps(metadata.processing_steps), metadata.chunks_created,
                metadata.vectors_created, metadata.error_message
            ))
            conn.commit()

    def update_file_status(self, file_id: str, status: str, error_message: str = None):
        """Update file processing status"""
        with sqlite3.connect(self.db_path) as conn:
            if error_message:
                conn.execute('''
                    UPDATE file_metadata 
                    SET status = ?, error_message = ?
                    WHERE id = ?
                ''', (status, error_message, file_id))
            else:
                conn.execute('''
                    UPDATE file_metadata 
                    SET status = ?
                    WHERE id = ?
                ''', (status, file_id))
            conn.commit()

    def process_file(self, metadata: FileMetadata):
        """Process file based on its type"""
        try:
            self.update_file_status(metadata.id, "processing")
            
            file_type = self.route_file(metadata.file_path)
            metadata.file_type = file_type
            
            if file_type == "document":
                self.process_document(metadata)
            elif file_type == "image":
                self.process_image(metadata)
            elif file_type == "audio":
                self.process_audio(metadata)
            elif file_type == "video":
                self.process_video(metadata)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Copy to processed directory
            processed_path = os.path.join(self.processed_dir, metadata.filename)
            import shutil
            shutil.copy2(metadata.file_path, processed_path)
            os.remove(metadata.file_path)  # Remove original file
            metadata.file_path = processed_path
            
            self.update_file_status(metadata.id, "completed")
            self.save_file_metadata(metadata)
            
        except Exception as e:
            logger.error(f"Error processing file {metadata.filename}: {e}")
            self.update_file_status(metadata.id, "error", str(e))

    def process_document(self, metadata: FileMetadata):
        """Process document files (PDF, DOCX, TXT)"""
        logger.info(f"Processing document: {metadata.filename}")
        
        # Extract text
        metadata.processing_steps.append("text_extraction")
        text = self.document_processor.extract_text(metadata.file_path)
        
        # Chunk text
        metadata.processing_steps.append("chunking")
        chunks = self.document_processor.chunk_text(text, chunk_size=500)
        metadata.chunks_created = len(chunks)
        
        # Clean chunks in parallel for better CPU performance
        if len(chunks) > 1000:
            logger.info(f"Large file detected ({len(chunks)} chunks), using parallel cleaning")
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                cleaned_chunks = list(executor.map(clean_chunk_parallel, chunks))
            # Filter out empty chunks
            chunks = [chunk for chunk in cleaned_chunks if chunk]
            logger.info(f"Cleaned chunks: {len(chunks)} valid chunks remaining")
        
        # Create embeddings with batching for large files
        metadata.processing_steps.append("embedding")
        if len(chunks) > 1000:  # Use batching for large files
            logger.info(f"Large file detected ({len(chunks)} chunks), using batch processing")
            embeddings = self.embedder.embed_texts_batch(chunks, batch_size=1000)
        else:
            embeddings = self.embedder.embed_texts(chunks)
        metadata.vectors_created = len(embeddings)
        
        # Store in Qdrant
        metadata.processing_steps.append("vector_storage")
        self.store_vectors_in_qdrant(chunks, embeddings, metadata.filename)

    def process_image(self, metadata: FileMetadata):
        """Process image files"""
        logger.info(f"Processing image: {metadata.filename}")
        
        # Extract image features
        metadata.processing_steps.append("image_analysis")
        image_features = self.multimodal_processor.process_image(metadata.file_path)
        
        # Generate caption if possible
        metadata.processing_steps.append("caption_generation")
        caption = self.multimodal_processor.generate_caption(metadata.file_path)
        
        # Create text embedding for caption
        metadata.processing_steps.append("embedding")
        caption_embedding = self.embedder.embed_texts([caption])[0]
        
        # Store in Qdrant
        metadata.processing_steps.append("vector_storage")
        self.store_vectors_in_qdrant([caption], [caption_embedding], metadata.filename)
        
        metadata.chunks_created = 1
        metadata.vectors_created = 1

    def process_audio(self, metadata: FileMetadata):
        """Process audio files"""
        logger.info(f"Processing audio: {metadata.filename}")
        
        # Transcribe audio
        metadata.processing_steps.append("transcription")
        transcript = self.multimodal_processor.transcribe_audio(metadata.file_path)
        
        # Chunk transcript
        metadata.processing_steps.append("chunking")
        chunks = self.document_processor.chunk_text(transcript, chunk_size=500)
        metadata.chunks_created = len(chunks)
        
        # Create embeddings with batching for large files
        metadata.processing_steps.append("embedding")
        if len(chunks) > 1000:  # Use batching for large files
            logger.info(f"Large audio file detected ({len(chunks)} chunks), using batch processing")
            embeddings = self.embedder.embed_texts_batch(chunks, batch_size=1000)
        else:
            embeddings = self.embedder.embed_texts(chunks)
        metadata.vectors_created = len(embeddings)
        
        # Store in Qdrant
        metadata.processing_steps.append("vector_storage")
        self.store_vectors_in_qdrant(chunks, embeddings, metadata.filename)

    def process_video(self, metadata: FileMetadata):
        """Process video files"""
        logger.info(f"Processing video: {metadata.filename}")
        
        # Extract audio from video
        metadata.processing_steps.append("audio_extraction")
        audio_path = self.multimodal_processor.extract_audio_from_video(metadata.file_path)
        
        # Transcribe audio
        metadata.processing_steps.append("transcription")
        transcript = self.multimodal_processor.transcribe_audio(audio_path)
        
        # Clean up temporary audio file
        os.remove(audio_path)
        
        # Chunk transcript
        metadata.processing_steps.append("chunking")
        chunks = self.document_processor.chunk_text(transcript, chunk_size=500)
        metadata.chunks_created = len(chunks)
        
        # Create embeddings with batching for large files
        metadata.processing_steps.append("embedding")
        if len(chunks) > 1000:  # Use batching for large files
            logger.info(f"Large video file detected ({len(chunks)} chunks), using batch processing")
            embeddings = self.embedder.embed_texts_batch(chunks, batch_size=1000)
        else:
            embeddings = self.embedder.embed_texts(chunks)
        metadata.vectors_created = len(embeddings)
        
        # Store in Qdrant
        metadata.processing_steps.append("vector_storage")
        self.store_vectors_in_qdrant(chunks, embeddings, metadata.filename)

    def store_vectors_in_qdrant(self, chunks: List[str], embeddings: List[List[float]], source: str):
        """Store vectors in Qdrant"""
        collection_name = "rag_chunks"
        
        # Ensure collection exists
        try:
            self.qdrant_client.get_collection(collection_name)
        except Exception as e:
            # Collection doesn't exist, create it
            try:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
                )
            except Exception as create_error:
                # If creation fails, check if it's because collection already exists
                if "already exists" in str(create_error):
                    # Collection already exists, that's fine
                    pass
                else:
                    # Some other error, raise it
                    raise create_error
        
        # Create points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": source,
                    "chunk_index": i,
                    "timestamp": datetime.now().isoformat()
                }
            )
            points.append(point)
        
        # Upsert points
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

# Initialize processor
file_processor = FileProcessor()

class UploadResponse(BaseModel):
    message: str
    file_id: str
    status: str
    file_type: str

class ProcessingStatus(BaseModel):
    file_id: str
    status: str
    processing_steps: List[str]
    chunks_created: int
    vectors_created: int
    error_message: Optional[str] = None

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing"""
    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Save file to upload directory
        file_path = os.path.join(file_processor.upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create metadata
        metadata = FileMetadata(
            id=file_id,
            filename=file.filename,
            file_path=file_path,
            file_type="unknown",
            file_size=len(content),
            upload_time=datetime.now(),
            status="uploaded",
            processing_steps=[],
            chunks_created=0,
            vectors_created=0
        )
        
        # Save metadata
        file_processor.save_file_metadata(metadata)
        
        # Add to processing queue
        with file_processor.processing_lock:
            file_processor.processing_queue.append(metadata)
        
        return UploadResponse(
            message=f"File {file.filename} uploaded successfully",
            file_id=file_id,
            status="uploaded",
            file_type="unknown"
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{file_id}", response_model=ProcessingStatus)
async def get_processing_status(file_id: str):
    """Get processing status for a file"""
    try:
        with sqlite3.connect(file_processor.db_path) as conn:
            cursor = conn.execute('''
                SELECT status, processing_steps, chunks_created, vectors_created, error_message
                FROM file_metadata WHERE id = ?
            ''', (file_id,))
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="File not found")
            
            status, processing_steps_json, chunks_created, vectors_created, error_message = row
            processing_steps = json.loads(processing_steps_json) if processing_steps_json else []
            
            return ProcessingStatus(
                file_id=file_id,
                status=status,
                processing_steps=processing_steps,
                chunks_created=chunks_created,
                vectors_created=vectors_created,
                error_message=error_message
            )
            
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files")
async def list_files():
    """List all uploaded files with their status"""
    try:
        with sqlite3.connect(file_processor.db_path) as conn:
            cursor = conn.execute('''
                SELECT id, filename, file_type, file_size, upload_time, status, 
                       chunks_created, vectors_created
                FROM file_metadata
                ORDER BY upload_time DESC
            ''')
            rows = cursor.fetchall()
            
            files = []
            for row in rows:
                file_id, filename, file_type, file_size, upload_time, status, chunks_created, vectors_created = row
                files.append({
                    "id": file_id,
                    "filename": filename,
                    "file_type": file_type,
                    "file_size": file_size,
                    "upload_time": upload_time,
                    "status": status,
                    "chunks_created": chunks_created,
                    "vectors_created": vectors_created
                })
            
            return {"files": files}
            
    except Exception as e:
        logger.error(f"List files error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file and its associated chunks and vectors"""
    try:
        logger.info(f"Deleting file with ID: {file_id}")
        
        # Get file metadata from database
        with sqlite3.connect(file_processor.db_path) as conn:
            cursor = conn.execute('''
                SELECT filename, file_path, file_type, status 
                FROM file_metadata WHERE id = ?
            ''', (file_id,))
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="File not found")
            
            filename, file_path, file_type, status = row
            
            # Check if file is still processing
            if status == "processing":
                raise HTTPException(
                    status_code=400, 
                    detail="Cannot delete file while it is being processed"
                )
            
            logger.info(f"Deleting file: {filename} (Type: {file_type}, Status: {status})")
            
            # Step 1: Delete file from disk
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted file from disk: {file_path}")
                else:
                    logger.warning(f"File not found on disk: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file from disk: {e}")
                # Continue with other cleanup steps
            
            # Step 2: Delete vectors from Qdrant
            try:
                delete_vectors_by_file_id(file_id)
                logger.info(f"Deleted vectors from Qdrant for file: {file_id}")
            except Exception as e:
                logger.error(f"Error deleting vectors from Qdrant: {e}")
                # Continue with metadata cleanup
            
            # Step 3: Delete metadata from database
            try:
                conn.execute('DELETE FROM file_metadata WHERE id = ?', (file_id,))
                conn.commit()
                logger.info(f"Deleted metadata from database for file: {file_id}")
            except Exception as e:
                logger.error(f"Error deleting metadata from database: {e}")
                raise HTTPException(status_code=500, detail="Failed to delete file metadata")
            
            return {
                "message": f"File {filename} deleted successfully",
                "file_id": file_id,
                "filename": filename,
                "deleted_components": ["file", "vectors", "metadata"]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def delete_vectors_by_file_id(file_id: str):
    """Delete all vectors associated with a file from Qdrant"""
    try:
        collection_name = "rag_chunks"
        
        # First, check if collection exists
        try:
            file_processor.qdrant_client.get_collection(collection_name)
        except:
            logger.warning(f"Collection {collection_name} does not exist")
            return
        
        # Delete points by file_id filter
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        file_processor.qdrant_client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=file_id))
                ]
            )
        )
        
        logger.info(f"Successfully deleted vectors for file_id: {file_id}")
        
    except Exception as e:
        logger.error(f"Error deleting vectors for file_id {file_id}: {e}")
        raise

@router.delete("/files/batch")
async def delete_multiple_files(file_ids: List[str]):
    """Delete multiple files and their associated chunks and vectors"""
    try:
        logger.info(f"Deleting multiple files: {file_ids}")
        
        results = []
        errors = []
        
        for file_id in file_ids:
            try:
                result = await delete_file(file_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error deleting file {file_id}: {e}")
                errors.append({"file_id": file_id, "error": str(e)})
        
        return {
            "message": f"Batch deletion completed",
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Error in batch deletion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{file_id}/deletion-preview")
async def get_deletion_preview(file_id: str):
    """Get preview of what will be deleted for a file"""
    try:
        with sqlite3.connect(file_processor.db_path) as conn:
            cursor = conn.execute('''
                SELECT filename, file_path, file_type, status, file_size, chunks_created, vectors_created
                FROM file_metadata WHERE id = ?
            ''', (file_id,))
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="File not found")
            
            filename, file_path, file_type, status, file_size, chunks_created, vectors_created = row
            
            # Check if file exists on disk
            file_exists = os.path.exists(file_path)
            
            # Count vectors in Qdrant (approximate)
            vector_count = 0
            try:
                collection_name = "rag_chunks"
                file_processor.qdrant_client.get_collection(collection_name)
                
                # This is an approximation - in production you might want to count more precisely
                vector_count = vectors_created or 0
            except:
                vector_count = 0
            
            return {
                "file_id": file_id,
                "filename": filename,
                "file_type": file_type,
                "status": status,
                "file_size": file_size,
                "deletion_preview": {
                    "file_on_disk": file_exists,
                    "file_size_bytes": file_size,
                    "chunks_created": chunks_created or 0,
                    "vectors_in_qdrant": vector_count,
                    "metadata_in_db": True
                },
                "can_delete": status != "processing"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deletion preview for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# TUS hooks endpoint for integration with tusd
@router.post("/tus-hooks")
async def tus_hooks(request: Request):
    """Handle TUS upload completion hooks"""
    try:
        # Get the request body
        body = await request.body()
        data = json.loads(body)
        
        logger.info(f"TUS hook received: {data}")
        
        # Check if this is a post-finish hook
        if data.get("Type") == "post-finish":
            upload_info = data.get("Event", {}).get("Upload", {})
            file_id = upload_info.get("ID")
            
            logger.info(f"Processing post-finish hook for file_id: {file_id}")
            
            if not file_id:
                logger.error("TUS hook received without file ID")
                return {"status": "error", "message": "No file ID provided"}
            
            # Try multiple possible file paths
            possible_paths = [
                os.path.join("/app/uploads", file_id),
                os.path.join(file_processor.upload_dir, file_id),
                os.path.join("./data/uploads", file_id),
                os.path.join("/data", file_id)
            ]
            
            file_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    file_path = path
                    logger.info(f"Found file at: {path}")
                    break
            
            if not file_path:
                logger.error(f"File {file_id} not found in any expected location")
                return {"status": "error", "message": f"File {file_id} not found"}
            
            try:
                # Get file metadata from file system (since TUS might not send complete data)
                file_size = os.path.getsize(file_path)
                file_type = file_processor.detect_file_type(file_path)
                
                # Get original filename from metadata if available, otherwise use file_id
                original_filename = f"uploaded_file_{file_id[:8]}"
                if upload_info.get("MetaData", {}).get("filename"):
                    try:
                        import base64
                        filename_encoded = upload_info["MetaData"]["filename"]
                        original_filename = base64.b64decode(filename_encoded).decode('utf-8')
                    except:
                        pass
                
                # Also check if there's a .info file with metadata
                info_file_path = f"{file_path}.info"
                if os.path.exists(info_file_path):
                    try:
                        with open(info_file_path, 'r') as f:
                            info_data = json.loads(f.read())
                            if info_data.get("MetaData", {}).get("filename"):
                                try:
                                    import base64
                                    filename_encoded = info_data["MetaData"]["filename"]
                                    original_filename = base64.b64decode(filename_encoded).decode('utf-8')
                                except:
                                    pass
                    except:
                        pass
                
                metadata = FileMetadata(
                    id=file_id,
                    filename=original_filename,
                    file_path=file_path,
                    file_type=file_type,
                    file_size=file_size,
                    upload_time=datetime.now(),
                    status="uploaded",
                    processing_steps=[],
                    chunks_created=0,
                    vectors_created=0
                )
                
                # Save metadata and queue for processing
                file_processor.save_file_metadata(metadata)
                file_processor.processing_queue.append(metadata)
                
                logger.info(f"File {file_id} ({original_filename}) queued for processing")
                
            except Exception as e:
                logger.error(f"Error processing file {file_id}: {e}")
                return {"status": "error", "message": str(e)}
                
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Error in TUS hook: {e}")
        return {"status": "error", "message": str(e)} 