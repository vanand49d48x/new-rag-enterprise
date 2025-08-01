import os
import json
import sqlite3
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psutil
import qdrant_client
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

class SystemStatus(BaseModel):
    status: str
    services: Dict[str, str]
    resources: Dict[str, Any]
    uptime: str
    tier: Optional[str] = None
    model: Optional[str] = None

class FileInfo(BaseModel):
    id: str
    name: str
    type: str
    size: int
    status: str
    upload_time: str
    processing_steps: List[str]
    chunks_created: int
    vectors_created: int
    error_message: Optional[str] = None

class DashboardStats(BaseModel):
    total_files: int
    completed_files: int
    processing_files: int
    error_files: int
    total_chunks: int
    total_vectors: int
    storage_used: int
    system_health: str
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    available_models: Optional[int] = None
    total_size: Optional[str] = None

class DashboardAPI:
    def __init__(self):
        self.upload_dir = os.getenv('UPLOAD_DIR', '/app/uploads')
        self.processed_dir = os.getenv('PROCESSED_DIR', '/app/processed')
        self.qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        self.qdrant_port = int(os.getenv('QDRANT_PORT', '6334'))
        
        # Initialize database connection
        self.db_path = os.path.join(self.upload_dir, 'file_metadata.db')
        
        # Initialize Qdrant client
        try:
            self.qdrant_client = qdrant_client.QdrantClient(
                host=self.qdrant_host, 
                port=self.qdrant_port
            )
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}")
            self.qdrant_client = None

    def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        try:
            # Check service health
            services = {}
            
            # Check backend - if this code is running, backend is healthy
            services["backend"] = "healthy"
            
            # Check Qdrant
            try:
                if self.qdrant_client:
                    self.qdrant_client.get_collections()
                    services["qdrant"] = "healthy"
                else:
                    services["qdrant"] = "error"
            except:
                services["qdrant"] = "error"
            
            # Check TUS service
            try:
                response = requests.get("http://tusd:1080/", timeout=2)
                services["tus"] = "healthy" if response.status_code == 200 else "warning"
            except:
                services["tus"] = "error"
            
            # Check LLaMA service
            try:
                # Try to get the model info or check if server is responding
                response = requests.get("http://llama-cpp:8080/v1/models", timeout=2)
                services["llama"] = "healthy" if response.status_code == 200 else "warning"
            except:
                services["llama"] = "error"
            
            # Get system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resources = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "memory_total": memory.total,
                "disk_percent": disk.percent,
                "disk_free": disk.free,
                "disk_total": disk.total
            }
            
            # Calculate uptime
            uptime_seconds = psutil.boot_time()
            uptime = datetime.fromtimestamp(uptime_seconds)
            uptime_str = str(datetime.now() - uptime).split('.')[0]
            
            # Determine overall status
            overall_status = "healthy"
            if any(status == "error" for status in services.values()):
                overall_status = "error"
            elif any(status == "warning" for status in services.values()):
                overall_status = "warning"
            
            # Get model information
            model_name = os.getenv('MODEL_NAME', 'Unknown')
            
            # Determine tier based on system capabilities
            try:
                from backend.utils.adaptive_config import detect_tier
                detected_tier = detect_tier()
                if detected_tier == "enterprise":
                    tier = "Enterprise"
                elif detected_tier == "server":
                    tier = "Server"
                elif detected_tier == "workstation":
                    tier = "Standard"
                elif detected_tier == "laptop":
                    tier = "Low Tier"
                else:
                    tier = "Standard"
            except Exception as e:
                logger.warning(f"Could not detect tier: {e}")
                tier = "Standard"
            
            return SystemStatus(
                status=overall_status,
                services=services,
                resources=resources,
                uptime=uptime_str,
                tier=tier,
                model=model_name
            )
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return SystemStatus(
                status="error",
                services={},
                resources={},
                uptime="unknown",
                tier="Unknown",
                model="Unknown"
            )

    def get_dashboard_stats(self) -> DashboardStats:
        """Get dashboard statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get file counts
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_files,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_files,
                        SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing_files,
                        SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_files,
                        SUM(chunks_created) as total_chunks,
                        SUM(vectors_created) as total_vectors
                    FROM file_metadata
                ''')
                row = cursor.fetchone()
                
                if row:
                    total_files, completed_files, processing_files, error_files, total_chunks, total_vectors = row
                else:
                    total_files = completed_files = processing_files = error_files = total_chunks = total_vectors = 0
                
                # Calculate storage used
                storage_used = 0
                if os.path.exists(self.upload_dir):
                    for root, dirs, files in os.walk(self.upload_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            storage_used += os.path.getsize(file_path)
                
                if os.path.exists(self.processed_dir):
                    for root, dirs, files in os.walk(self.processed_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            storage_used += os.path.getsize(file_path)
                
                # Get system resources for performance metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Determine system health
                system_health = "healthy"
                if error_files > 0:
                    system_health = "warning"
                if error_files > completed_files:
                    system_health = "error"
                
                # Count available models
                models_dir = os.getenv('MODELS_DIR', '/app/models')
                available_models = 0
                if os.path.exists(models_dir):
                    for file in os.listdir(models_dir):
                        if file.endswith('.gguf'):
                            available_models += 1
                
                # Calculate total size in GB
                total_size_gb = round(storage_used / (1024**3), 2)
                total_size_str = f"{total_size_gb} GB"
                
                return DashboardStats(
                    total_files=total_files or 0,
                    completed_files=completed_files or 0,
                    processing_files=processing_files or 0,
                    error_files=error_files or 0,
                    total_chunks=total_chunks or 0,
                    total_vectors=total_vectors or 0,
                    storage_used=storage_used,
                    system_health=system_health,
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    available_models=available_models,
                    total_size=total_size_str
                )
                
        except Exception as e:
            logger.error(f"Error getting dashboard stats: {e}")
            # Count available models even in error case
            models_dir = os.getenv('MODELS_DIR', '/app/models')
            available_models = 0
            try:
                if os.path.exists(models_dir):
                    for file in os.listdir(models_dir):
                        if file.endswith('.gguf'):
                            available_models += 1
            except:
                pass
            
            return DashboardStats(
                total_files=0,
                completed_files=0,
                processing_files=0,
                error_files=0,
                total_chunks=0,
                total_vectors=0,
                storage_used=0,
                system_health="error",
                cpu_usage=0.0,
                memory_usage=0.0,
                available_models=available_models,
                total_size="0 GB"
            )

    def get_files(self, limit: int = 50, offset: int = 0, status: Optional[str] = None, file_type: Optional[str] = None) -> List[FileInfo]:
        """Get files with filtering and pagination"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT id, filename, file_type, file_size, status, upload_time, 
                           processing_steps, chunks_created, vectors_created, error_message
                    FROM file_metadata
                    WHERE 1=1
                '''
                params = []
                
                if status:
                    query += ' AND status = ?'
                    params.append(status)
                
                if file_type:
                    query += ' AND file_type = ?'
                    params.append(file_type)
                
                query += ' ORDER BY upload_time DESC LIMIT ? OFFSET ?'
                params.extend([limit, offset])
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                files = []
                for row in rows:
                    file_id, filename, file_type, file_size, status, upload_time, processing_steps_json, chunks_created, vectors_created, error_message = row
                    
                    processing_steps = json.loads(processing_steps_json) if processing_steps_json else []
                    
                    files.append(FileInfo(
                        id=file_id,
                        name=filename,
                        type=file_type,
                        size=file_size,
                        status=status,
                        upload_time=upload_time,
                        processing_steps=processing_steps,
                        chunks_created=chunks_created or 0,
                        vectors_created=vectors_created or 0,
                        error_message=error_message
                    ))
                
                return files
                
        except Exception as e:
            logger.error(f"Error getting files: {e}")
            return []

    def search_files(self, query: str, limit: int = 50) -> List[FileInfo]:
        """Search files by name"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, filename, file_type, file_size, status, upload_time, 
                           processing_steps, chunks_created, vectors_created, error_message
                    FROM file_metadata
                    WHERE filename LIKE ?
                    ORDER BY upload_time DESC
                    LIMIT ?
                ''', [f'%{query}%', limit])
                
                rows = cursor.fetchall()
                files = []
                
                for row in rows:
                    file_id, filename, file_type, file_size, status, upload_time, processing_steps_json, chunks_created, vectors_created, error_message = row
                    
                    processing_steps = json.loads(processing_steps_json) if processing_steps_json else []
                    
                    files.append(FileInfo(
                        id=file_id,
                        name=filename,
                        type=file_type,
                        size=file_size,
                        status=status,
                        upload_time=upload_time,
                        processing_steps=processing_steps,
                        chunks_created=chunks_created or 0,
                        vectors_created=vectors_created or 0,
                        error_message=error_message
                    ))
                
                return files
                
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return []

    def get_file_preview(self, file_id: str) -> Dict[str, Any]:
        """Get file preview content"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT filename, file_path, file_type, processing_steps, error_message
                    FROM file_metadata WHERE id = ?
                ''', [file_id])
                
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="File not found")
                
                filename, file_path, file_type, processing_steps_json, error_message = row
                processing_steps = json.loads(processing_steps_json) if processing_steps_json else []
                
                # Generate preview based on file type
                preview_content = ""
                if file_type == "document":
                    preview_content = f"Document: {filename}\nType: {file_type}\nProcessing steps: {', '.join(processing_steps)}"
                elif file_type == "image":
                    preview_content = f"Image: {filename}\nType: {file_type}\nProcessing steps: {', '.join(processing_steps)}"
                elif file_type == "audio":
                    preview_content = f"Audio: {filename}\nType: {file_type}\nProcessing steps: {', '.join(processing_steps)}"
                elif file_type == "video":
                    preview_content = f"Video: {filename}\nType: {file_type}\nProcessing steps: {', '.join(processing_steps)}"
                
                if error_message:
                    preview_content += f"\n\nError: {error_message}"
                
                return {
                    "filename": filename,
                    "file_type": file_type,
                    "preview": preview_content,
                    "processing_steps": processing_steps,
                    "error_message": error_message
                }
                
        except Exception as e:
            logger.error(f"Error getting file preview: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize API
dashboard_api = DashboardAPI()

@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    return dashboard_api.get_system_status()

@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics"""
    return dashboard_api.get_dashboard_stats()

@router.get("/files", response_model=List[FileInfo])
async def get_files(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    file_type: Optional[str] = None
):
    """Get files with filtering and pagination"""
    return dashboard_api.get_files(limit, offset, status, file_type)

@router.get("/search", response_model=List[FileInfo])
async def search_files(query: str, limit: int = 50):
    """Search files by name"""
    return dashboard_api.search_files(query, limit)

@router.get("/preview/{file_id}")
async def get_file_preview(file_id: str):
    """Get file preview content"""
    return dashboard_api.get_file_preview(file_id)

@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file and its associated chunks and vectors"""
    try:
        logger.info(f"Dashboard deleting file with ID: {file_id}")
        
        # Get file metadata from database
        with sqlite3.connect(dashboard_api.db_path) as conn:
            cursor = conn.execute('''
                SELECT filename, file_path, file_type, status 
                FROM file_metadata WHERE id = ?
            ''', [file_id])
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
            
            logger.info(f"Dashboard deleting file: {filename} (Type: {file_type}, Status: {status})")
            
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
                conn.execute('DELETE FROM file_metadata WHERE id = ?', [file_id])
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
            dashboard_api.qdrant_client.get_collection(collection_name)
        except:
            logger.warning(f"Collection {collection_name} does not exist")
            return
        
        # Delete points by file_id filter
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        dashboard_api.qdrant_client.delete(
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
        logger.info(f"Dashboard deleting multiple files: {file_ids}")
        
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
        with sqlite3.connect(dashboard_api.db_path) as conn:
            cursor = conn.execute('''
                SELECT filename, file_path, file_type, status, file_size, chunks_created, vectors_created
                FROM file_metadata WHERE id = ?
            ''', [file_id])
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
                dashboard_api.qdrant_client.get_collection(collection_name)
                
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