# 🔍 Local RAG Dashboard Guide

## Overview

The Local RAG Dashboard is a comprehensive, privacy-first interface for uploading and managing large multimodal files in your local RAG system. It provides enterprise-grade features for handling documents, images, audio, and video files with real-time processing and vector storage.

## 🚀 Quick Start

### 1. Start the System
```bash
./start_enhanced.sh
```

### 2. Access the Dashboard
Open your browser and navigate to: **http://localhost:8000/dashboard**

## 📱 Dashboard Features

### 🎯 Core Functionality

| Feature | Description |
|---------|-------------|
| **📂 Upload Panel** | Drag & drop + file browser. Supports docs, images, audio, video |
| **📋 File Queue** | Shows upload progress, file type, size, and status |
| **⚙️ Ingestion Status** | Status: Uploaded → Processing → Vectorized |
| **🧠 Preview** | Preview extracted text, audio transcript, image OCR, or video transcript |
| **📂 Library / History** | View all previously uploaded/processed files (filterable) |

### 🎨 UI Components

#### 1. Upload Area
- **Drag & Drop**: Simply drag files onto the upload area
- **File Browser**: Click to select files manually
- **Progress Tracking**: Real-time upload progress with TUS protocol
- **File Validation**: Automatic file type detection and size validation

#### 2. File Queue
Each file shows:
- **Name & Size**: File information with formatted size display
- **Type Icon**: Visual indicator (📄 Document, 🖼️ Image, 🔊 Audio, 🎥 Video)
- **Status Badge**: Uploading, Processing, Completed, or Error
- **Progress Bar**: Visual progress indicator
- **Actions**: Preview, Retry, Delete buttons

#### 3. System Status Panel
- **Service Health**: Backend, Qdrant, TUS, LLaMA service status
- **Resource Monitoring**: CPU, Memory, Disk usage
- **Uptime**: System uptime display
- **Processing Steps**: Real-time processing status

#### 4. File Library
- **Search**: Find files by name
- **Filter**: Filter by file type (Document, Image, Audio, Video)
- **Pagination**: Handle large file collections
- **Preview**: Quick file content preview
- **Management**: Delete and manage files

## 📊 Supported File Types

### Documents
- **PDF** (.pdf) - Text extraction and chunking
- **DOCX** (.docx) - Word document processing
- **TXT** (.txt) - Plain text files
- **CSV** (.csv) - Spreadsheet data
- **Excel** (.xlsx, .xls) - Excel file processing

### Images
- **JPG/JPEG** (.jpg, .jpeg) - Image analysis and captioning
- **PNG** (.png) - Image processing
- **GIF** (.gif) - Animated image support
- **BMP** (.bmp) - Bitmap image processing
- **TIFF** (.tiff) - High-quality image support

### Audio
- **MP3** (.mp3) - Audio transcription
- **WAV** (.wav) - Wave audio processing
- **FLAC** (.flac) - Lossless audio
- **AAC** (.aac) - Advanced audio coding
- **OGG** (.ogg) - Open audio format

### Video
- **MP4** (.mp4) - Video processing with audio extraction
- **AVI** (.avi) - Audio Video Interleave
- **MOV** (.mov) - QuickTime format
- **MKV** (.mkv) - Matroska video
- **WMV** (.wmv) - Windows Media Video

## 🔧 Technical Architecture

### Frontend Components
```javascript
// Upload Component
class UploadPanel {
  - Drag & drop handling
  - File validation
  - Progress tracking (TUS)
  - Error handling
}

// File Queue Component
class FileQueue {
  - Real-time status updates
  - Progress visualization
  - Action buttons
  - Status badges
}

// Dashboard Component
class Dashboard {
  - System monitoring
  - File management
  - Search & filtering
  - Preview functionality
}
```

### Backend API Endpoints

#### Dashboard API (`/dashboard`)
- `GET /status` - System health and resource monitoring
- `GET /stats` - Dashboard statistics
- `GET /files` - File list with filtering and pagination
- `GET /search` - File search functionality
- `GET /preview/{file_id}` - File preview content
- `DELETE /files/{file_id}` - File deletion

#### Enhanced Upload API (`/enhanced-upload`)
- `POST /upload` - File upload with metadata
- `GET /status/{file_id}` - Processing status
- `GET /files` - Uploaded files list
- `DELETE /files/{file_id}` - File deletion

## 🎯 Usage Examples

### 1. Upload a Document
1. Open http://localhost:8000/dashboard
2. Drag a PDF file onto the upload area
3. Watch real-time progress
4. View processing status updates
5. Access the processed file in the library

### 2. Process Multiple Files
1. Select multiple files of different types
2. Upload all at once
3. Monitor individual file progress
4. View processing steps for each file
5. Access all processed files in the library

### 3. Search and Filter
1. Use the search box to find specific files
2. Filter by file type (Document, Image, Audio, Video)
3. View file details and processing history
4. Preview file content
5. Manage files (delete, retry, etc.)

### 4. Monitor System Health
1. Check system status panel
2. Monitor resource usage (CPU, Memory, Disk)
3. View service health indicators
4. Track processing queue status
5. Monitor storage usage

## 🔍 API Reference

### System Status
```bash
curl http://localhost:8000/dashboard/status
```
Response:
```json
{
  "status": "healthy",
  "services": {
    "backend": "healthy",
    "qdrant": "healthy",
    "tus": "healthy",
    "llama": "healthy"
  },
  "resources": {
    "cpu_percent": 15.2,
    "memory_percent": 45.8,
    "disk_percent": 23.1
  },
  "uptime": "2 days, 3:45:12"
}
```

### Dashboard Statistics
```bash
curl http://localhost:8000/dashboard/stats
```
Response:
```json
{
  "total_files": 25,
  "completed_files": 23,
  "processing_files": 1,
  "error_files": 1,
  "total_chunks": 156,
  "total_vectors": 156,
  "storage_used": 1048576,
  "system_health": "healthy"
}
```

### File List with Filtering
```bash
curl "http://localhost:8000/dashboard/files?status=completed&file_type=document&limit=10"
```

### File Search
```bash
curl "http://localhost:8000/dashboard/search?query=report&limit=5"
```

## 🛠️ Configuration

### Environment Variables
```bash
# Upload directories
UPLOAD_DIR=/app/uploads
PROCESSED_DIR=/app/processed

# Database configuration
QDRANT_HOST=localhost
QDRANT_PORT=6334

# Processing settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### Docker Configuration
The dashboard is included in the enhanced docker-compose configuration with:
- **Backend API**: FastAPI with enhanced upload and dashboard APIs
- **TUS Service**: Chunked file uploads
- **Qdrant**: Vector database for storage
- **LLaMA.cpp**: Local LLM service
- **Monitoring**: Prometheus and Grafana

## 🧪 Testing

### Run Dashboard Tests
```bash
python test_dashboard.py
```

### Test Specific Features
```bash
# Test UI accessibility
curl http://localhost:8000/dashboard

# Test API endpoints
curl http://localhost:8000/dashboard/status
curl http://localhost:8000/dashboard/stats
curl http://localhost:8000/dashboard/files
```

## 🔒 Privacy & Security

### Privacy-First Design
- **Local Processing**: All files processed locally
- **No Cloud Dependencies**: No external services required
- **Data Control**: Complete control over your data
- **Secure Storage**: Files stored in local directories

### Security Features
- **File Validation**: Type and size validation
- **Error Handling**: Comprehensive error management
- **Resource Monitoring**: System resource protection
- **Access Control**: Local network access only

## 🗑️ File Deletion System

### Overview
The Local RAG Dashboard includes a comprehensive file deletion system that ensures complete data removal for privacy, storage management, and regulatory compliance (e.g., GDPR "right to erasure").

### Deletion Features

#### 🔍 Deletion Preview
Before deleting any file, the system provides a detailed preview of what will be removed:
- **File on disk**: Physical file size and location
- **Vectors in database**: Number of vectors to be deleted from Qdrant
- **Metadata**: Processing history and file information
- **Processing status**: Whether the file can be safely deleted

#### 🧹 Complete Data Removal
The deletion process removes data from all components:
1. **File System**: Physical file deletion from disk
2. **Vector Database**: All associated vectors removed from Qdrant
3. **Metadata Database**: File records and processing history cleaned up
4. **UI Updates**: Real-time removal from dashboard interface

#### 🔐 Safety Features
- **Processing Protection**: Cannot delete files while they're being processed
- **Confirmation Dialog**: User must confirm deletion with detailed preview
- **Error Handling**: Graceful handling of partial deletion failures
- **Logging**: Complete audit trail of deletion operations

### Deletion API Endpoints

#### Single File Deletion
```bash
DELETE /dashboard/files/{file_id}
```
Response:
```json
{
  "message": "File document.pdf deleted successfully",
  "file_id": "uuid-here",
  "filename": "document.pdf",
  "deleted_components": ["file", "vectors", "metadata"]
}
```

#### Deletion Preview
```bash
GET /dashboard/files/{file_id}/deletion-preview
```
Response:
```json
{
  "file_id": "uuid-here",
  "filename": "document.pdf",
  "file_type": "document",
  "status": "completed",
  "file_size": 1048576,
  "deletion_preview": {
    "file_on_disk": true,
    "file_size_bytes": 1048576,
    "chunks_created": 15,
    "vectors_in_qdrant": 15,
    "metadata_in_db": true
  },
  "can_delete": true
}
```

#### Batch Deletion
```bash
DELETE /dashboard/files/batch
Content-Type: application/json

{
  "file_ids": ["uuid1", "uuid2", "uuid3"]
}
```

### UI Deletion Workflow

#### 1. Delete Button
Each file in the library has a "🗑️ Delete" button that triggers the deletion process.

#### 2. Confirmation Dialog
The system shows a detailed confirmation dialog with:
- **File information**: Name, size, type
- **Deletion impact**: What will be removed
- **Warning message**: "This action cannot be undone"
- **Action buttons**: Cancel or Delete Permanently

#### 3. Success Notification
After successful deletion:
- **Success message**: Confirms file deletion
- **UI update**: File removed from library
- **Stats update**: Dashboard statistics refreshed

### Error Handling

#### Common Deletion Errors
- **File not found**: 404 error for non-existent files
- **Processing in progress**: 400 error for files being processed
- **Permission denied**: 500 error for file system issues
- **Database errors**: 500 error for metadata cleanup failures

#### Error Recovery
- **Partial deletion**: System continues with remaining cleanup steps
- **Logging**: All errors logged for debugging
- **User feedback**: Clear error messages in UI
- **Retry mechanism**: Failed deletions can be retried

### Privacy & Compliance

#### GDPR Compliance
- **Right to erasure**: Complete data removal on request
- **Data control**: Users have full control over their data
- **Audit trail**: All deletion operations logged
- **Verification**: Deletion confirmed across all systems

#### Data Protection
- **Complete removal**: No data remnants left behind
- **Secure deletion**: Proper file system cleanup
- **Vector cleanup**: All embeddings removed from database
- **Metadata cleanup**: Processing history completely removed

### Testing Deletion

#### Manual Testing
1. Upload a test file
2. Wait for processing to complete
3. Click the delete button
4. Review the deletion preview
5. Confirm deletion
6. Verify file is removed from all locations

#### Automated Testing
```bash
# Run the deletion test suite
python test_deletion.py
```

#### API Testing
```bash
# Test deletion preview
curl http://localhost:8000/dashboard/files/{file_id}/deletion-preview

# Test file deletion
curl -X DELETE http://localhost:8000/dashboard/files/{file_id}
```

### Storage Management

#### Benefits
- **Space recovery**: Free up disk space by removing large files
- **Database cleanup**: Remove unused vectors to improve performance
- **Cost control**: Manage storage costs effectively
- **Performance**: Keep the system running optimally

#### Best Practices
- **Regular cleanup**: Periodically review and remove old files
- **Backup before deletion**: Ensure important data is backed up
- **Batch operations**: Use batch deletion for multiple files
- **Monitoring**: Track storage usage and deletion patterns

## 🚀 Performance Optimization

### Large File Handling
- **Chunked Uploads**: TUS protocol for files >2GB
- **Progress Tracking**: Real-time upload progress
- **Resume Capability**: Resume failed uploads
- **Memory Efficient**: Streaming processing

### Processing Optimization
- **Background Processing**: Non-blocking file processing
- **Queue Management**: Efficient processing queue
- **Resource Monitoring**: System resource tracking
- **Error Recovery**: Automatic retry mechanisms

## 🎨 Customization

### UI Customization
The dashboard uses CSS custom properties for easy theming:
```css
:root {
  --primary: #3b82f6;
  --success: #10b981;
  --warning: #f59e0b;
  --error: #ef4444;
}
```

### API Customization
Extend the dashboard API by adding new endpoints:
```python
@router.get("/custom-endpoint")
async def custom_function():
    return {"message": "Custom functionality"}
```

## 📈 Monitoring & Analytics

### System Metrics
- **CPU Usage**: Real-time CPU monitoring
- **Memory Usage**: Memory consumption tracking
- **Disk Usage**: Storage space monitoring
- **Network**: Upload/download speeds

### Processing Metrics
- **File Count**: Total files processed
- **Success Rate**: Processing success percentage
- **Processing Time**: Average processing duration
- **Error Rate**: Processing error tracking

## 🔧 Troubleshooting

### Common Issues

#### 1. Dashboard Not Loading
```bash
# Check if services are running
docker-compose ps

# Check logs
docker-compose logs backend
```

#### 2. File Upload Failing
```bash
# Check TUS service
curl http://localhost:1080/

# Check upload directory permissions
ls -la uploads/
```

#### 3. Processing Errors
```bash
# Check backend logs
docker-compose logs backend

# Check Qdrant connection
curl http://localhost:6334/health
```

### Debug Commands
```bash
# Test system health
python test_dashboard.py

# Check API endpoints
curl http://localhost:8000/dashboard/status

# Monitor logs
docker-compose logs -f
```

## 🎉 Conclusion

The Local RAG Dashboard provides a comprehensive, privacy-first solution for managing large multimodal files in your local RAG system. With its intuitive interface, real-time monitoring, and enterprise-grade features, it's the perfect tool for building and managing your local AI infrastructure.

**Key Benefits:**
- ✅ Privacy-first local processing
- ✅ Support for all major file types
- ✅ Real-time progress tracking
- ✅ Comprehensive file management
- ✅ System health monitoring
- ✅ Scalable architecture
- ✅ Easy deployment and maintenance

Start using the dashboard today at **http://localhost:8000/dashboard**! 