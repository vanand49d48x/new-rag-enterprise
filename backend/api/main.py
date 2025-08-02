from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from datetime import datetime
from backend.api import ingest_api_simple as ingest_api
from backend.api.search_api import router as search_router
from backend.api.chat_api import router as chat_router
from backend.api.model_api import router as model_router
from backend.api.enhanced_upload_api import router as enhanced_upload_router
from backend.api.dashboard_api import router as dashboard_router
from backend.utils.adaptive_config import get_adaptive_config
from backend.utils.logging_config import setup_logging, get_logger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Setup centralized logging
setup_logging()
logger = get_logger(__name__)


app = FastAPI(
    title="Enterprise RAG System",
    description="Multi-modal document processing and intelligent chat",
    version="1.0.0",
    # Increase timeout for large file uploads
    docs_url="/docs",
    redoc_url="/redoc"
)
app.include_router(ingest_api.router)
app.include_router(search_router, prefix="/search")
app.include_router(chat_router)
app.include_router(model_router, prefix="/api")
app.include_router(enhanced_upload_router)
app.include_router(dashboard_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: Timeout handling is done at the frontend level with AbortSignal.timeout()

@app.get("/", response_class=HTMLResponse)
def read_root():
    logger.info(f"Main interface accessed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    with open("frontend/pages/clean_template.html", "r") as f:
        return f.read()

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    # Placeholder: save to /tmp or pass to ingestion pipeline
    return {"filename": file.filename}

@app.get("/debug", response_class=HTMLResponse)
def debug_upload():
    """Debug endpoint for file upload testing"""
    with open("frontend/pages/debug_upload.html", "r") as f:
        return f.read()

@app.get("/test", response_class=HTMLResponse)
def test_upload():
    """Simple test endpoint for file upload testing"""
    with open("frontend/pages/test_upload_simple.html", "r") as f:
        return f.read()

@app.get("/demo", response_class=HTMLResponse)
def client_demo():
    """Client demo interface for adaptive configuration"""
    with open("frontend/pages/client_demo_interface.html", "r") as f:
        return f.read()

@app.get("/dashboard", response_class=HTMLResponse)
def universal_dashboard():
    """Universal dashboard with system monitoring and file management"""
    with open("frontend/pages/universal_dashboard.html", "r") as f:
        return f.read()

@app.get("/documentation", response_class=HTMLResponse)
def documentation():
    """Comprehensive documentation and user guide"""
    with open("frontend/pages/documentation.html", "r") as f:
        return f.read()

@app.get("/enhanced", response_class=HTMLResponse)
def enhanced_interface():
    """Enhanced RAG interface with drag-and-drop upload and chat"""
    logger.info(f"Enhanced interface accessed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    with open("frontend/pages/enhanced_rag_interface.html", "r") as f:
        return f.read()

@app.get("/enhanced-upload", response_class=HTMLResponse)
def enhanced_upload_interface():
    """Enhanced upload interface with chunked uploads and processing"""
    logger.info(f"Enhanced upload interface accessed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    with open("frontend/pages/enhanced_upload_interface.html", "r") as f:
        return f.read()

@app.get("/file-dashboard", response_class=HTMLResponse)
def local_rag_dashboard():
    """Local RAG upload dashboard with comprehensive file management"""
    logger.info(f"Local RAG dashboard accessed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    with open("frontend/pages/local_rag_dashboard.html", "r") as f:
        return f.read()

@app.get("/progress-test", response_class=HTMLResponse)
def progress_test():
    """Test page for real-time progress tracking"""
    with open("frontend/pages/progress_test.html", "r") as f:
        return f.read()

@app.get("/system")
def system_info():
    """Show detected system tier and recommended config"""
    return get_adaptive_config()

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
