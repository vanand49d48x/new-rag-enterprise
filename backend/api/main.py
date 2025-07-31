from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from backend.api import ingest_api
from backend.api.search_api import router as search_router
from backend.api.chat_api import router as chat_router
from backend.api.model_api import router as model_router
from backend.utils.adaptive_config import get_adaptive_config


app = FastAPI()
app.include_router(ingest_api.router)
app.include_router(search_router, prefix="/search")
app.include_router(chat_router)
app.include_router(model_router, prefix="/api")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
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
def client_dashboard():
    """Professional client dashboard with comprehensive system monitoring"""
    with open("frontend/pages/client_dashboard.html", "r") as f:
        return f.read()

@app.get("/documentation", response_class=HTMLResponse)
def documentation():
    """Comprehensive documentation and user guide"""
    with open("frontend/pages/documentation.html", "r") as f:
        return f.read()

@app.get("/system")
def system_info():
    """Show detected system tier and recommended config"""
    return get_adaptive_config()
