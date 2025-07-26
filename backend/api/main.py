from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from backend.api import ingest_api
from backend.api.search_api import router as search_router

app = FastAPI()
app.include_router(ingest_api.router)
app.include_router(search_router, prefix="/search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "RAG System API is running."}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    # Placeholder: save to /tmp or pass to ingestion pipeline
    return {"filename": file.filename}
