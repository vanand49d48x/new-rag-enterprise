# File: backend/ingest/processor.py

from datetime import datetime
from backend.ingest.chunker import chunk_text
from typing import List

# Placeholder functions for future multimodal support
def extract_text_from_document(file_bytes: bytes, extension: str) -> str:
    from backend.ingest.file_loader import extract_text
    return extract_text(file_bytes, extension)

def extract_text_from_image(file_bytes: bytes) -> str:
    # Placeholder - implement OCR later
    return "[Image to text placeholder]"

def extract_text_from_audio(file_bytes: bytes) -> str:
    # Placeholder - implement Whisper or ASR later
    return "[Audio to text placeholder]"

def extract_text_from_video(file_bytes: bytes) -> str:
    # Placeholder - extract audio then transcribe
    return "[Video to text placeholder]"

def process_document(text: str, title: str, doc_type: str = "uploaded") -> List[dict]:
    created = datetime.utcnow()
    chunks = chunk_text(text)
    return [
        {
            "id": f"{title}-{i}",
            "text": chunk,
            "metadata": {
                "title": title,
                "chunk": i,
                "created": created.isoformat(),
                "doc_type": doc_type
            }
        }
        for i, chunk in enumerate(chunks)
    ]

def process_file(file_bytes: bytes, filename: str) -> List[dict]:
    ext = filename.lower().split(".")[-1]

    if ext in ["pdf", "docx", "txt", "csv", "xlsx", "html", "rtf", "eml"]:
        text = extract_text_from_document(file_bytes, ext)
    elif ext in ["png", "jpg", "jpeg"]:
        text = extract_text_from_image(file_bytes)
    elif ext in ["mp3", "wav"]:
        text = extract_text_from_audio(file_bytes)
    elif ext in ["mp4", "mov"]:
        text = extract_text_from_video(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return process_document(
        text=text,
        title=filename,
        doc_type=ext
    )
