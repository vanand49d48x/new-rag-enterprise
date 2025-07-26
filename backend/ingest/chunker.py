from typing import List
from backend.ingest.metadata import DocumentMetadata
from datetime import datetime
import re

# Example function to chunk by paragraphs (simple version)
def chunk_text(text: str, max_tokens: int = 400) -> List[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_tokens:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks