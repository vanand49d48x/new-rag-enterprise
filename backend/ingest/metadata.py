# backend/ingest/metadata.py

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DocumentMetadata(BaseModel):
    title: str
    doc_type: str  # e.g. "research", "regulatory", "trial"
    section: Optional[str] = None  # e.g. "abstract", "methods"
    chunk_level: str  # "document", "section", "paragraph", etc.
    parent_id: Optional[str] = None
    created_at: datetime
    keywords: List[str]
    regulatory_tag: Optional[str] = None  # "FDA", "EMA", etc.

