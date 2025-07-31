"""
Enterprise Multi-Modal Document Processor
Implements hierarchical chunking strategy for 50K+ documents
Based on enterprise RAG best practices
"""

import os
import io
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import mimetypes
import magic

# Document processing
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import csv
import openpyxl
import email
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text

# Image processing
import cv2
import numpy as np
from PIL import Image
import easyocr
import pytesseract

# Audio/Video processing
import librosa
import soundfile as sf
from pydub import AudioSegment
import whisper
from moviepy.editor import VideoFileClip

# Text processing
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnterpriseMultimodalProcessor:
    """
    Enterprise-grade multi-modal document processor with hierarchical chunking
    """
    
    def __init__(self):
        self.ocr_reader = easyocr.Reader(['en'])
        self.whisper_model = whisper.load_model("base")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
    def process_file(self, file_bytes: bytes, filename: str, doc_type: str = "uploaded") -> List[Dict[str, Any]]:
        """
        Main entry point for processing any file type
        """
        try:
            # Detect file type
            file_type = self._detect_file_type(file_bytes, filename)
            
            # Extract content based on file type
            if file_type in ["pdf", "docx", "txt", "csv", "xlsx", "html", "rtf", "eml"]:
                content = self._extract_text_from_document(file_bytes, file_type)
            elif file_type in ["png", "jpg", "jpeg", "bmp", "tiff"]:
                content = self._extract_text_from_image(file_bytes)
            elif file_type in ["mp3", "wav", "flac", "m4a"]:
                content = self._extract_text_from_audio(file_bytes)
            elif file_type in ["mp4", "mov", "avi", "mkv"]:
                content = self._extract_text_from_video(file_bytes)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Process with hierarchical chunking
            return self._process_with_hierarchical_chunking(
                content=content,
                filename=filename,
                file_type=file_type,
                doc_type=doc_type
            )
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise
    
    def _detect_file_type(self, file_bytes: bytes, filename: str) -> str:
        """Detect file type using both magic numbers and extension"""
        # Use python-magic for MIME type detection
        mime_type = magic.from_buffer(file_bytes, mime=True)
        
        # Map MIME types to file extensions
        mime_to_ext = {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'text/plain': 'txt',
            'text/csv': 'csv',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            'text/html': 'html',
            'text/rtf': 'rtf',
            'message/rfc822': 'eml',
            'image/png': 'png',
            'image/jpeg': 'jpg',
            'image/bmp': 'bmp',
            'image/tiff': 'tiff',
            'audio/mpeg': 'mp3',
            'audio/wav': 'wav',
            'audio/flac': 'flac',
            'audio/mp4': 'm4a',
            'video/mp4': 'mp4',
            'video/quicktime': 'mov',
            'video/x-msvideo': 'avi',
            'video/x-matroska': 'mkv'
        }
        
        detected_type = mime_to_ext.get(mime_type)
        if detected_type:
            return detected_type
        
        # Fallback to file extension
        ext = filename.lower().split(".")[-1]
        return ext
    
    def _extract_text_from_document(self, file_bytes: bytes, file_type: str) -> str:
        """Extract text from various document formats"""
        try:
            if file_type == "pdf":
                reader = PdfReader(io.BytesIO(file_bytes))
                content = "\n\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                
            elif file_type in ["docx", "doc"]:
                doc = DocxDocument(io.BytesIO(file_bytes))
                content = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                
            elif file_type == "csv":
                decoded = file_bytes.decode("utf-8")
                csv_reader = csv.reader(io.StringIO(decoded))
                content = "\n".join([", ".join(row) for row in csv_reader])
                
            elif file_type == "xlsx":
                workbook = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True)
                sheet_text = []
                for sheet in workbook.worksheets:
                    for row in sheet.iter_rows(values_only=True):
                        sheet_text.append(", ".join([str(cell) if cell is not None else "" for cell in row]))
                content = "\n".join(sheet_text)
                
            elif file_type in ["html", "htm"]:
                html = file_bytes.decode("utf-8")
                soup = BeautifulSoup(html, "html.parser")
                content = soup.get_text(separator="\n")
                
            elif file_type == "rtf":
                rtf = file_bytes.decode("utf-8")
                content = rtf_to_text(rtf)
                
            elif file_type == "eml":
                msg = email.message_from_bytes(file_bytes)
                content = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            content += part.get_payload(decode=True).decode("utf-8")
                else:
                    content = msg.get_payload(decode=True).decode("utf-8")
                    
            else:  # txt and other text files
                content = file_bytes.decode("utf-8")
                
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_type}: {str(e)}")
            return f"[Error extracting text from {file_type} file]"
    
    def _extract_text_from_image(self, file_bytes: bytes) -> str:
        """Extract text from images using OCR"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(file_bytes))
            
            # Convert to OpenCV format for better OCR
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Try EasyOCR first (better for complex layouts)
            try:
                results = self.ocr_reader.readtext(cv_image)
                text = " ".join([result[1] for result in results])
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
            
            # Fallback to Tesseract
            try:
                text = pytesseract.image_to_string(image)
                return text.strip()
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")
            
            return "[No text extracted from image]"
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return "[Error extracting text from image]"
    
    def _extract_text_from_audio(self, file_bytes: bytes) -> str:
        """Extract text from audio using Whisper"""
        try:
            # Save temporary audio file
            temp_path = f"/tmp/audio_{hashlib.md5(file_bytes).hexdigest()}.wav"
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            
            # Transcribe audio using Whisper
            result = self.whisper_model.transcribe(temp_path)
            audio_text = result["text"]
            
            # Clean up
            os.remove(temp_path)
            
            return audio_text.strip() if audio_text.strip() else "[No speech detected in audio]"
            
        except Exception as e:
            logger.error(f"Error extracting text from audio: {str(e)}")
            return f"[Error extracting text from audio: {str(e)}]"
    
    def _extract_text_from_video(self, file_bytes: bytes) -> str:
        """Extract text from video (audio + OCR on frames)"""
        try:
            # Save temporary file
            temp_path = f"/tmp/video_{hashlib.md5(file_bytes).hexdigest()}.mp4"
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            
            # Extract audio and transcribe
            video = VideoFileClip(temp_path)
            audio_path = temp_path.replace(".mp4", "_audio.wav")
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            
            # Transcribe audio using Whisper
            result = self.whisper_model.transcribe(audio_path)
            audio_text = result["text"]
            
            # Extract frames for OCR (every 5 seconds)
            frame_texts = []
            fps = video.fps
            frame_interval = int(fps * 5)  # Every 5 seconds
            
            for i, frame in enumerate(video.iter_frames()):
                if i % frame_interval == 0:
                    # Convert frame to PIL Image
                    frame_pil = Image.fromarray(frame)
                    
                    # OCR on frame
                    try:
                        frame_text = pytesseract.image_to_string(frame_pil)
                        if frame_text.strip():
                            frame_texts.append(f"[Frame {i//frame_interval}]: {frame_text.strip()}")
                    except:
                        pass
            
            # Combine audio and visual text
            combined_text = f"Audio Transcription: {audio_text}\n\nVisual Text:\n" + "\n".join(frame_texts)
            
            # Clean up
            video.close()
            os.remove(temp_path)
            os.remove(audio_path)
            
            return combined_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from video: {str(e)}")
            return "[Error extracting text from video]"
    
    def _process_with_hierarchical_chunking(self, content: str, filename: str, file_type: str, doc_type: str) -> List[Dict[str, Any]]:
        """
        Implement hierarchical chunking strategy for enterprise documents
        Level 1: Document-level metadata
        Level 2: Section-level chunks (if applicable)
        Level 3: Paragraph-level chunks
        Level 4: Sentence-level for precise retrieval
        """
        created = datetime.utcnow()
        file_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Level 1: Document-level metadata
        doc_metadata = {
            "title": filename,
            "file_type": file_type,
            "doc_type": doc_type,
            "created": created.isoformat(),
            "file_hash": file_hash,
            "content_length": len(content),
            "chunk_level": "document"
        }
        
        # Level 2: Section-level chunks (for structured documents)
        section_chunks = self._extract_sections(content, filename, file_type)
        
        # Level 3: Paragraph-level chunks
        paragraph_chunks = self.text_splitter.split_text(content)
        
        # Level 4: Sentence-level chunks for precise retrieval
        sentence_chunks = self._split_into_sentences(content)
        
        # Combine all chunks with appropriate metadata
        all_chunks = []
        
        # Add document-level chunk
        all_chunks.append({
            "id": f"{filename}-doc-{file_hash}",
            "text": content[:1000] + "..." if len(content) > 1000 else content,  # Truncated for document-level
            "metadata": {**doc_metadata, "chunk_type": "document_summary"}
        })
        
        # Add section chunks
        for i, section in enumerate(section_chunks):
            all_chunks.append({
                "id": f"{filename}-section-{i}-{file_hash}",
                "text": section["text"],
                "metadata": {
                    **doc_metadata,
                    "chunk_type": "section",
                    "section_title": section.get("title", ""),
                    "section_index": i,
                    "chunk_level": "section"
                }
            })
        
        # Add paragraph chunks
        for i, chunk in enumerate(paragraph_chunks):
            all_chunks.append({
                "id": f"{filename}-para-{i}-{file_hash}",
                "text": chunk,
                "metadata": {
                    **doc_metadata,
                    "chunk_type": "paragraph",
                    "paragraph_index": i,
                    "chunk_level": "paragraph"
                }
            })
        
        # Add sentence chunks (for high-precision retrieval)
        for i, sentence in enumerate(sentence_chunks):
            if len(sentence.strip()) > 10:  # Only meaningful sentences
                all_chunks.append({
                    "id": f"{filename}-sent-{i}-{file_hash}",
                    "text": sentence,
                    "metadata": {
                        **doc_metadata,
                        "chunk_type": "sentence",
                        "sentence_index": i,
                        "chunk_level": "sentence"
                    }
                })
        
        return all_chunks
    
    def _extract_sections(self, content: str, filename: str, file_type: str) -> List[Dict[str, str]]:
        """Extract sections from structured documents"""
        sections = []
        
        if file_type == "pdf":
            # Try to extract sections based on common patterns
            lines = content.split('\n')
            current_section = {"title": "Introduction", "text": ""}
            
            for line in lines:
                # Detect section headers (all caps, numbered, etc.)
                if (line.isupper() and len(line.strip()) > 3) or \
                   re.match(r'^\d+\.\s+[A-Z]', line.strip()):
                    if current_section["text"].strip():
                        sections.append(current_section)
                    current_section = {"title": line.strip(), "text": ""}
                else:
                    current_section["text"] += line + "\n"
            
            if current_section["text"].strip():
                sections.append(current_section)
        
        return sections
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for precise retrieval"""
        # Simple sentence splitting - can be enhanced with NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10] 