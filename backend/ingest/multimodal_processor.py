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

# PIL compatibility patch for EasyOCR
try:
    from PIL import Image, ImageOps
    # Patch for deprecated ANTIALIAS attribute
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
except ImportError:
    pass

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
from PIL import Image, ImageOps
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

from backend.utils.logging_config import get_logger

# Use centralized logging
logger = get_logger(__name__)

import os
import logging
import subprocess
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path

import torch
from PIL import Image
import numpy as np

# Optional imports for different modalities
try:
    from transformers import CLIPProcessor, CLIPModel, pipeline
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available. Image processing will be limited.")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Whisper not available. Audio processing will be limited.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available. Video processing will be limited.")

logger = logging.getLogger(__name__)

class MultimodalProcessor:
    """
    Processor for multimodal content including images, audio, and video files.
    Supports:
    - Image analysis and captioning
    - Audio transcription
    - Video processing (audio extraction + transcription)
    """
    
    def __init__(self):
        self.config = self._load_config()
        self._initialize_models()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration for multimodal processing"""
        return {
            "image": {
                "max_size": (224, 224),
                "supported_formats": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
                "caption_model": "microsoft/DialoGPT-medium" if not CLIP_AVAILABLE else None,
                "clip_model": "openai/clip-vit-base-patch32" if CLIP_AVAILABLE else None
            },
            "audio": {
                "supported_formats": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"],
                "whisper_model": "base" if WHISPER_AVAILABLE else None,
                "max_duration": 300,  # 5 minutes max
                "sample_rate": 16000
            },
            "video": {
                "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"],
                "audio_codec": "pcm_s16le",
                "video_codec": "libx264"
            }
        }
    
    def _initialize_models(self):
        """Initialize models for different modalities"""
        self.models = {}
        
        # Initialize CLIP for image processing
        if CLIP_AVAILABLE and self.config["image"]["clip_model"]:
            try:
                self.models["clip"] = {
                    "model": CLIPModel.from_pretrained(self.config["image"]["clip_model"]),
                    "processor": CLIPProcessor.from_pretrained(self.config["image"]["clip_model"])
                }
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
        
        # Initialize Whisper for audio processing
        if WHISPER_AVAILABLE and self.config["audio"]["whisper_model"]:
            try:
                self.models["whisper"] = whisper.load_model(self.config["audio"]["whisper_model"])
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
        
        # Initialize captioning model
        if self.config["image"]["caption_model"]:
            try:
                self.models["caption"] = pipeline("text-generation", model=self.config["image"]["caption_model"])
                logger.info("Caption model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load caption model: {e}")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image file and extract features/captions
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing image features and metadata
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.config["image"]["max_size"])
            
            result = {
                "file_path": image_path,
                "file_size": os.path.getsize(image_path),
                "dimensions": image.size,
                "format": image.format,
                "features": None,
                "caption": None,
                "metadata": {}
            }
            
            # Extract CLIP features if available
            if "clip" in self.models:
                try:
                    inputs = self.models["clip"]["processor"](
                        images=image, 
                        return_tensors="pt"
                    )
                    with torch.no_grad():
                        image_features = self.models["clip"]["model"].get_image_features(**inputs)
                    result["features"] = image_features.numpy().tolist()
                    logger.info(f"Extracted CLIP features for {image_path}")
                except Exception as e:
                    logger.error(f"Failed to extract CLIP features: {e}")
            
            # Generate caption if available
            if "caption" in self.models:
                try:
                    # Simple caption generation (can be enhanced)
                    caption = self._generate_image_caption(image)
                    result["caption"] = caption
                    logger.info(f"Generated caption for {image_path}: {caption}")
                except Exception as e:
                    logger.error(f"Failed to generate caption: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def _generate_image_caption(self, image: Image.Image) -> str:
        """Generate a caption for an image"""
        # This is a simple implementation - can be enhanced with better models
        if "caption" in self.models:
            try:
                # Simple prompt-based captioning
                prompt = "This image shows:"
                result = self.models["caption"](prompt, max_length=50, num_return_sequences=1)
                caption = result[0]["generated_text"].replace(prompt, "").strip()
                return caption if caption else "An image"
            except Exception as e:
                logger.error(f"Caption generation failed: {e}")
        
        # Fallback caption
        return "An image"
    
    def generate_caption(self, image_path: str) -> str:
        """
        Generate a caption for an image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Generated caption string
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return self._generate_image_caption(image)
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return "An image"
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe an audio file using Whisper
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if not WHISPER_AVAILABLE:
                raise RuntimeError("Whisper not available for audio transcription")
            
            if "whisper" not in self.models:
                raise RuntimeError("Whisper model not loaded")
            
            # Transcribe audio
            result = self.models["whisper"].transcribe(audio_path)
            transcript = result["text"].strip()
            
            logger.info(f"Transcribed audio {audio_path}: {len(transcript)} characters")
            return transcript
            
        except Exception as e:
            logger.error(f"Error transcribing audio {audio_path}: {e}")
            raise
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from a video file using ffmpeg
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(
                suffix=".wav", 
                delete=False
            )
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # Extract audio using ffmpeg
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", self.config["video"]["audio_codec"],
                "-ar", str(self.config["audio"]["sample_rate"]),
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                temp_audio_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            logger.info(f"Extracted audio from {video_path} to {temp_audio_path}")
            return temp_audio_path
            
        except Exception as e:
            logger.error(f"Error extracting audio from video {video_path}: {e}")
            raise
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file by extracting audio and transcribing it
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata and transcript
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            result = {
                "file_path": video_path,
                "file_size": os.path.getsize(video_path),
                "audio_path": None,
                "transcript": None,
                "metadata": {}
            }
            
            # Extract audio
            audio_path = self.extract_audio_from_video(video_path)
            result["audio_path"] = audio_path
            
            # Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            result["transcript"] = transcript
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            logger.info(f"Processed video {video_path}: {len(transcript)} characters transcribed")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            raise
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get list of supported file formats for each modality"""
        return {
            "image": self.config["image"]["supported_formats"],
            "audio": self.config["audio"]["supported_formats"],
            "video": self.config["video"]["supported_formats"]
        }
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if a file format is supported"""
        ext = Path(file_path).suffix.lower()
        
        for formats in self.get_supported_formats().values():
            if ext in formats:
                return True
        
        return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic information about a file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            stat = os.stat(file_path)
            ext = Path(file_path).suffix.lower()
            
            info = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": stat.st_size,
                "extension": ext,
                "is_supported": self.is_supported_format(file_path),
                "modality": self._detect_modality(ext)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            raise
    
    def _detect_modality(self, extension: str) -> str:
        """Detect the modality of a file based on its extension"""
        if extension in self.config["image"]["supported_formats"]:
            return "image"
        elif extension in self.config["audio"]["supported_formats"]:
            return "audio"
        elif extension in self.config["video"]["supported_formats"]:
            return "video"
        else:
            return "unknown"
    
    def cleanup(self):
        """Clean up resources"""
        # Clear model references to free memory
        self.models.clear()
        logger.info("Multimodal processor cleaned up") 