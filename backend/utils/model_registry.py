import os
import requests
import hashlib
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Model Registry - defines available models for each tier
MODEL_REGISTRY = {
    "laptop": {
        "primary": {
            "name": "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
            "size_mb": 580,
            "description": "Fastest, lowest memory model"
        },
        "fallback": {
            "name": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "size_mb": 320,
            "description": "Ultra-lightweight fallback"
        }
    },
    "workstation": {
        "primary": {
            "name": "qwen2.5-3b-instruct-q4_k_m.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
            "size_mb": 1800,
            "description": "Balanced performance and quality"
        },
        "fallback": {
            "name": "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
            "size_mb": 580,
            "description": "Fast fallback option"
        }
    },
    "server": {
        "primary": {
            "name": "qwen2-7b-instruct-q4_k_m.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf",
            "size_mb": 4200,
            "description": "High performance server model"
        },
        "fallback": {
            "name": "qwen2.5-3b-instruct-q4_k_m.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
            "size_mb": 1800,
            "description": "Balanced fallback"
        }
    },
    "enterprise": {
        "primary": {
            "name": "qwen2-7b-instruct-q4_k_m.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf",
            "size_mb": 4200,
            "description": "Maximum performance for enterprise"
        },
        "fallback": {
            "name": "qwen2.5-3b-instruct-q4_k_m.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
            "size_mb": 1800,
            "description": "High-quality fallback"
        }
    }
}

class ModelManager:
    def __init__(self, models_dir: str = "/app/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def get_available_models(self) -> List[str]:
        """Get list of models currently downloaded"""
        return [f.name for f in self.models_dir.glob("*.gguf")]
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is downloaded"""
        return (self.models_dir / model_name).exists()
    
    def get_model_info(self, tier: str, model_type: str = "primary") -> Dict:
        """Get model information for a specific tier"""
        if tier not in MODEL_REGISTRY:
            raise ValueError(f"Unknown tier: {tier}")
        return MODEL_REGISTRY[tier][model_type]
    
    def download_model(self, tier: str, model_type: str = "primary") -> bool:
        """Download a model for a specific tier"""
        try:
            model_info = self.get_model_info(tier, model_type)
            model_path = self.models_dir / model_info["name"]
            
            if model_path.exists():
                logger.info(f"Model {model_info['name']} already exists")
                return True
            
            logger.info(f"Downloading {model_info['name']} ({model_info['size_mb']}MB)...")
            
            # Download with progress
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
            
            logger.info(f"✅ Downloaded {model_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            return False
    
    def get_recommended_model(self, tier: str) -> str:
        """Get the recommended model name for a tier"""
        # Try primary model first
        if self.is_model_available(self.get_model_info(tier, "primary")["name"]):
            return self.get_model_info(tier, "primary")["name"]
        
        # Try fallback model
        if self.is_model_available(self.get_model_info(tier, "fallback")["name"]):
            return self.get_model_info(tier, "fallback")["name"]
        
        # Download primary model
        if self.download_model(tier, "primary"):
            return self.get_model_info(tier, "primary")["name"]
        
        # Download fallback model
        if self.download_model(tier, "fallback"):
            return self.get_model_info(tier, "fallback")["name"]
        
        raise Exception(f"No available models for tier: {tier}")
    
    def get_model_stats(self) -> Dict:
        """Get statistics about available models"""
        available_models = self.get_available_models()
        total_size = sum(
            (self.models_dir / model).stat().st_size 
            for model in available_models
        )
        
        return {
            "available_models": available_models,
            "total_models": len(available_models),
            "total_size_gb": round(total_size / (1024**3), 2),
            "models_dir": str(self.models_dir)
        }

# Global model manager instance
model_manager = ModelManager() 