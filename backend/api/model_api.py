from fastapi import APIRouter, HTTPException
from typing import Dict, List
import logging
from backend.utils.model_registry import model_manager
from backend.utils.adaptive_config import get_tier_config, detect_tier

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/models")
def get_models():
    """Get available models and model statistics"""
    try:
        return {
            "available_models": model_manager.get_available_models(),
            "model_stats": model_manager.get_model_stats(),
            "current_tier": detect_tier()
        }
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{tier}")
def get_tier_models(tier: str):
    """Get model information for a specific tier"""
    try:
        from backend.utils.model_registry import MODEL_REGISTRY
        if tier not in MODEL_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Unknown tier: {tier}")
        
        tier_models = MODEL_REGISTRY[tier]
        available_models = model_manager.get_available_models()
        
        # Check which models are available
        for model_type, model_info in tier_models.items():
            model_info["available"] = model_info["name"] in available_models
        
        return {
            "tier": tier,
            "models": tier_models,
            "recommended_model": model_manager.get_recommended_model(tier)
        }
    except Exception as e:
        logger.error(f"Error getting tier models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/download/{tier}")
def download_model(tier: str, model_type: str = "primary"):
    """Download a model for a specific tier"""
    try:
        success = model_manager.download_model(tier, model_type)
        if success:
            return {
                "message": f"Successfully downloaded model for tier {tier}",
                "tier": tier,
                "model_type": model_type,
                "model_name": model_manager.get_model_info(tier, model_type)["name"]
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to download model")
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tiers")
def get_tiers():
    """Get all available tiers and their configurations"""
    try:
        from backend.utils.adaptive_config import TIERS
        tiers = {}
        for tier in TIERS.keys():
            try:
                tiers[tier] = get_tier_config(tier)
            except Exception as e:
                logger.warning(f"Could not get config for tier {tier}: {e}")
                tiers[tier] = {"error": str(e)}
        
        return {
            "tiers": tiers,
            "current_tier": detect_tier()
        }
    except Exception as e:
        logger.error(f"Error getting tiers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tiers/{tier}")
def get_tier_info(tier: str):
    """Get configuration for a specific tier"""
    try:
        return get_tier_config(tier)
    except Exception as e:
        logger.error(f"Error getting tier info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tiers/{tier}/switch")
def switch_tier(tier: str):
    """Switch to a different tier (for demo purposes)"""
    try:
        config = get_tier_config(tier)
        return {
            "message": f"Switched to tier: {tier}",
            "config": config,
            "note": "This is a demo switch. Restart required for full effect."
        }
    except Exception as e:
        logger.error(f"Error switching tier: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
def get_performance_metrics():
    """Get performance metrics for different tiers"""
    try:
        import psutil
        import time
        
        # Get current system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Simulate performance metrics for different tiers
        performance_data = {
            "current_system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2)
            },
            "tier_performance": {
                "laptop": {
                    "estimated_response_time_ms": 500,
                    "memory_usage_mb": 512,
                    "cpu_usage_percent": 15
                },
                "workstation": {
                    "estimated_response_time_ms": 800,
                    "memory_usage_mb": 1024,
                    "cpu_usage_percent": 25
                },
                "server": {
                    "estimated_response_time_ms": 1200,
                    "memory_usage_mb": 2048,
                    "cpu_usage_percent": 40
                },
                "enterprise": {
                    "estimated_response_time_ms": 1500,
                    "memory_usage_mb": 4096,
                    "cpu_usage_percent": 60
                }
            }
        }
        
        return performance_data
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 