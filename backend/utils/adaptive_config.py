import psutil
import platform
import os
from typing import Dict
from .model_registry import model_manager

TIERS = {
    "laptop": {
        "threads": 2,
        "context_size": 512,
        "batch_size": 64,
        "max_tokens": 32,
        "description": "Fastest, lowest memory, for laptops or VMs."
    },
    "workstation": {
        "threads": 4,
        "context_size": 1024,
        "batch_size": 128,
        "max_tokens": 64,
        "description": "Balanced for desktops and dev workstations."
    },
    "server": {
        "threads": 8,
        "context_size": 2048,
        "batch_size": 256,
        "max_tokens": 128,
        "description": "High performance for servers."
    },
    "enterprise": {
        "threads": 16,
        "context_size": 4096,
        "batch_size": 512,
        "max_tokens": 256,
        "description": "Max performance for enterprise hardware."
    }
}

def detect_tier() -> str:
    # Check for host system info from environment variables first
    host_ram_gb = os.getenv("HOST_RAM_GB")
    host_cpu_count = os.getenv("HOST_CPU_COUNT")
    
    if host_ram_gb and host_cpu_count:
        # Use host system info
        ram_gb = float(host_ram_gb)
        cpu_count = int(host_cpu_count)
        print(f"Using host system info: {ram_gb}GB RAM, {cpu_count} CPU cores")
    else:
        # Fall back to container detection
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_count = psutil.cpu_count(logical=True)
        print(f"Using container detection: {ram_gb:.1f}GB RAM, {cpu_count} CPU cores")
    
    if ram_gb < 8 or cpu_count <= 2:
        return "laptop"
    elif ram_gb < 16 or cpu_count <= 4:
        return "workstation"
    elif ram_gb < 32 or cpu_count <= 8:
        return "server"
    else:
        return "enterprise"

def get_adaptive_config() -> Dict:
    tier = detect_tier()
    
    # Try to load from auto-generated config first
    try:
        from .config import load_config
        yaml_config = load_config()
        
        # Use YAML config values if available
        config = {
            "threads": yaml_config.get("llm", {}).get("threads", TIERS[tier]["threads"]),
            "context_size": yaml_config.get("llm", {}).get("context_size", TIERS[tier]["context_size"]),
            "batch_size": yaml_config.get("llm", {}).get("batch_size", TIERS[tier]["batch_size"]),
            "max_tokens": yaml_config.get("llm", {}).get("max_tokens", TIERS[tier]["max_tokens"]),
            "description": TIERS[tier]["description"],
            "tier": tier,
            "model": yaml_config.get("llm", {}).get("model", "qwen2.5-3b-instruct-q4_k_m.gguf")
        }
        print(f"Using auto-generated configuration")
    except Exception as e:
        print(f"Warning: Could not load auto-generated config: {e}")
        # Fallback to tier-based config
        config = TIERS[tier].copy()
        config["tier"] = tier
        
        # Check for environment variable override
        env_model = os.getenv("MODEL_NAME")
        if env_model:
            config["model"] = env_model
            print(f"Using environment variable model: {env_model}")
        else:
            # Get recommended model for this tier
            try:
                config["model"] = model_manager.get_recommended_model(tier)
            except Exception as e:
                print(f"Warning: Could not get recommended model for tier {tier}: {e}")
                config["model"] = "qwen2.5-3b-instruct-q4_k_m.gguf"
    
    # Get actual system info (host or container)
    host_ram_gb = os.getenv("HOST_RAM_GB")
    if host_ram_gb:
        config["ram_gb"] = float(host_ram_gb)
        config["cpu_count"] = int(os.getenv("HOST_CPU_COUNT", "0"))
        config["system"] = "macOS"  # Assuming host is macOS
    else:
        config["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        config["cpu_count"] = psutil.cpu_count(logical=True)
        config["system"] = platform.system()
    
    # Add model statistics
    try:
        model_stats = model_manager.get_model_stats()
        config["model_stats"] = model_stats
    except Exception as e:
        config["model_stats"] = {"error": str(e)}
    
    return config

def get_tier_config(tier: str) -> Dict:
    """Get configuration for a specific tier"""
    if tier not in TIERS:
        raise ValueError(f"Unknown tier: {tier}")
    
    config = TIERS[tier].copy()
    config["tier"] = tier
    
    # Get recommended model for this tier
    try:
        config["model"] = model_manager.get_recommended_model(tier)
    except Exception as e:
        print(f"Warning: Could not get recommended model for tier {tier}: {e}")
        config["model"] = "qwen2-7b-instruct-q4_k_m.gguf"
    
    return config 