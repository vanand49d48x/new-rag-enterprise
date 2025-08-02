import os
import yaml

def load_config():
    """Load configuration with priority order for auto-generated configs"""
    # Priority order for config files
    config_paths = [
        "configs/auto_generated.yaml",  # Auto-generated config (highest priority)
        os.getenv("CONFIG_PATH", "config.yaml"),  # Manual config
        "configs/config.yaml"           # Fallback config
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            print(f"Loading configuration from: {config_path}")
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
    
    # Return default configuration if no config file found
    print("No configuration file found, using default config")
    return get_default_config()

def get_default_config():
    """Return default configuration"""
    return {
        "llm": {
            "model": "qwen2.5-3b-instruct-q4_k_m.gguf",
            "backend": "llama_cpp",
            "api_url": "http://llama-cpp:8080/completion",
            "timeout": 60,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "context_size": 4096,
            "batch_size": 256,
            "threads": 8,
            "gpu_layers": 0,
            "stream": True,
            "flash_attention": False
        },
        "processing": {
            "chunk_size": 512,
            "chunk_overlap": 128,
            "max_workers": 6,
            "embedding_batch_size": 32,
            "prompt_trimming": True,
            "max_prompt_tokens": 500,
            "max_chunks": 3
        },
        "api": {
            "timeout": 60,
            "stream_responses": True,
            "enable_flash_attention": False
        },
        "vector_store": {
            "max_results": 5,
            "max_chunks_per_query": 3,
            "similarity_threshold": 0.7
        }
    }

def get_config():
    """Get configuration - alias for load_config for backward compatibility"""
    return load_config()
