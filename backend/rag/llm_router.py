import yaml
from backend.rag.llm_clients import call_ollama

with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def route_llm(prompt: str) -> str:
    try:
        return call_ollama(prompt, cfg["ollama"])
    except Exception as e:
        print(f"❌ Ollama failed: {e}")
        return "⚠️ Failed to generate response."
