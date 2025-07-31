import requests

def call_ollama(prompt: str, cfg: dict) -> str:
    r = requests.post(
        f"http://{cfg['host']}:{cfg['port']}/api/chat",
        json={
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        },
        timeout=10
    )
    r.raise_for_status()
    return r.json()["message"]["content"]

