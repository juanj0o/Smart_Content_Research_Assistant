import os

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

MODELS = {
    "ollama": {
        "fast":    "qwen2.5:7b",
        "smart":   "qwen2.5:32b",
        "premium": "qwen2.5:32b",
    },
    "groq": {
        "fast":    "llama-3.1-8b-instant",
        "smart":   "llama-3.3-70b-versatile",
        "premium": "llama-3.3-70b-versatile",
    },
}

PRICING = {
    "qwen2.5:7b":                {"input": 0.0,  "output": 0.0},
    "qwen2.5:32b":               {"input": 0.0,  "output": 0.0},
    "llama-3.1-8b-instant":      {"input": 0.0,  "output": 0.0},
    "llama-3.3-70b-versatile":   {"input": 0.0,  "output": 0.0},
}

MODEL_DISPLAY_NAMES = {
    "qwen2.5:7b":                "qwen2.5-7b",
    "qwen2.5:32b":               "qwen2.5-32b",
    "llama-3.1-8b-instant":      "llama-3.1-8b",
    "llama-3.3-70b-versatile":   "llama-3.3-70b",
}

def get_model(tier: str) -> str:
    """Retorna el nombre del modelo para el tier y provider actuales."""
    return MODELS[PROVIDER][tier]