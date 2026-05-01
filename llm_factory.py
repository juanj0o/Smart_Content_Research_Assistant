"""
Factory que retorna el LLM correcto según LLM_PROVIDER en .env.
Los nodos importan de acá en lugar de importar ChatOllama/ChatGroq directo.
"""

from config import PROVIDER


def get_llm(model: str, **kwargs):
    """
    Retorna una instancia del LLM para el provider configurado.
    kwargs se pasan directo al constructor (ej: temperature=0).
    """
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, **kwargs)

    if PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, **kwargs)

    raise ValueError(f"Provider no soportado: '{PROVIDER}'. Opciones: ollama | groq")