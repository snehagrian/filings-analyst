import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")


def get_llm() -> BaseChatModel:
    """Return the configured chat model, selected by LLM_PROVIDER."""
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'groq' or 'ollama'.")
