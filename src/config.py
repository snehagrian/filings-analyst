import os
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def _get(key: str, default: str | None = None) -> str | None:
    """Read a setting from the environment (.env) or Streamlit secrets.

    Environment variables take precedence. Streamlit secrets are consulted only
    when streamlit is already loaded (i.e. inside the app), so CLI scripts don't
    pull in streamlit or require a secrets file.
    """
    value = os.getenv(key)
    if value not in (None, ""):
        return value

    streamlit = sys.modules.get("streamlit")
    if streamlit is not None:
        try:
            if key in streamlit.secrets:
                return str(streamlit.secrets[key])
        except Exception:
            pass
    return default


LLM_PROVIDER = _get("LLM_PROVIDER", "groq")
GROQ_API_KEY = _get("GROQ_API_KEY")
GROQ_MODEL = _get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = _get("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
OLLAMA_MODEL = _get("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434")
SEC_USER_AGENT = _get("SEC_USER_AGENT")


def get_llm() -> BaseChatModel:
    """Return the configured chat model, selected by LLM_PROVIDER."""
    if LLM_PROVIDER == "groq":
        return get_groq_llm()

    if LLM_PROVIDER == "ollama":
        return get_ollama_llm()

    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'groq' or 'ollama'.")


def get_groq_llm(model: str | None = None) -> BaseChatModel:
    """Return a Groq chat model for the given model name."""
    from langchain_groq import ChatGroq

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or Streamlit secrets."
        )
    return ChatGroq(api_key=GROQ_API_KEY, model=model or GROQ_MODEL)


def get_ollama_llm() -> BaseChatModel:
    """Return the local Ollama chat model, regardless of the active provider."""
    from langchain_ollama import ChatOllama

    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


def ollama_fallback_enabled() -> bool:
    """Return True when the Ollama endpoint looks reachable from cloud.

    The default localhost URL is only valid on a developer machine. In cloud
    deployments we only use Ollama when the base URL is a non-local host or the
    user explicitly opts in via OLLAMA_FALLBACK=1.
    """
    if os.getenv("OLLAMA_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True

    parsed = urlparse(OLLAMA_BASE_URL or "")
    host = (parsed.hostname or "").lower()
    return host not in {"", "localhost", "127.0.0.1", "::1"}
