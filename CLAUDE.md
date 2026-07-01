# Project: Agentic-RAG SEC Filings Analyst

An agent that answers analyst-grade questions about public companies by doing
self-correcting, multi-hop RAG over SEC filings (10-Ks, 10-Qs). LangGraph drives
the retrieve -> grade -> re-query loop.

## Stack (do not substitute without asking)
- Python 3.10+
- LangGraph + LangChain (agent orchestration)
- LLM: Llama via Groq API (hosted) OR Ollama (local), toggled by env var
- Embeddings: sentence-transformers, model BAAI/bge-small-en-v1.5 (local, free)
- Vector store: Chroma (local, persistent, ./chroma_db)
- Data: SEC EDGAR via the sec-edgar-api wrapper; filing HTML parsed with
  BeautifulSoup
- UI: Streamlit
- Config via python-dotenv reading .env

## Project layout
- src/config.py        -> loads env, returns the LLM client (Groq or Ollama)
- src/ingest.py        -> fetch + cache filings from EDGAR to data/cache/
- src/index.py         -> chunk, embed, upsert into Chroma
- src/agent.py         -> the LangGraph agent (nodes + conditional edges)
- src/app.py           -> Streamlit UI
- scripts/             -> small runnable test scripts per phase
- data/cache/          -> raw cached filings (gitignored)
- chroma_db/           -> vector store (gitignored)

## Commands
- Install: pip install -r requirements.txt
- Run app: streamlit run src/app.py
- Ingest: python -m src.ingest --tickers AAPL MSFT
- Index:  python -m src.index

## Conventions
- Read config ONLY through src/config.py. No hardcoded keys or model names
  elsewhere.
- The Groq/Ollama choice is one env var (LLM_PROVIDER). Both must work; never
  break local Ollama to make Groq work or vice versa.
- Always cache EDGAR responses to data/cache/ and read from cache if present.
  Never re-fetch a filing we already have.

## Always / Never
- ALWAYS send the SEC_USER_AGENT header on every EDGAR request.
- ALWAYS rate-limit EDGAR calls to <= 8 requests/second with a small delay.
- NEVER commit .env, API keys, or the data/cache and chroma_db folders.
- NEVER put secrets in code; read them from env.
- This project does NOT execute model-generated code, so no sandbox/Docker is
  needed. Do not add one.

## Build approach
- We build in phases. Implement ONLY the phase I ask for. Keep each phase
  runnable and add a small script in scripts/ to verify it before moving on.
- Prefer small, readable functions over cleverness. Add brief docstrings.
