# Verdant — an SEC filings analyst that checks its own work

Verdant is a research assistant for reading 10-Ks and 10-Qs. You ask it something
like "what are Apple's main risk factors?" or "compare Apple and Microsoft's
AI-related risks," and it finds the relevant passages in the actual SEC filings,
writes an answer with citations, and then fact-checks that answer against the
source text before showing it to you.

I built it because most "chat with your documents" demos will happily make things
up, and with financial filings that's a problem. So the point here is that the
agent doesn't just retrieve-and-generate. It grades whether it actually found what
it needs, goes back for more if it didn't, and strips out any claim it can't back
up with a citation.

## What it does

- Answers questions about public companies straight from their SEC filings.
- Cites every claim with its source (ticker, form, filing date).
- Handles multi-company comparisons by retrieving each company separately, so one
  doesn't drown out the other.
- Fetches any company you ask about on the fly. If it isn't already indexed, it
  pulls the latest filings from EDGAR and indexes them in about 30 seconds, then
  answers.
- Drops unsupported claims in a verification pass and tells you what it removed.

## How it works

Under the hood it's a small LangGraph state machine:

```
plan -> retrieve -> grade -> (loop back if not enough) -> synthesize -> verify -> done
```

- **plan** splits your question into focused sub-queries (a separate one per company).
- **retrieve** pulls the top matching chunks from Chroma. If a sub-query names a
  single company, the search is filtered to that company's filings so results
  stay on topic.
- **grade** decides whether there's enough to answer well, and if not, asks for
  the specific missing piece and loops back.
- **synthesize** writes the answer and cites each claim.
- **verify** checks each claim against the retrieved text, removes the ones that
  aren't supported, and attaches a short confidence note.

Retrieval is capped at three rounds so it can't loop forever.

## Stack

- Python 3.10+
- LangGraph + LangChain for the agent
- Groq (Llama 3.3 70B) for the LLM, swappable to a local Ollama model with one env var
- sentence-transformers (BAAI/bge-small-en-v1.5) for embeddings, running locally
- Chroma for the vector store, persisted to disk
- SEC EDGAR via sec-edgar-api, filings parsed with BeautifulSoup
- Streamlit for the UI

## Running it locally (Windows)

```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:

- `GROQ_API_KEY` — grab one from console.groq.com
- `SEC_USER_AGENT` — SEC wants a name and email, e.g. `Jane Doe jane@example.com`

The repo ships with a prebuilt index for ten companies (Apple, Microsoft, Google,
Nvidia, Amazon, Meta, Tesla, Netflix, Oracle, Walmart), so you can go straight to:

```
streamlit run src/app.py
```

If you want to rebuild the index or add companies:

```
python -m src.ingest --tickers AAPL MSFT --count 1
python -m src.index
```

## Project layout

```
src/config.py   config and LLM selection (Groq or Ollama)
src/ingest.py   fetch and cache filings from EDGAR
src/index.py    chunk, embed, and store in Chroma
src/agent.py    the LangGraph agent
src/app.py      the Streamlit UI
scripts/        small runnable checks for each piece
```

## Deploying

It runs on Streamlit Community Cloud. Point it at `src/app.py`, and put
`GROQ_API_KEY`, `LLM_PROVIDER`, and `SEC_USER_AGENT` in the app's Secrets. The
prebuilt index is committed to the repo, so the app has data to answer from the
moment it starts, and the on-demand fetching still works for anything new.

## A few honest caveats

- Groq's free tier has a tokens-per-minute cap. A big multi-hop question on a
  large filing can occasionally hit it, so give it a minute and retry.
- The answer is only as good as the model and the chunks it pulled. On a thin
  corpus the verifier is strict, and it would rather give a bare answer than guess.
- Embeddings run locally, so the first launch downloads the model (~130 MB) and
  needs a little memory.
- This reads filings. It is not financial advice.
