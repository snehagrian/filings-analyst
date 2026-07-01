"""Chunk cached filings, embed them, and upsert into a persistent Chroma store."""

import json
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CACHE_DIR = Path("data/cache")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "filings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def load_cached_filings() -> list[dict]:
    """Pair each cached filing .txt with its matching .json metadata."""
    filings = []
    for txt_path in sorted(CACHE_DIR.glob("*/*.txt")):
        meta_path = txt_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        filings.append(
            {
                "text": txt_path.read_text(encoding="utf-8"),
                "metadata": json.loads(meta_path.read_text(encoding="utf-8")),
            }
        )
    return filings


def chunk_filing(
    text: str, metadata: dict, splitter: RecursiveCharacterTextSplitter
) -> tuple[list[Document], list[str]]:
    """Split a filing into chunks with stable IDs, so re-indexing upserts in place."""
    chunks = splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata={**metadata, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    ids = [f"{metadata['accession']}_{metadata['form']}_{i}" for i in range(len(chunks))]
    return documents, ids


def index_ticker(ticker: str, vectorstore: Chroma | None = None) -> int:
    """Embed and upsert ONLY one ticker's cached filings into Chroma.

    Mirrors build_index() but scoped to a single company, so the app's
    on-demand path can make a just-fetched company searchable without
    re-embedding the entire corpus. Reuses the same chunking and stable IDs,
    so it upserts idempotently. Returns the number of chunks added.
    """
    ticker = ticker.upper()
    if vectorstore is None:
        vectorstore = get_vectorstore(get_embeddings())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    total = 0
    for filing in load_cached_filings():
        metadata = filing["metadata"]
        if str(metadata.get("ticker", "")).upper() != ticker:
            continue
        documents, ids = chunk_filing(filing["text"], metadata, splitter)
        vectorstore.add_documents(documents, ids=ids)
        total += len(documents)
    return total


def build_index() -> None:
    filings = load_cached_filings()
    if not filings:
        print("No cached filings found in data/cache/. Run `python -m src.ingest` first.")
        return

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    total_chunks = 0
    for filing in filings:
        metadata = filing["metadata"]
        documents, ids = chunk_filing(filing["text"], metadata, splitter)
        vectorstore.add_documents(documents, ids=ids)
        total_chunks += len(documents)
        print(
            f"[{metadata['ticker']}] indexed {metadata['form']} "
            f"{metadata['accession']} ({len(documents)} chunks)"
        )

    print(f"Indexed {len(filings)} filings into {total_chunks} chunks at ./{CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
