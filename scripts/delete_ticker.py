"""Remove all chunks for a ticker from Chroma, and optionally its cached filings.

Useful for cleaning up a mislabeled/duplicate ticker (e.g. a share-class variant
like 'ORCL-PD' that got indexed alongside the canonical 'ORCL').

Usage:
    python scripts/delete_ticker.py ORCL-PD          # drop from Chroma + cache
    python scripts/delete_ticker.py ORCL-PD --keep-cache
"""

import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import CACHE_DIR, get_embeddings, get_vectorstore


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keep_cache = "--keep-cache" in sys.argv
    if not args:
        print("Usage: python scripts/delete_ticker.py <TICKER> [--keep-cache]")
        sys.exit(1)

    ticker = args[0].upper()
    vectorstore = get_vectorstore(get_embeddings())

    existing = vectorstore.get(where={"ticker": ticker})
    ids = existing.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
        print(f"Deleted {len(ids)} chunks for {ticker} from Chroma.")
    else:
        print(f"No chunks found for {ticker} in Chroma.")

    cache_dir = CACHE_DIR / ticker
    if keep_cache:
        print(f"Left cached filings in place at {cache_dir}.")
    elif cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Removed cached filings directory {cache_dir}.")
    else:
        print(f"No cache directory at {cache_dir}.")


if __name__ == "__main__":
    main()
