import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import get_embeddings, get_vectorstore


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_retrieve.py "<query>"')
        sys.exit(1)

    query = sys.argv[1]
    vectorstore = get_vectorstore(get_embeddings())
    results = vectorstore.similarity_search(query, k=4)

    if not results:
        print("No results. Have you run `python -m src.index` yet?")
        return

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(doc.metadata)
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()
