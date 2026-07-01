import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_llm


def main() -> None:
    llm = get_llm()
    response = llm.invoke("Reply with the single word: OK")
    print(response.content)


if __name__ == "__main__":
    main()
