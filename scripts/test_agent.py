import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import build_agent


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_agent.py "<question>"')
        sys.exit(1)

    question = sys.argv[1]
    agent = build_agent()

    print(f"QUESTION: {question}\n")

    hops = 0
    final_answer = None
    for step in agent.stream({"question": question}):
        for node, update in step.items():
            if node == "plan":
                print("PLAN — sub-queries:")
                for q in update["sub_queries"]:
                    print(f"  - {q}")
                print()
            elif node == "retrieve":
                hops = update["round"]
                print(
                    f"RETRIEVE — round {update['round']}: "
                    f"+{update['retrieved_count']} new chunks "
                    f"({len(update['chunks'])} total)\n"
                )
            elif node == "grade":
                print(f"GRADE — decision: {update['decision']}")
                if update["decision"] == "insufficient" and update.get("reformulated_query"):
                    print(f"        follow-up query (missing piece): {update['reformulated_query']}")
                print()
            elif node == "synthesize":
                print("SYNTHESIZE — drafted answer\n")
            elif node == "verify":
                print("VERIFY —", update.get("verification", ""))
                flagged = update.get("flagged_claims") or []
                if flagged:
                    print("        claims flagged as unsupported:")
                    for c in flagged:
                        print(f"          * {c}")
                else:
                    print("        claims flagged as unsupported: none")
                if update.get("verify_decision") == "needs_more":
                    print(f"        verification gap -> one more retrieval: {update.get('verify_followup')}")
                if update.get("confidence_note"):
                    print(f"        {update['confidence_note']}")
                if update.get("final_answer"):
                    final_answer = update["final_answer"]
                print()

    print("=" * 70)
    print(f"HOP COUNT (retrieval rounds): {hops}")
    print("=" * 70)
    print("FINAL ANSWER:\n")
    print(final_answer)


if __name__ == "__main__":
    main()
