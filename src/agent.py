"""Self-correcting, multi-hop RAG agent over indexed SEC filings."""

import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from src.config import get_llm
from src.index import get_embeddings, get_vectorstore

MAX_ROUNDS = 3
TOP_K = 4
MAX_CONTEXT_CHUNKS = 16


class AgentState(TypedDict, total=False):
    """State threaded through the graph."""

    question: str
    sub_queries: list[str]
    chunks: list[Document]
    round: int
    decision: str
    reformulated_query: str
    retrieved_count: int
    answer: str
    verification: str
    flagged_claims: list[str]
    confidence_note: str
    verify_decision: str
    verify_followup: str
    final_answer: str


def _chunk_key(doc: Document) -> tuple:
    """Identity of a chunk, used to de-duplicate across retrieval rounds."""
    m = doc.metadata
    return (m.get("accession"), m.get("form"), m.get("chunk_index"))


def _select_context(chunks: list[Document], limit: int = MAX_CONTEXT_CHUNKS) -> list[Document]:
    """Pick up to `limit` chunks, round-robin by ticker.

    Round-robin guarantees every retrieved company is represented within the cap,
    so multi-hop evidence (e.g. both Apple and Microsoft) survives truncation.
    For a single-company question this preserves the original front-to-back order.
    """
    groups: dict[str, list[Document]] = {}
    order: list[str] = []
    for doc in chunks:
        ticker = doc.metadata.get("ticker")
        if ticker not in groups:
            groups[ticker] = []
            order.append(ticker)
        groups[ticker].append(doc)

    selected: list[Document] = []
    i = 0
    while len(selected) < limit and any(groups[t] for t in order):
        ticker = order[i % len(order)]
        if groups[ticker]:
            selected.append(groups[ticker].pop(0))
        i += 1
    return selected


def _format_context(chunks: list[Document], limit: int = MAX_CONTEXT_CHUNKS) -> str:
    """Render selected chunks as labeled excerpts with their citation header."""
    lines = []
    for doc in _select_context(chunks, limit):
        m = doc.metadata
        cite = f"{m.get('ticker')} {m.get('form')} {m.get('filing_date')}"
        lines.append(f"[{cite}]\n{doc.page_content}")
    return "\n\n".join(lines)


def _retrieved_sources(chunks: list[Document]) -> str:
    """Distinct list of source filings already retrieved, for gap detection."""
    seen: list[str] = []
    for doc in chunks:
        m = doc.metadata
        src = f"{m.get('ticker')} {m.get('form')} ({m.get('filing_date')})"
        if src not in seen:
            seen.append(src)
    return ", ".join(seen) if seen else "none"


_TICKER_ALIASES = {
    "GOOGL": ("google", "alphabet"),
}
_TICKERS_CACHE_PATH = Path("data/cache/company_tickers.json")
_NAME_SUFFIXES = re.compile(
    r"[,\.]|\b(?:inc|incorporated|corp|corporation|co|company|companies|ltd|"
    r"limited|llc|plc|lp|holdings?|group|sa|nv|ag|the|class|common|stock)\b",
    re.I,
)


def _company_core_name(title: str) -> str:
    """Reduce a company title to its core name, e.g. 'ORACLE CORP' -> 'oracle'."""
    stripped = _NAME_SUFFIXES.sub(" ", title)
    return re.sub(r"\s+", " ", stripped).strip().lower()


def _load_ticker_titles() -> dict[str, str]:
    """Map ticker -> SEC title from ingest's cached company_tickers.json."""
    if not _TICKERS_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(_TICKERS_CACHE_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return {str(r["ticker"]).upper(): r["title"] for r in data.values()}


def _build_alias_map(available: set[str]) -> dict[str, tuple[str, ...]]:
    """Build ticker -> name aliases for the tickers currently in the store.

    Combines hardcoded nickname overrides with names derived from SEC titles,
    so any company (pre-indexed or fetched on demand) can be recognized by name.
    """
    titles = _load_ticker_titles()
    alias_map: dict[str, tuple[str, ...]] = {}
    for ticker in available:
        aliases = list(_TICKER_ALIASES.get(ticker, ()))
        title = titles.get(ticker) or titles.get(ticker.split("-")[0])
        if title:
            core = _company_core_name(title)
            if len(core) >= 4:
                aliases.append(core)
                first = core.split()[0]
                if len(first) >= 4:
                    aliases.append(first)
        alias_map[ticker] = tuple(dict.fromkeys(aliases))
    return alias_map


def _load_available_tickers(vectorstore) -> set[str]:
    """Distinct tickers actually present in the vector store.

    Used so we only ever filter to a ticker that has chunks; an empty result
    would otherwise silently starve a retrieval round.
    """
    try:
        data = vectorstore.get(include=["metadatas"])
    except Exception:
        return set()
    return {m.get("ticker") for m in data.get("metadatas", []) if m.get("ticker")}


def _detect_ticker(
    query: str, available: set[str], alias_map: dict[str, tuple[str, ...]]
) -> str | None:
    """Return the single ticker a sub-query targets, else None.

    Matches the ticker symbol or a name alias as a whole word. Returns None when
    zero or MORE THAN ONE company is referenced (e.g. an "Apple and Microsoft"
    query), so those fall back to an unfiltered global search.
    """
    q = query.lower()
    matched = set()
    for ticker in available:
        terms = (ticker.lower(), *alias_map.get(ticker, ()))
        if any(re.search(rf"\b{re.escape(term)}\b", q) for term in terms):
            matched.add(ticker)
    return next(iter(matched)) if len(matched) == 1 else None


def _finalize_answer(
    draft: str, cleaned: str, flagged: list[str], gaps: bool, round_: int
) -> tuple[str, str]:
    """Build the final answer with unsupported claims removed, plus a note.

    The verifier's VERDICT word is unreliable, so removal is driven by whether
    there are flagged claims (or a gaps verdict) AND whether the cleaned rewrite
    actually differs from the draft. The note reports honestly: it only claims
    removal when the answer body genuinely changed.
    """
    has_unsupported = gaps or bool(flagged)
    cleaned_ok = bool(cleaned) and cleaned.upper() != "NONE"
    cleaned_empty = bool(cleaned) and cleaned.upper() == "NONE"

    if has_unsupported and cleaned_ok:
        body = cleaned
    elif has_unsupported and cleaned_empty:
        body = (
            "After verification, no claims in the draft could be supported "
            "by the retrieved excerpts."
        )
    else:
        body = draft

    limit_suffix = " (Retrieval round limit reached.)" if round_ >= MAX_ROUNDS else ""
    removed_happened = body.strip() != draft.strip()

    if removed_happened:
        n = len(flagged)
        if n:
            verb = "was" if n == 1 else "were"
            note = (
                f"Note: {n} unsupported claim{'' if n == 1 else 's'} {verb} "
                "removed during verification. Removed: " + "; ".join(flagged) + "."
            )
        else:
            note = "Note: unsupported claims were removed during verification."
        note += limit_suffix
    elif has_unsupported and flagged:
        note = (
            "Confidence note: some claims could not be fully verified and should "
            "be treated with caution: " + "; ".join(flagged) + "." + limit_suffix
        )
    else:
        note = "Confidence note: all claims were verified against the retrieved filings."

    return f"{body}\n\n{note}", note


def build_agent():
    """Construct and compile the LangGraph agent.

    The LLM, embeddings, and vector store are loaded once here and closed over
    by the node functions, so the embedding model is not reloaded each round.
    """
    llm = get_llm()
    vectorstore = get_vectorstore(get_embeddings())
    available_tickers = _load_available_tickers(vectorstore)
    alias_map = _build_alias_map(available_tickers)

    def plan(state: AgentState) -> dict:
        """Break the question into 1-3 focused retrieval sub-queries."""
        prompt = (
            "You are a financial research assistant. Break the user's question "
            "into 1 to 3 focused search queries for retrieving passages from SEC "
            "filings (10-K/10-Q). If the question compares multiple companies or "
            "periods, create a separate query for EACH one. Return ONLY the "
            "queries, one per line, with no numbering, bullets, or extra text.\n\n"
            f"Question: {state['question']}"
        )
        raw = llm.invoke(prompt).content
        sub_queries = []
        for line in str(raw).splitlines():
            cleaned = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
            if cleaned:
                sub_queries.append(cleaned)
        sub_queries = sub_queries[:3] or [state["question"]]
        return {"sub_queries": sub_queries, "round": 0}

    def retrieve(state: AgentState) -> dict:
        """Query Chroma for each sub-query and accumulate de-duplicated chunks."""
        existing = list(state.get("chunks", []))
        seen = {_chunk_key(d) for d in existing}

        new_count = 0
        for query in state["sub_queries"]:
            ticker = _detect_ticker(query, available_tickers, alias_map)
            search_kwargs = {"filter": {"ticker": ticker}} if ticker else {}
            for doc in vectorstore.similarity_search(query, k=TOP_K, **search_kwargs):
                key = _chunk_key(doc)
                if key not in seen:
                    seen.add(key)
                    existing.append(doc)
                    new_count += 1

        return {
            "chunks": existing,
            "round": state.get("round", 0) + 1,
            "retrieved_count": new_count,
        }

    def grade(state: AgentState) -> dict:
        """Judge sufficiency and, for multi-hop, target any MISSING source."""
        context = _format_context(state["chunks"])
        sources = _retrieved_sources(state["chunks"])
        prompt = (
            "You are evaluating retrieved context for a question about SEC "
            "filings, gathering evidence across MULTIPLE filings when needed.\n\n"
            f"Question: {state['question']}\n\n"
            f"Sources already retrieved: {sources}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Decide whether the context covers everything needed to answer well. "
            "Mark 'insufficient' ONLY if the QUESTION itself names or requires a "
            "specific company, time period, or topic that is genuinely NOT among "
            "the sources already retrieved. Never introduce companies the "
            "question did not mention. If the question concerns a single company "
            "and that company's filings are already in the sources, answer "
            "'sufficient'. When the question compares companies or periods, EACH "
            "one named in the question must appear in the sources; if one is "
            "missing, the follow-up query must target that MISSING "
            "company/period specifically. Once EVERY company and period named in "
            "the question appears in the sources, answer 'sufficient' even if the "
            "coverage is imperfect.\n\n"
            "Respond in EXACTLY this format:\n"
            "DECISION: sufficient OR insufficient\n"
            "QUERY: <if insufficient, a natural-language description of the "
            "missing company's risk factors or topics, naming the company in "
            "plain English (e.g. 'Microsoft cybersecurity and competition risk "
            "factors'). Describe the subject matter as it would read in the "
            "filing's prose. Do NOT use form types, filing dates, accession "
            "numbers, or section labels like 'Item 1A'. Otherwise NONE>"
        )
        raw = str(llm.invoke(prompt).content)

        decision = "insufficient" if re.search(r"insufficient", raw, re.I) else "sufficient"
        query_match = re.search(r"QUERY:\s*(.+)", raw, re.I)
        reformulated = query_match.group(1).strip() if query_match else ""

        out = {"decision": decision, "reformulated_query": reformulated}
        if decision == "insufficient" and reformulated and reformulated.upper() != "NONE":
            out["sub_queries"] = [reformulated]
        return out

    def synthesize(state: AgentState) -> dict:
        """Write a cited answer grounded in the retrieved excerpts."""
        context = _format_context(state["chunks"])
        prompt = (
            "Answer the question using ONLY the provided filing excerpts. Cite "
            "every claim with its source in the form (TICKER FORM FILING_DATE). "
            "If the excerpts do not contain the answer, say so plainly.\n\n"
            f"Question: {state['question']}\n\n"
            f"Excerpts:\n{context}\n\n"
            "Answer:"
        )
        answer = str(llm.invoke(prompt).content)
        return {"answer": answer}

    def verify(state: AgentState) -> dict:
        """Check each drafted claim against the retrieved evidence.

        Flags unsupported claims, may trigger one more targeted retrieval, and
        attaches a confidence note to the final output.
        """
        context = _format_context(state["chunks"])
        prompt = (
            "You are a fact-checker. Check EACH claim in the draft answer against "
            "the filing excerpts. A claim is SUPPORTED only if the excerpts back "
            "it up. List any claims that are unsupported or only partially "
            "supported.\n\n"
            f"Question: {state['question']}\n\n"
            f"Draft answer:\n{state['answer']}\n\n"
            f"Filing excerpts:\n{context}\n\n"
            "Respond in EXACTLY this format, with CLEANED last:\n"
            "VERDICT: complete OR gaps\n"
            "UNSUPPORTED: <short descriptions of unsupported claims separated by "
            "semicolons, or NONE>\n"
            "MISSING: <a targeted search query for the most important missing "
            "evidence, or NONE>\n"
            "CLEANED:\n"
            "<the draft answer with every unsupported or only-partially-supported "
            "sentence REMOVED, keeping only statements grounded in the excerpts "
            "and preserving their (TICKER FORM FILING_DATE) citations. Do not add "
            "new claims or reword supported ones. If nothing is supported, write "
            "exactly: NONE>"
        )
        raw = str(llm.invoke(prompt).content)

        verdict_match = re.search(r"VERDICT:\s*(\w+)", raw, re.I)
        verdict = verdict_match.group(1).lower() if verdict_match else "complete"
        gaps = verdict.startswith("gap")

        unsupported_match = re.search(r"UNSUPPORTED:\s*(.+)", raw, re.I)
        unsupported_raw = unsupported_match.group(1).strip() if unsupported_match else "NONE"
        _labels = ("missing:", "cleaned:", "verdict:", "unsupported:")
        flagged = (
            []
            if unsupported_raw.upper() == "NONE"
            else [
                s.strip()
                for s in re.split(r"[;\n]", unsupported_raw)
                if s.strip() and not s.strip().lower().startswith(_labels)
            ]
        )

        missing_match = re.search(r"MISSING:\s*(.+)", raw, re.I)
        missing = missing_match.group(1).strip() if missing_match else "NONE"
        has_missing = missing.upper() != "NONE"

        cleaned_match = re.search(r"CLEANED:\s*(.+)", raw, re.I | re.S)
        cleaned = cleaned_match.group(1).strip() if cleaned_match else ""

        verification = (
            f"Checked the draft against {len(state['chunks'])} retrieved chunks; "
            f"verdict: {'gaps' if gaps else 'complete'}."
        )

        if gaps and has_missing and state["round"] < MAX_ROUNDS:
            return {
                "verification": verification + f" Seeking missing evidence: {missing}",
                "flagged_claims": flagged,
                "verify_decision": "needs_more",
                "verify_followup": missing,
                "sub_queries": [missing],
            }

        final_answer, note = _finalize_answer(
            state["answer"], cleaned, flagged, gaps, state["round"]
        )
        return {
            "verification": verification,
            "flagged_claims": flagged,
            "confidence_note": note,
            "verify_decision": "ok",
            "final_answer": final_answer,
        }

    def route(state: AgentState) -> str:
        """Loop back to retrieve while insufficient and under the round cap."""
        if state["decision"] == "insufficient" and state["round"] < MAX_ROUNDS:
            return "retrieve"
        return "synthesize"

    def verify_route(state: AgentState) -> str:
        """After verify, take one more targeted hop if support is missing."""
        if state.get("verify_decision") == "needs_more" and state["round"] < MAX_ROUNDS:
            return "retrieve"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("synthesize", synthesize)
    graph.add_node("verify", verify)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", route, {"retrieve": "retrieve", "synthesize": "synthesize"}
    )
    graph.add_edge("synthesize", "verify")
    graph.add_conditional_edges(
        "verify", verify_route, {"retrieve": "retrieve", END: END}
    )

    return graph.compile()


def answer_question(question: str) -> AgentState:
    """Run the agent end-to-end and return the final state.

    The verified output (draft answer plus confidence note) is in
    state["final_answer"].
    """
    agent = build_agent()
    return agent.invoke({"question": question})
