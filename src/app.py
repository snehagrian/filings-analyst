"""Streamlit chat UI for the SEC filings RAG agent."""

import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st
from sec_edgar_api import EdgarClient

from src import ingest as ingest_mod
from src.agent import _load_available_tickers, build_agent
from src.index import get_embeddings, get_vectorstore, index_ticker

_NAME_SUFFIXES = re.compile(
    r"[,\.]|\b(?:inc|incorporated|corp|corporation|co|company|companies|ltd|"
    r"limited|llc|plc|lp|holdings?|group|sa|nv|ag|the|class|common|stock)\b",
    re.I,
)
FILINGS_PER_FORM = 2
MAX_ONDEMAND_COMPANIES = 3

COMPANY_CHIPS = [
    ("Apple", "What are the key risk factors disclosed by Apple?"),
    ("Microsoft", "What are the key risk factors disclosed by Microsoft?"),
    ("Google", "What are the key risk factors disclosed by Google (Alphabet)?"),
    ("Nvidia", "What are the key risk factors disclosed by Nvidia?"),
    ("Amazon", "What are the key risk factors disclosed by Amazon?"),
    ("Tesla", "What are the key risk factors disclosed by Tesla?"),
    ("Netflix", "What are the key risk factors disclosed by Netflix?"),
    ("Meta", "What are the key risk factors disclosed by Meta Platforms (META)?"),
]
COMPARISON_CHIPS = [
    (
        "⚖️ Apple vs Microsoft",
        "Compare the risk factors disclosed by Apple and Microsoft in their latest 10-Ks",
    ),
    (
        "⚖️ Nvidia vs Amazon",
        "Compare the risk factors disclosed by Nvidia and Amazon in their latest 10-Ks",
    ),
]

APP_NAME = "Verdant"
APP_ICON = "🌿"
APP_TAGLINE = "Agentic SEC Filings Analyst"

APP_CSS = """
<style>
/* ---- App canvas: white with soft green radial glows ---- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1100px 480px at 12% -12%, rgba(34,197,94,0.14), transparent 60%),
        radial-gradient(1000px 460px at 100% 0%, rgba(16,185,129,0.12), transparent 55%),
        #ffffff;
}
[data-testid="stHeader"] { background: transparent; }

/* Hide the sidebar entirely (company index removed). */
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* ---- Brand header ---- */
.brand { display: flex; align-items: center; gap: 16px; margin: 4px 0 2px; }
.brand-badge {
    width: 56px; height: 56px; border-radius: 18px;
    display: flex; align-items: center; justify-content: center;
    font-size: 30px;
    background: linear-gradient(145deg, #ffffff, #d7f6e2);
    border: 1px solid rgba(22,163,74,0.35);
    box-shadow: 0 12px 26px rgba(22,163,74,0.30),
                inset 0 1px 0 rgba(255,255,255,0.9);
    animation: floaty 4.5s ease-in-out infinite;
}
.app-title {
    font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1;
    background: linear-gradient(90deg, #065f46, #16a34a, #22c55e);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.app-sub { color: #3f6b52; font-weight: 500; margin-top: 4px; }
.section-label { color: #15803d; font-weight: 700; margin: 6px 0 2px; }

/* ---- Buttons: 3D with green back-glow, pop on hover ---- */
div[data-testid="stButton"] > button {
    border: 1px solid rgba(22,163,74,0.35);
    border-radius: 14px;
    background: linear-gradient(180deg, #ffffff, #e9f9ef);
    color: #065f46; font-weight: 600;
    box-shadow: 0 8px 18px rgba(22,163,74,0.18),
                0 2px 5px rgba(22,163,74,0.15),
                inset 0 1px 0 rgba(255,255,255,0.9);
    transition: transform .16s ease, box-shadow .16s ease, background .16s ease;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px) scale(1.05);
    background: linear-gradient(180deg, #ffffff, #d3f5df);
    box-shadow: 0 18px 38px rgba(22,163,74,0.38),
                0 6px 14px rgba(22,163,74,0.25),
                inset 0 1px 0 rgba(255,255,255,0.95);
}
div[data-testid="stButton"] > button:active {
    transform: translateY(-1px) scale(0.99);
    box-shadow: 0 6px 14px rgba(22,163,74,0.25);
}

/* Suggestion chips (secondary) = rounded pills. */
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] {
    border-radius: 999px;
    padding: 0.32rem 1.05rem;
    font-size: 0.85rem;
}

/* Send (primary) = filled green with a living glow. */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(180deg, #22c55e, #16a34a);
    color: #ffffff; border: 1px solid #15803d; border-radius: 14px;
    box-shadow: 0 12px 26px rgba(22,163,74,0.45),
                inset 0 1px 0 rgba(255,255,255,0.35);
    animation: glowPulse 2.8s ease-in-out infinite;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-3px) scale(1.05);
    background: linear-gradient(180deg, #26d366, #17a94d);
    box-shadow: 0 20px 44px rgba(22,163,74,0.58);
}

/* ---- Composer "taskbar": glowing 3D input ---- */
div[data-testid="stTextArea"] textarea {
    border-radius: 16px !important;
    border: 1px solid rgba(22,163,74,0.35) !important;
    background: #ffffff !important;
    box-shadow: 0 12px 34px rgba(22,163,74,0.22),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
    transition: box-shadow .18s ease, border-color .18s ease;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #16a34a !important;
    box-shadow: 0 16px 44px rgba(22,163,74,0.36),
                0 0 0 3px rgba(34,197,94,0.20) !important;
}

/* ---- Chat bubbles + status: soft green 3D cards ---- */
div[data-testid="stChatMessage"] {
    background: linear-gradient(180deg, #ffffff, #f2fbf5);
    border: 1px solid rgba(22,163,74,0.16);
    border-radius: 16px;
    box-shadow: 0 8px 22px rgba(22,163,74,0.10);
}
div[data-testid="stStatus"] {
    border-radius: 14px;
    border: 1px solid rgba(22,163,74,0.22);
    box-shadow: 0 8px 22px rgba(22,163,74,0.12);
}
div[data-testid="stAlert"] { border-radius: 14px; }

/* ---- Animations ---- */
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 12px 26px rgba(22,163,74,0.42),
                           inset 0 1px 0 rgba(255,255,255,0.35); }
    50%      { box-shadow: 0 16px 36px rgba(22,163,74,0.62),
                           inset 0 1px 0 rgba(255,255,255,0.35); }
}
@keyframes floaty {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-5px); }
}
</style>
"""


@st.cache_resource(show_spinner="Loading vector store…")
def load_vectorstore():
    """Single shared Chroma handle, reused for reads and on-demand writes."""
    return get_vectorstore(get_embeddings())


@st.cache_resource(show_spinner="Loading agent (LLM + embeddings + vector store)...")
def load_agent():
    """Build the compiled LangGraph agent once and reuse it across reruns."""
    return build_agent()


@st.cache_resource(show_spinner="Reading indexed companies...")
def load_indexed_tickers() -> list[str]:
    """Distinct tickers currently present in the Chroma store, sorted."""
    return sorted(_load_available_tickers(load_vectorstore()))


@st.cache_resource(show_spinner=False)
def load_tickers_data() -> dict:
    """SEC's ticker/company map, read from ingest's local cache (no network)."""
    if ingest_mod.TICKERS_CACHE_PATH.exists():
        return json.loads(ingest_mod.TICKERS_CACHE_PATH.read_text(encoding="utf-8"))
    session, rate_limiter, _ = _edgar_clients()
    return ingest_mod.fetch_company_tickers(session, rate_limiter)


def _edgar_clients():
    """Build the (session, rate_limiter, EdgarClient) trio ingest.py expects."""
    user_agent = ingest_mod.require_user_agent()
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    rate_limiter = ingest_mod.RateLimiter(ingest_mod.MAX_REQUESTS_PER_SECOND)
    client = EdgarClient(user_agent=user_agent)
    return session, rate_limiter, client


def _core_name(title: str) -> str:
    """Reduce a company title to its core name, e.g. 'Tesla, Inc.' -> 'tesla'."""
    stripped = _NAME_SUFFIXES.sub(" ", title)
    return re.sub(r"\s+", " ", stripped).strip().lower()


def _short_title(title: str) -> str:
    """Display name: the part before the first comma, e.g. 'Tesla, Inc.' -> 'Tesla'."""
    return title.split(",")[0].strip()


def _canonical_ticker(ticker: str) -> str:
    """Base symbol without a share-class suffix, e.g. 'ORCL-PD' -> 'ORCL'.

    SEC lists preferred/depositary variants (ORCL-PD, BRK-B) that share a CIK
    with the common stock; collapsing to the base avoids indexing a company
    twice under different labels.
    """
    return re.split(r"[-.]", ticker.upper(), maxsplit=1)[0]


def resolve_companies(question: str, tickers_data: dict) -> dict[str, str]:
    """Map companies named in the question to {canonical_ticker: title}.

    Matches an UPPERCASE ticker symbol token (so lowercase English words aren't
    mistaken for tickers) OR a company's core name as a whole word/phrase. Each
    company collapses to ONE canonical ticker, preferring the clean base symbol
    over share-class variants like 'ORCL-PD'.
    """
    q_lower = question.lower()
    upper_tokens = set(re.findall(r"\b[A-Z.\-]{1,8}\b", question))

    resolved: dict[str, str] = {}
    for record in tickers_data.values():
        raw_ticker = str(record["ticker"]).upper()
        canonical = _canonical_ticker(raw_ticker)
        title = record["title"]
        core = _core_name(title)
        by_symbol = raw_ticker in upper_tokens or canonical in upper_tokens
        by_name = len(core) >= 4 and re.search(rf"\b{re.escape(core)}\b", q_lower)
        if not (by_symbol or by_name):
            continue
        if canonical not in resolved or raw_ticker == canonical:
            resolved[canonical] = title
    return resolved


def ensure_companies_indexed(question: str, current_tickers: set[str]):
    """Fetch + index any company named in the question that isn't stored yet.

    Reuses ingest.ingest_ticker and index.index_ticker. Returns
    (added_tickers, notes) where notes are user-facing status/error messages.
    """
    try:
        tickers_data = load_tickers_data()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        return [], [f"Couldn't load the SEC company list: {exc}"]

    resolved = resolve_companies(question, tickers_data)
    missing = {t: title for t, title in resolved.items() if t not in current_tickers}
    if not missing:
        return [], []

    if len(missing) > MAX_ONDEMAND_COMPANIES:
        names = ", ".join(_short_title(t) for t in list(missing.values())[:6])
        return [], [
            f"Your question seems to match many companies ({names}, …). Please "
            "name one or two specific companies (or their tickers) to fetch."
        ]

    try:
        session, rate_limiter, client = _edgar_clients()
    except Exception as exc:  # noqa: BLE001
        return [], [f"Live fetch unavailable: {exc}"]

    ticker_to_cik = ingest_mod.build_ticker_to_cik(tickers_data)
    vectorstore = load_vectorstore()

    added: list[str] = []
    notes: list[str] = []
    for ticker, title in missing.items():
        short = _short_title(title)
        cik = ticker_to_cik.get(ticker)
        if not cik:
            notes.append(f"Couldn't resolve {short} ({ticker}) to an SEC CIK; skipping.")
            continue
        with st.spinner(
            f"Fetching and indexing {short}'s filings… this takes ~30s the first time."
        ):
            try:
                ingest_mod.ingest_ticker(
                    ticker, cik, client, session, rate_limiter, FILINGS_PER_FORM
                )
                n_chunks = index_ticker(ticker, vectorstore)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Couldn't fetch filings for {short} ({ticker}): {exc}")
                continue
        if n_chunks == 0:
            notes.append(f"No recent 10-K/10-Q filings found for {short} ({ticker}).")
        else:
            added.append(ticker)
            notes.append(f"Indexed {n_chunks} chunks for {short} ({ticker}).")
    return added, notes


def stream_answer(agent, question: str):
    """Run the agent, rendering progress live, and return (answer, note, hops).

    Mirrors scripts/test_agent.py's streaming, but writes into a Streamlit
    st.status panel so the user can watch each node as it fires.
    """
    hops = 0
    final_answer = None
    confidence_note = None

    with st.status("Working through your question...", expanded=True) as status:
        for step in agent.stream({"question": question}):
            for node, update in step.items():
                if node == "plan":
                    st.markdown("**Plan — sub-queries:**")
                    for q in update["sub_queries"]:
                        st.markdown(f"- {q}")
                elif node == "retrieve":
                    hops = update["round"]
                    st.markdown(
                        f"**Retrieve — round {update['round']}:** "
                        f"+{update['retrieved_count']} new chunks "
                        f"({len(update['chunks'])} total)"
                    )
                elif node == "grade":
                    st.markdown(f"**Grade — decision:** `{update['decision']}`")
                    if update["decision"] == "insufficient" and update.get("reformulated_query"):
                        st.caption(f"Follow-up (missing piece): {update['reformulated_query']}")
                elif node == "synthesize":
                    st.markdown("**Synthesize —** drafted a cited answer.")
                elif node == "verify":
                    st.markdown(f"**Verify —** {update.get('verification', '')}")
                    flagged = update.get("flagged_claims") or []
                    if flagged:
                        st.markdown("Claims flagged as unsupported:")
                        for c in flagged:
                            st.markdown(f"- {c}")
                    else:
                        st.caption("No claims flagged as unsupported.")
                    if update.get("verify_decision") == "needs_more":
                        st.caption(
                            f"Verification gap → one more retrieval: "
                            f"{update.get('verify_followup')}"
                        )
                    if update.get("confidence_note"):
                        confidence_note = update["confidence_note"]
                    if update.get("final_answer"):
                        final_answer = update["final_answer"]
        status.update(
            label=f"Done — {hops} retrieval round(s).", state="complete", expanded=False
        )

    return final_answer, confidence_note, hops


def render_result(final_answer: str | None, note: str | None, hops: int) -> str:
    """Render the verified answer + note + hop count; return stored markdown.

    The agent appends the confidence note to final_answer; split it back out so
    the note can be shown in its own callout rather than duplicated inline.
    """
    if not final_answer:
        err = "No answer was produced. Is the index built? Run `python -m src.index`."
        st.error(err)
        return err

    body = final_answer
    if note and body.endswith(note):
        body = body[: -len(note)].rstrip()

    st.markdown(body)
    if note:
        st.info(note)
    st.caption(f"Hop count (retrieval rounds): {hops}")

    stored = body
    if note:
        stored += f"\n\n> {note}"
    stored += f"\n\n*Hop count (retrieval rounds): {hops}*"
    return stored


def _prefill_composer(question: str) -> None:
    """Chip callback: drop a ready-made question into the composer (no submit)."""
    st.session_state.composer = question


def _submit_composer() -> None:
    """Send callback: capture the composed question and clear the box."""
    text = st.session_state.get("composer", "").strip()
    if text:
        st.session_state.pending_question = text
    st.session_state.composer = ""


def render_chip_row(chips: list[tuple[str, str]], per_row: int = 4) -> None:
    """Render clickable suggestion chips that prefill (not submit) the composer."""
    for start in range(0, len(chips), per_row):
        row = chips[start : start + per_row]
        cols = st.columns(per_row)
        for col, (label, question) in zip(cols, row):
            with col:
                st.button(
                    label,
                    key=f"chip_{label}",
                    type="secondary",
                    on_click=_prefill_composer,
                    args=(question,),
                )


def handle_question(question: str, known_tickers: set[str]) -> None:
    """Run one full chat turn: on-demand index if needed, then stream the agent."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        added, notes = ensure_companies_indexed(question, known_tickers)
        for note_line in notes:
            st.caption(note_line)

        agent = load_agent()
        if added:
            load_vectorstore.clear()
            load_indexed_tickers.clear()
            load_agent.clear()
            agent = load_agent()

        final_answer, note, hops = stream_answer(agent, question)
        stored = render_result(final_answer, note, hops)
        st.session_state.messages.append({"role": "assistant", "content": stored})


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} · {APP_TAGLINE}", page_icon=APP_ICON, layout="wide"
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="brand">
            <div class="brand-badge">{APP_ICON}</div>
            <div>
                <div class="app-title">{APP_NAME}</div>
                <div class="app-sub">{APP_TAGLINE} · multi-hop, self-verifying RAG
                over SEC 10-K / 10-Q filings</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tickers = load_indexed_tickers()

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("composer", "")
    st.session_state.setdefault("pending_question", None)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        handle_question(question, set(tickers))

    st.markdown(
        '<div class="section-label">Try one — click to prefill, edit, then Send</div>',
        unsafe_allow_html=True,
    )
    render_chip_row(COMPANY_CHIPS)
    render_chip_row(COMPARISON_CHIPS, per_row=2)

    st.text_area(
        "Your question",
        key="composer",
        placeholder="Ask about any public company's filings…",
        height=80,
    )
    st.button("Send", type="primary", on_click=_submit_composer)


if __name__ == "__main__":
    main()
