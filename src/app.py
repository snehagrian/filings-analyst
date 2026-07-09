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
import streamlit.components.v1 as components
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

_COMPANY_NICKNAMES = {"google", "alphabet", "meta", "facebook"}
_NAME_STOPWORDS = {
    "compare", "what", "whats", "how", "why", "when", "who", "which", "the", "a",
    "an", "tell", "show", "give", "list", "find", "explain", "describe", "is",
    "are", "do", "does", "did", "and", "or", "vs", "versus", "between", "with",
    "their", "its", "latest", "recent", "risk", "risks", "factor", "factors",
    "report", "reports", "filing", "filings", "revenue", "revenues", "profit",
    "profits", "growth", "strategy", "overview", "summary", "performance",
    "business", "company", "companies", "please", "also", "about", "for", "in",
    "of", "on", "to", "i", "me", "ai", "sec", "us", "usa", "u.s", "america",
    "q1", "q2", "q3", "q4", "fy", "ceo", "cfo", "inc", "corp",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

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
@property --a { syntax: "<angle>"; inherits: false; initial-value: 0deg; }

/* ---- Dark canvas; the animated network + globe is drawn on a full-page
   canvas injected behind the content, so containers stay transparent ---- */
html, body, .stApp { background: #05130d !important; }
[data-testid="stAppViewContainer"] {
    background: transparent !important;
    position: relative;
    z-index: 1;
    color: #dbeee4;
}
[data-testid="stMain"], [data-testid="stMainBlockContainer"], .block-container {
    background: transparent !important;
}
[data-testid="stHeader"] { background: transparent; }
.stApp, .stMarkdown, p, li, label, span, h1, h2, h3, h4 { color: #dbeee4; }
#verdant-bg { position: fixed; inset: 0; z-index: 0; pointer-events: none; }

/* Tighten layout toward a single screen and collapse the hidden bg component. */
.block-container { padding-top: 1.4rem !important; padding-bottom: 1rem !important; }
[data-testid="stElementContainer"]:has(iframe) {
    height: 0 !important; min-height: 0 !important; margin: 0 !important;
    overflow: hidden !important;
}

/* ---- Welcome hero ---- */
.hero { text-align: center; margin: 12px 0 4px; }
.hero-head {
    font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", sans-serif;
    font-size: 2.5rem; font-weight: 800; letter-spacing: -0.025em; line-height: 1.08;
    background: linear-gradient(90deg, #eafff6, #a7f3d0, #6ee7b7);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    filter: drop-shadow(0 0 28px rgba(52,211,153,0.35));
}
.hero-desc {
    font-family: ui-monospace, "SFMono-Regular", "JetBrains Mono", Menlo, Consolas, monospace;
    font-size: 1.0rem; font-weight: 600; max-width: 660px; margin: 12px auto 0;
    line-height: 1.55; letter-spacing: 0.01em;
    background: linear-gradient(90deg, #34d399, #4ade80, #86efac);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-gap { height: 26vh; min-height: 130px; }

/* Keep the suggestion chips compact and tight above the composer. */
[data-testid="stHorizontalBlock"] { gap: 0.4rem !important; }
[data-testid="stTextArea"] { margin-top: 2px; }

/* Hide the sidebar entirely (company index removed). */
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* ---- Brand header ---- */
.brand { display: flex; align-items: center; gap: 16px; margin: 4px 0 2px; }
.brand-badge {
    width: 58px; height: 58px; border-radius: 18px;
    display: flex; align-items: center; justify-content: center;
    font-size: 30px;
    background: radial-gradient(circle at 30% 25%, #0e3f2c, #05130d);
    border: 1px solid rgba(16,185,129,0.45);
    box-shadow: 0 0 24px rgba(16,185,129,0.45),
                inset 0 1px 0 rgba(255,255,255,0.08);
    animation: floaty 4.5s ease-in-out infinite;
}
.app-title {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1;
    background: linear-gradient(90deg, #34d399, #10b981, #2dd4bf, #a7f3d0);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    filter: drop-shadow(0 0 26px rgba(16,185,129,0.35));
}
.app-sub { color: #8fc9b0; font-weight: 500; margin-top: 4px; }
.section-label { color: #6ee7b7; font-weight: 700; margin: 8px 0 2px; letter-spacing: .02em; }

/* ---- Buttons: dark glass, emerald edge + glow, pop on hover ---- */
div[data-testid="stButton"] > button {
    border: 1px solid rgba(16,185,129,0.40);
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(13,45,33,0.92), rgba(7,26,18,0.92));
    color: #c9f5e0; font-weight: 600;
    box-shadow: 0 8px 22px rgba(0,0,0,0.5),
                0 0 14px rgba(16,185,129,0.18),
                inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform .16s ease, box-shadow .16s ease, background .16s ease, border-color .16s ease;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px) scale(1.05);
    border-color: rgba(52,211,153,0.85);
    background: linear-gradient(180deg, rgba(16,60,44,0.96), rgba(9,34,24,0.96));
    box-shadow: 0 14px 34px rgba(0,0,0,0.55),
                0 0 28px rgba(16,185,129,0.5);
}
div[data-testid="stButton"] > button:active {
    transform: translateY(-1px) scale(0.99);
    box-shadow: 0 0 16px rgba(16,185,129,0.35);
}

/* Suggestion chips (secondary) = rounded pills. */
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] {
    border-radius: 999px;
    padding: 0.32rem 1.05rem;
    font-size: 0.85rem;
}

/* Send (primary) = filled emerald with a living glow. */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(180deg, #10b981, #059669);
    color: #04160e; border: 1px solid #34d399; border-radius: 14px; font-weight: 700;
    box-shadow: 0 0 26px rgba(16,185,129,0.55),
                inset 0 1px 0 rgba(255,255,255,0.25);
    animation: glowPulse 2.8s ease-in-out infinite;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-3px) scale(1.05);
    background: linear-gradient(180deg, #14c98d, #06a271);
    box-shadow: 0 0 42px rgba(52,211,153,0.8);
}

/* ---- Composer: animated glowing gradient "striplight" border ---- */
/* Border lives ONLY on the outer wrapper; inner layers are transparent so no
   second square box shows over the rounded corners. */
div[data-testid="stTextArea"] div[data-baseweb="textarea"] {
    border: 3px solid transparent !important;
    border-radius: 18px !important;
    background:
        linear-gradient(#07160f, #07160f) padding-box,
        conic-gradient(from var(--a),
            #052e16, #16a34a, #22c55e, #4ade80, #86efac, #22c55e, #15803d, #052e16) border-box !important;
    animation: spin 4s linear infinite;
    box-shadow: 0 0 34px rgba(34,197,94,0.42), 0 0 70px rgba(22,163,74,0.22);
    overflow: hidden;
}
div[data-testid="stTextArea"] div[data-baseweb="base-input"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 15px !important;
}
div[data-testid="stTextArea"] textarea {
    background: transparent !important;
    color: #e6f4ec !important;
    border: none !important;
    border-radius: 15px !important;
}
div[data-testid="stTextArea"] textarea::placeholder { color: #5f8a75 !important; }
div[data-testid="stTextArea"] textarea:focus { box-shadow: none !important; outline: none !important; }

/* ---- Chat bubbles + status: dark glass cards ---- */
div[data-testid="stChatMessage"] {
    background: linear-gradient(180deg, rgba(12,32,24,0.92), rgba(7,22,15,0.92));
    border: 1px solid rgba(16,185,129,0.20);
    border-radius: 16px;
    box-shadow: 0 8px 26px rgba(0,0,0,0.45), 0 0 18px rgba(16,185,129,0.10);
}
/* ---- "Thinking" stream: sleek dark-green terminal with a live cursor ---- */
div[data-testid="stStatus"] {
    border-radius: 14px;
    border: 1px solid rgba(16,185,129,0.28);
    border-left: 3px solid #10b981;
    background: linear-gradient(180deg, rgba(9,28,20,0.92), rgba(6,20,14,0.94));
    box-shadow: 0 0 26px rgba(16,185,129,0.16), inset 0 0 24px rgba(16,185,129,0.05);
    font-family: ui-monospace, "SFMono-Regular", "JetBrains Mono", Menlo, Consolas, monospace;
    overflow: hidden;
}
div[data-testid="stStatus"] summary { color: #6ee7b7 !important; font-weight: 700; }
div[data-testid="stStatus"] [data-testid="stExpanderDetails"] p,
div[data-testid="stStatus"] [data-testid="stExpanderDetails"] li {
    color: #a7f3d0 !important; font-size: 0.86rem; line-height: 1.55;
    font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
div[data-testid="stStatus"] [data-testid="stExpanderDetails"] strong { color: #5eead4 !important; }
div[data-testid="stStatus"] [data-testid="stExpanderDetails"] code {
    color: #d7fff0 !important; background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.25); border-radius: 6px; padding: 0 5px;
}
/* moving scan-line accent along the top, like a speedometer sweep */
div[data-testid="stStatus"]::before {
    content: ""; position: absolute; top: 0; left: -40%; height: 2px; width: 40%;
    background: linear-gradient(90deg, transparent, #34d399, #a7f3d0, transparent);
    animation: scan 1.8s linear infinite; opacity: 0.9;
}
/* blinking cursor at the current line */
div[data-testid="stStatus"] [data-testid="stExpanderDetails"]::after {
    content: "▍"; color: #34d399; margin-left: 2px;
    animation: caret 1s steps(1) infinite;
}
div[data-testid="stAlert"] { border-radius: 14px; }

/* ---- Animations ---- */
@keyframes spin { to { --a: 360deg; } }
@keyframes scan { 0% { left: -40%; } 100% { left: 100%; } }
@keyframes caret { 50% { opacity: 0; } }
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 22px rgba(16,185,129,0.45),
                           inset 0 1px 0 rgba(255,255,255,0.25); }
    50%      { box-shadow: 0 0 42px rgba(52,211,153,0.78),
                           inset 0 1px 0 rgba(255,255,255,0.25); }
}
@keyframes floaty {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-5px); }
}
</style>
"""

# Full-page animated background: a drifting particle network plus a central
# particle globe (the "AI assistant"), injected onto the parent page from a
# zero-height component iframe. The globe energizes while the app is running.
BACKGROUND_HTML = """
<script>
(function () {
  const pdoc = window.parent.document, pwin = window.parent;
  const gen = (pwin.__verdantGen = (pwin.__verdantGen || 0) + 1);

  let cv = pdoc.getElementById('verdant-bg');
  if (cv) cv.remove();
  cv = pdoc.createElement('canvas');
  cv.id = 'verdant-bg';
  cv.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;z-index:0;pointer-events:none;';
  pdoc.body.appendChild(cv);
  const ctx = cv.getContext('2d', { alpha: false });

  let W, H, DPR;
  function resize() {
    DPR = Math.min(1.5, pwin.devicePixelRatio || 1);
    W = pwin.innerWidth; H = pwin.innerHeight;
    cv.width = W * DPR; cv.height = H * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  resize();
  pwin.addEventListener('resize', resize);

  const TAU = 6.283185;

  const N = Math.max(38, Math.min(72, Math.floor(W * H / 24000)));
  const pts = [];
  for (let i = 0; i < N; i++)
    pts.push({ x: Math.random() * W, y: Math.random() * H,
               vx: (Math.random() - 0.5) * 0.22, vy: (Math.random() - 0.5) * 0.22 });
  const D1SQ = 125 * 125;

  const G = 190;
  const gx = new Float32Array(G), gy = new Float32Array(G), gz = new Float32Array(G);
  const px = new Float32Array(G), py = new Float32Array(G), pd = new Float32Array(G);
  const inc = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < G; i++) {
    const y = 1 - (i / (G - 1)) * 2, r = Math.sqrt(Math.max(0, 1 - y * y)), phi = i * inc;
    gx[i] = Math.cos(phi) * r; gy[i] = y; gz[i] = Math.sin(phi) * r;
  }

  let energy = 0, t = 0, last = 0, rot = 0;
  function busy() {
    try {
      const app = pdoc.querySelector('[data-testid="stApp"]');
      const s = app && (app.getAttribute('data-test-script-state') ||
                        app.getAttribute('data-teststate') || '');
      if (s && /run/i.test(s)) return true;
      if (pdoc.querySelector('.stSpinner')) return true;
    } catch (e) {}
    return false;
  }

  function frame(now) {
    if (gen !== pwin.__verdantGen) return;
    pwin.requestAnimationFrame(frame);
    const dt = Math.min(2.4, (now - last) / 16.7) || 1; last = now;
    t += 0.016 * dt;
    energy += ((busy() ? 1 : 0) - energy) * 0.05;

    ctx.fillStyle = 'rgba(5,19,13,0.30)';
    ctx.fillRect(0, 0, W, H);

    // ambient network
    for (let i = 0; i < N; i++) {
      const p = pts[i]; p.x += p.vx * dt; p.y += p.vy * dt;
      if (p.x < 0) p.x += W; else if (p.x > W) p.x -= W;
      if (p.y < 0) p.y += H; else if (p.y > H) p.y -= H;
    }
    ctx.strokeStyle = 'rgba(94,234,212,0.10)'; ctx.lineWidth = 1; ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const a = pts[i];
      for (let j = i + 1; j < N; j++) {
        const b = pts[j], dx = a.x - b.x, dy = a.y - b.y;
        if (dx * dx + dy * dy < D1SQ) { ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); }
      }
    }
    ctx.stroke();
    ctx.fillStyle = 'rgba(110,231,183,0.4)'; ctx.beginPath();
    for (let i = 0; i < N; i++) { const p = pts[i]; ctx.moveTo(p.x + 1.2, p.y); ctx.arc(p.x, p.y, 1.2, 0, TAU); }
    ctx.fill();

    // ---- particle globe (tiny, crisp points on a rotating sphere) ----
    rot += (0.0022 + energy * 0.012) * dt;
    const cx = W / 2, cy = H * 0.40 - energy * 46 + Math.sin(t * 1.3) * (4 + energy * 12);
    const R = Math.min(W, H) * 0.16 * (1 + energy * 0.20);
    const cR = Math.cos(rot), sR = Math.sin(rot), ct = Math.cos(0.5), stt = Math.sin(0.5);
    for (let i = 0; i < G; i++) {
      const x = gx[i] * cR - gz[i] * sR, z = gx[i] * sR + gz[i] * cR;
      const y2 = gy[i] * ct - z * stt, z2 = gy[i] * stt + z * ct;
      const jit = 1 + energy * Math.sin(t * 6 + gx[i] * 9) * 0.05;
      px[i] = cx + x * R * jit; py[i] = cy + y2 * R * jit; pd[i] = (z2 + 1) * 0.5;
    }
    const limSq = (R * 0.30) * (R * 0.30);
    ctx.strokeStyle = 'rgba(52,211,153,0.15)'; ctx.lineWidth = 0.6; ctx.beginPath();
    for (let i = 0; i < G; i++) {
      if (pd[i] < 0.34) continue;
      const ax = px[i], ay = py[i];
      for (let j = i + 1; j < G; j++) {
        if (pd[j] < 0.34) continue;
        const dx = ax - px[j], dy = ay - py[j];
        if (dx * dx + dy * dy < limSq) { ctx.moveTo(ax, ay); ctx.lineTo(px[j], py[j]); }
      }
    }
    ctx.stroke();
    for (let i = 0; i < G; i++) {
      const d = pd[i];
      ctx.fillStyle = 'rgba(188,255,224,' + (0.16 + d * 0.62) + ')';
      ctx.beginPath(); ctx.arc(px[i], py[i], 0.3 + d * 1.05, 0, TAU); ctx.fill();
    }
    const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 1.7);
    grd.addColorStop(0, 'rgba(16,185,129,' + (0.09 + energy * 0.15) + ')');
    grd.addColorStop(1, 'rgba(16,185,129,0)');
    ctx.fillStyle = grd;
    ctx.beginPath(); ctx.arc(cx, cy, R * 1.7, 0, TAU); ctx.fill();

    if (Math.random() < 0.03 + energy * 0.2) {
      const p = pts[(Math.random() * N) | 0];
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.beginPath(); ctx.arc(p.x, p.y, 1.8, 0, TAU); ctx.fill();
    }
  }
  pwin.requestAnimationFrame(frame);
})();
</script>
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


@st.cache_resource(show_spinner=False)
def _sec_name_index() -> set[str]:
    """Set of lowercase names/symbols that identify a real SEC filer.

    Includes every ticker symbol, each company's core name and its first word,
    plus common nicknames. Used to tell whether a name in a question belongs to
    a company that actually files with the SEC.
    """
    try:
        data = load_tickers_data()
    except Exception:
        return set()
    names: set[str] = set(_COMPANY_NICKNAMES)
    for record in data.values():
        names.add(str(record["ticker"]).lower())
        core = _core_name(record["title"])
        if len(core) >= 3:
            names.add(core)
            first = core.split()[0]
            if len(first) >= 3:
                names.add(first)
    return names


def _candidate_company_names(question: str) -> list[str]:
    """Pull likely company mentions (runs of capitalized non-stopword words)."""
    phrases: list[str] = []
    current: list[str] = []
    for token in re.findall(r"[A-Za-z0-9&.\-]+", question):
        if token[:1].isupper() and token.lower() not in _NAME_STOPWORDS:
            current.append(token)
        elif current:
            phrases.append(" ".join(current))
            current = []
    if current:
        phrases.append(" ".join(current))

    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        if phrase.lower() not in seen:
            seen.add(phrase.lower())
            unique.append(phrase)
    return unique


def _is_known_filer(name: str, name_index: set[str]) -> bool:
    low = name.lower()
    return low in name_index or low.split()[0] in name_index


def find_unrecognized_companies(question: str) -> tuple[list[str], list[str]]:
    """Split company mentions into (recognized, not-in-SEC-list) names."""
    name_index = _sec_name_index()
    if not name_index:
        return [], []
    recognized, unknown = [], []
    for candidate in _candidate_company_names(question):
        (recognized if _is_known_filer(candidate, name_index) else unknown).append(candidate)
    return recognized, unknown


def ensure_companies_indexed(question: str, current_tickers: set[str]):
    """Fetch + index any company named in the question that isn't stored yet.

    Reuses ingest.ingest_ticker and index.index_ticker. Returns
    (added_tickers, notes) where notes are user-facing status/error messages.
    """
    try:
        tickers_data = load_tickers_data()
    except Exception as exc:  # noqa: BLE001
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
        try:
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
        except Exception as exc:  # noqa: BLE001 - keep the app responsive on model failures
            message = (
                "Unable to finish the answer right now because the model provider "
                "is rate-limited or unavailable. Please try again in a moment."
            )
            st.error(message)
            st.caption(str(exc))
            status.update(label="Stopped — model unavailable.", state="error", expanded=False)
            return message, "Temporary model failure. No answer was generated.", hops

    return final_answer, confidence_note, hops


def _strip_scaffolding(text: str) -> str:
    """Remove format placeholders the model sometimes echoes into the answer."""
    text = re.sub(r"\(\s*no\s+(?:relevant\s+)?sources?\s*\)", "", text, flags=re.I)
    text = text.strip()
    if text.upper().endswith("NONE"):
        text = text[:-4].rstrip()
    return text


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
    body = _strip_scaffolding(body)

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
        recognized, unknown = find_unrecognized_companies(question)
        for name in unknown:
            st.warning(
                f"“{name}” isn’t in the SEC EDGAR filings list — it looks like a "
                "private company or a non-U.S. filer, so there are no 10-K/10-Q "
                "filings to analyze."
            )

        if unknown and not recognized:
            msg = (
                "I can only answer about companies that file 10-Ks/10-Qs with the "
                "SEC. Try a public company such as Apple, Microsoft, or Walmart."
            )
            st.info(msg)
            not_listed = "; ".join(f"“{n}” is not in the SEC filings list" for n in unknown)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"{not_listed}.\n\n{msg}"}
            )
            return

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
    components.html(BACKGROUND_HTML, height=0)
    st.markdown(
        """
        <div class="hero">
            <div class="hero-head">Your AI analyst for SEC filings is here.</div>
            <div class="hero-desc">Ask about any public company's 10-Ks and 10-Qs —
            get cited, fact-checked answers, pulled live from EDGAR when needed.</div>
        </div>
        <div class="hero-gap"></div>
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

    render_chip_row(COMPANY_CHIPS)
    render_chip_row(COMPARISON_CHIPS, per_row=2)

    st.text_area(
        "Your question",
        key="composer",
        placeholder="Ask about any public company's filings…",
        height=80,
        label_visibility="collapsed",
    )
    st.button("Send", type="primary", on_click=_submit_composer)


if __name__ == "__main__":
    main()
