"""Download SEC 10-K/10-Q filings for a list of tickers into data/cache/."""

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from sec_edgar_api import EdgarClient

from src.config import SEC_USER_AGENT

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

CACHE_DIR = Path("data/cache")
TICKERS_CACHE_PATH = CACHE_DIR / "company_tickers.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
MAX_REQUESTS_PER_SECOND = 8
FORMS = ("10-K", "10-Q")


class RateLimiter:
    def __init__(self, max_per_second: float):
        self._min_interval = 1.0 / max_per_second
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


def require_user_agent() -> str:
    if not SEC_USER_AGENT:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. Add it to your .env file, "
            "e.g. 'Your Name your@email.com'."
        )
    return SEC_USER_AGENT


def fetch_company_tickers(session: requests.Session, rate_limiter: RateLimiter) -> dict:
    if TICKERS_CACHE_PATH.exists():
        return json.loads(TICKERS_CACHE_PATH.read_text(encoding="utf-8"))

    rate_limiter.wait()
    resp = session.get(TICKERS_URL)
    resp.raise_for_status()
    data = resp.json()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TICKERS_CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")
    return data


def build_ticker_to_cik(tickers_data: dict) -> dict[str, str]:
    return {
        record["ticker"].upper(): f"{record['cik_str']:010d}"
        for record in tickers_data.values()
    }


def select_recent_filings(submissions: dict, forms: tuple[str, ...], count: int) -> list[dict]:
    recent = submissions["filings"]["recent"]
    by_form: dict[str, list[dict]] = {form: [] for form in forms}

    for i, form in enumerate(recent["form"]):
        if form not in by_form:
            continue
        by_form[form].append(
            {
                "form": form,
                "accession": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "primary_document": recent["primaryDocument"][i],
            }
        )

    selected = []
    for form in forms:
        filings = sorted(by_form[form], key=lambda f: f["filing_date"], reverse=True)
        selected.extend(filings[:count])
    return selected


def build_source_url(cik: str, accession: str, primary_document: str) -> str:
    cik_no_zeros = str(int(cik))
    accession_no_dashes = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_zeros}/{accession_no_dashes}/{primary_document}"
    )


def fetch_filing_text(session: requests.Session, rate_limiter: RateLimiter, url: str) -> str:
    rate_limiter.wait()
    resp = session.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none")):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def save_filing(ticker: str, filing: dict, source_url: str, text: str) -> None:
    out_dir = CACHE_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{filing['accession']}_{filing['form']}"
    (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "ticker": ticker,
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "accession": filing["accession"],
                "source_url": source_url,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def ingest_ticker(
    ticker: str,
    cik: str,
    client: EdgarClient,
    session: requests.Session,
    rate_limiter: RateLimiter,
    count: int,
) -> None:
    rate_limiter.wait()
    submissions = client.get_submissions(cik)
    filings = select_recent_filings(submissions, FORMS, count)

    for filing in filings:
        out_path = CACHE_DIR / ticker / f"{filing['accession']}_{filing['form']}.txt"
        if out_path.exists():
            print(f"[{ticker}] {filing['form']} {filing['accession']} already cached, skipping")
            continue

        source_url = build_source_url(cik, filing["accession"], filing["primary_document"])
        print(f"[{ticker}] fetching {filing['form']} {filing['accession']}")
        text = fetch_filing_text(session, rate_limiter, source_url)
        save_filing(ticker, filing, source_url, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols, e.g. AAPL MSFT")
    parser.add_argument("--count", type=int, default=2, help="Number of each form (10-K/10-Q) to fetch per ticker")
    args = parser.parse_args()

    user_agent = require_user_agent()
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_SECOND)
    client = EdgarClient(user_agent=user_agent)

    tickers_data = fetch_company_tickers(session, rate_limiter)
    ticker_to_cik = build_ticker_to_cik(tickers_data)

    for raw_ticker in args.tickers:
        ticker = raw_ticker.upper()
        cik = ticker_to_cik.get(ticker)
        if cik is None:
            print(f"[{ticker}] unknown ticker, skipping")
            continue
        ingest_ticker(ticker, cik, client, session, rate_limiter, args.count)


if __name__ == "__main__":
    main()
