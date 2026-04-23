from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Settings
from src.preprocess import build_raw_text


class SECClient:
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    EXTRACTOR_URL = "https://api.sec-api.io/extractor"

    _TICKER_CACHE = None

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({
            # MUST be a real identifier per SEC policy
            "User-Agent": settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        })

    def enabled(self) -> bool:
        return self.settings.sec_enabled

    # ─────────────────────────────────────────────────────────────
    # TICKER → CIK RESOLUTION (CACHED)
    # ─────────────────────────────────────────────────────────────

    def _load_tickers(self):
        if SECClient._TICKER_CACHE is None:
            resp = self.session.get(
                self.TICKERS_URL,
                timeout=self.settings.request_timeout
            )
            resp.raise_for_status()
            SECClient._TICKER_CACHE = resp.json()
        return SECClient._TICKER_CACHE

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def resolve_cik(self, company_seed: str) -> str | None:
        data = self._load_tickers()

        seed_upper = company_seed.upper().strip()

        for _, item in data.items():
            title = str(item.get("title", "")).upper()

            # simple, stable containment match
            if seed_upper in title:
                cik = str(item["cik_str"]).zfill(10)
                print(f"[DEBUG] resolved CIK for {company_seed}: {cik}")
                return cik

        print(f"[DEBUG] no CIK match for {company_seed}")
        return None

    # ─────────────────────────────────────────────────────────────
    # FILINGS INDEX
    # ─────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def get_recent_filings(
        self,
        cik: str,
        forms: list[str] | None = None,
        limit: int = 20
    ) -> list[dict]:

        cik = str(int(cik)).zfill(10)

        url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        resp = self.session.get(
            url,
            timeout=self.settings.request_timeout
        )
        print(f"[DEBUG] filings status for CIK {cik}: {resp.status_code}")
        resp.raise_for_status()

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})

        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        form_types = recent.get("form", [])
        primary_docs = recent.get("primaryDocument", [])

        company_name = data.get("name", "")

        rows = []

        for acc, filed, form, doc in zip(
            accession_numbers,
            filing_dates,
            form_types,
            primary_docs
        ):
            if forms and form not in forms:
                continue

            acc_nodash = acc.replace("-", "")

            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_nodash}/{doc}"
            )

            rows.append({
                "company_name": company_name,
                "form": form,
                "filing_date": filed,
                "filing_url": filing_url,
                "accession_number": acc,
            })

            if len(rows) >= limit:
                break

        print(f"[DEBUG] found {len(rows)} filings for CIK {cik}")
        return rows

    # ─────────────────────────────────────────────────────────────
    # FORM LOGIC
    # ─────────────────────────────────────────────────────────────

    def get_item_code(self, form: str) -> str | None:
        mapping = {
            "10-K": "1A",
            "10-K/A": "1A",
            "10-Q": "part2item1a",
            "10-Q/A": "part2item1a",
            "S-1": "1A",
        }
        return mapping.get(form)

    # ─────────────────────────────────────────────────────────────
    # EXTRACTOR
    # ─────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def extract_section_text(self, filing_url: str, item: str) -> str:
        params = {
            "url": filing_url,
            "item": item,
            "type": "text",
            "token": self.settings.sec_api_key,
        }

        resp = self.session.get(
            self.EXTRACTOR_URL,
            params=params,
            timeout=self.settings.request_timeout
        )

        print(f"[DEBUG] extractor status {resp.status_code} for {filing_url}")
        resp.raise_for_status()

        return resp.text.strip()

    # ─────────────────────────────────────────────────────────────
    # FULL FILING FETCH (FALLBACK)
    # ─────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def fetch_full_filing_text(self, filing_url: str) -> str:
        resp = self.session.get(filing_url, timeout=self.settings.request_timeout)

        print(f"[DEBUG] filing fetch status {resp.status_code}")

        resp.raise_for_status()

        text = resp.text
        content_type = resp.headers.get("Content-Type", "")

        if "html" in content_type.lower() or "<html" in text.lower():
            try:
                soup = BeautifulSoup(text, "html.parser")
                filing_text = soup.get_text(" ", strip=True)
            except Exception:
                filing_text = text
        else:
            filing_text = text

        return filing_text

    # ─────────────────────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────────────────────

    def collect_company_records(
        self,
        company_seed: str,
        forms: list[str] | None = None,
        limit: int = 10
    ) -> list[dict]:

        if not self.settings.sec_enabled:
            return []

        cik = self.resolve_cik(company_seed)
        if not cik:
            return []

        filings = self.get_recent_filings(
            cik=cik,
            forms=forms,
            limit=limit
        )

        records = []

        for filing in filings:
            form = filing["form"]
            filing_url = filing["filing_url"]
            item_code = self.get_item_code(form)

            try:
                if self.settings.sec_extractor_enabled and item_code:
                    extracted_text = self.extract_section_text(
                        filing_url,
                        item_code
                    )

                    raw_text = build_raw_text(
                        filing["company_name"],
                        form,
                        f"Extracted item {item_code}",
                        extracted_text
                    )
                else:
                    filing_text = self.fetch_full_filing_text(filing_url)

                    raw_text = build_raw_text(
                        filing["company_name"],
                        form,
                        filing_text
                    )

                records.append({
                    "source_type": "SEC",
                    "raw_text": raw_text,
                    "source_url": filing_url,
                    "date": filing["filing_date"],
                    "company_seed": company_seed,
                })

            except Exception as e:
                print(
                    f"[WARN] SEC filing failed {filing_url}: "
                    f"{type(e).__name__}: {e}"
                )

        return records