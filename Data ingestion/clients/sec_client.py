from __future__ import annotations
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Settings
from processing.preprocess import build_raw_text


class SECClient:
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    EXTRACTOR_URL = "https://api.sec-api.io/extractor"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        })

    def enabled(self) -> bool:
        return self.settings.sec_enabled

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def resolve_cik(self, company_seed: str) -> str | None:
        resp = self.session.get(self.TICKERS_URL, timeout=self.settings.request_timeout)
        print(f"[DEBUG] resolve_cik ticker status for {company_seed}: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()

        seed_upper = company_seed.upper()
        for _, item in data.items():
            title = str(item.get("title", "")).upper()
            if seed_upper in title or title in seed_upper:
                cik = str(item["cik_str"]).zfill(10)
                print(f"[DEBUG] resolved CIK for {company_seed}: {cik}")
                return cik

        print(f"[DEBUG] no CIK match for {company_seed}")
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def get_recent_filings(self, cik: str, forms: list[str] | None = None, limit: int = 20) -> list[dict]:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = self.session.get(url, timeout=self.settings.request_timeout)
        print(f"[DEBUG] get_recent_filings status for CIK {cik}: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        form_types = recent.get("form", [])
        primary_docs = recent.get("primaryDocument", [])
        company_name = data.get("name", "")

        rows = []
        for acc, filed, form, doc in zip(accession_numbers, filing_dates, form_types, primary_docs):
            if forms and form not in forms:
                continue

            acc_nodash = acc.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"

            rows.append({
                "company_name": company_name,
                "form": form,
                "filing_date": filed,
                "filing_url": filing_url,
                "accession_number": acc,
            })

            if len(rows) >= limit:
                break

        print(f"[DEBUG] found {len(rows)} recent filings for CIK {cik}")
        return rows

    def get_item_code(self, form: str) -> str | None:
        if form == "10-K":
            return "1A"
        if form == "10-Q":
            return "part2item1a"
        if form == "8-K":
            return None
        if form == "S-1":
            return "1A"
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
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
        print(f"[DEBUG] extractor status for {filing_url}: {resp.status_code}")
        resp.raise_for_status()
        return resp.text.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def fetch_full_filing_text(self, filing_url: str) -> str:
        resp = self.session.get(filing_url, timeout=self.settings.request_timeout)
        print(f"[DEBUG] filing text status for {filing_url}: {resp.status_code}")
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        text = resp.text

        if "html" in content_type.lower() or "<html" in text.lower():
            soup = BeautifulSoup(text, "lxml")
            filing_text = soup.get_text(" ", strip=True)
        else:
            filing_text = text

        return filing_text[:20000]

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

        filings = self.get_recent_filings(cik=cik, forms=forms, limit=limit)
        records = []

        for filing in filings:
            form = filing["form"]
            filing_url = filing["filing_url"]
            item_code = self.get_item_code(form)

            try:
                if self.settings.sec_extractor_enabled and item_code:
                    extracted_text = self.extract_section_text(filing_url, item_code)
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
                print(f"[WARN] Failed SEC filing {filing_url}: {type(e).__name__}: {e}")

        return records