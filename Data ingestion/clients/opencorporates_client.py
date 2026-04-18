from __future__ import annotations
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Settings
from processing.preprocess import canonicalize_company_for_lookup


class OpenCorporatesClient:
    BASE_URL = "https://api.opencorporates.com/v0.4"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()

    def enabled(self) -> bool:
        return bool(
            self.settings.use_opencorporates
            and self.settings.opencorporates_api_key
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def search_company(self, company_name: str) -> dict | None:
        if not self.enabled():
            return None

        query = canonicalize_company_for_lookup(company_name)
        if not query:
            return None

        url = f"{self.BASE_URL}/companies/search"
        params = {
            "q": query,
            "api_token": self.settings.opencorporates_api_key,
            "per_page": 5,
        }
        resp = self.session.get(url, params=params, timeout=self.settings.request_timeout)
        resp.raise_for_status()
        data = resp.json()

        companies = data.get("results", {}).get("companies", [])
        if not companies:
            return None

        company = companies[0].get("company", {})
        return {
            "matched_name": company.get("name"),
            "jurisdiction_code": company.get("jurisdiction_code"),
            "company_number": company.get("company_number"),
            "registry_url": company.get("registry_url"),
            "opencorporates_url": company.get("opencorporates_url"),
            "canonical_query": query,
        }