from __future__ import annotations
import requests
import xml.etree.ElementTree as ET
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Settings
from src.preprocess import build_raw_text


class NCBIClient:
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()

    def enabled(self) -> bool:
        return self.settings.ncbi_enabled

    def _base_params(self) -> dict:
        params = {
            "tool": self.settings.ncbi_tool,
            "email": self.settings.ncbi_email,
        }
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def search_pubmed_ids(self, company_seed: str, retmax: int = 20) -> list[str]:
        if not self.enabled():
            return []

        params = {
            **self._base_params(),
            "db": "pubmed",
            "term": f'"{company_seed}"[Title/Abstract] OR "{company_seed}"[Affiliation]',
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
        }
        resp = self.session.get(
            self.ESEARCH_URL,
            params=params,
            timeout=self.settings.request_timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def fetch_pubmed_records(self, pmids: list[str]) -> list[dict]:
        if not self.enabled():
            return []

        if not pmids:
            return []

        params = {
            **self._base_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        resp = self.session.get(
            self.EFETCH_URL,
            params=params,
            timeout=self.settings.request_timeout
        )
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        records: list[dict] = []

        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            article_title = article.findtext(".//ArticleTitle")
            abstract_nodes = article.findall(".//Abstract/AbstractText")
            abstract = " ".join(
                ["".join(node.itertext()).strip() for node in abstract_nodes if node is not None]
            )

            pub_year = (
                article.findtext(".//PubDate/Year")
                or article.findtext(".//ArticleDate/Year")
            )
            pub_month = (
                article.findtext(".//PubDate/Month")
                or article.findtext(".//ArticleDate/Month")
            )

            date_text = None
            if pub_year and pub_month:
                date_text = f"{pub_month}-{pub_year}"
            elif pub_year:
                date_text = pub_year

            source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            raw_text = build_raw_text(article_title, abstract)

            records.append({
                "source_type": "PUBMED",
                "raw_text": raw_text,
                "source_url": source_url,
                "date": date_text,
            })

        return records