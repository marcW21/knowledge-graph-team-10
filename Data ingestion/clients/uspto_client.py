from __future__ import annotations
import requests
import xml.etree.ElementTree as ET
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Settings
from processing.preprocess import build_raw_text


class USPTOClient:
    """
    USPTO Patent Assignment Search API returns XML.
    This client uses assignment search as a practical source of
    patent-linked company relationship evidence for Stage 1.
    """

    SEARCH_URL = "https://assignment-api.uspto.gov/patent/search"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def search_assignments(self, company_seed: str, rows: int = 20) -> list[dict]:
        params = {
            "query": company_seed,
            "start": 0,
            "rows": rows,
        }
        resp = self.session.get(self.SEARCH_URL, params=params, timeout=self.settings.request_timeout)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        results: list[dict] = []

        for doc in root.findall(".//result/doc"):
            fields = {}
            for child in doc:
                tag = child.tag.lower()
                fields[tag] = (child.text or "").strip()

            reel = fields.get("reelno", "")
            frame = fields.get("frameno", "")
            execution_date = fields.get("executiondate", "")
            recorded_date = fields.get("recordeddate", "")
            assignee = fields.get("assignee", "")
            assignor = fields.get("assignor", "")
            patent = fields.get("patentnumber", "") or fields.get("applicationnumber", "")

            raw_text = build_raw_text(
                f"Assignor: {assignor}",
                f"Assignee: {assignee}",
                f"Patent/Application: {patent}",
                f"Execution date: {execution_date}",
                f"Recorded date: {recorded_date}",
            )

            source_url = (
                "https://assignmentcenter.uspto.gov/#!/patent/search/resultAbstract"
                f"?reelFrame={reel}:{frame}"
                if reel and frame else
                "https://assignmentcenter.uspto.gov/"
            )

            results.append({
                "source_type": "USPTO",
                "raw_text": raw_text,
                "source_url": source_url,
                "date": recorded_date or execution_date,
            })

        return results