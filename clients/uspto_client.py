"""
USPTO patent data client using the USPTO Assignment Search API.

Replacement for deprecated PatentsView (search.patentsview.org).

API:
    https://developer.uspto.gov/assignment-search-api/search

Notes:
- No API key required
- Returns XML
- Focuses on patent assignments (assignee-centric view)
- Does NOT include CPC/IPC/WIPO/inventor metadata (API limitation)

Output schema (Stage1Record-compatible):
    source_type : "USPTO"
    raw_text    : structured patent summary string
    source_url  : https://patents.google.com/patent/US{patent_id}
    date        : grant year (if available from execution date)
"""

from __future__ import annotations

import time
from typing import Any, Iterable
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import requests

from config import Settings


# ──────────────────────────────────────────────────────────────────────────────
# API CONFIG
# ──────────────────────────────────────────────────────────────────────────────

_BASE_URL = "https://developer.uspto.gov/assignment-search-api"


# ──────────────────────────────────────────────────────────────────────────────
# CLIENT
# ──────────────────────────────────────────────────────────────────────────────

class USPTOClient:
    """Fetch patent assignment records by assignee organization."""

    def __init__(self, settings: Settings) -> None:
        self._timeout = settings.request_timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/xml",
            "User-Agent": "USPTOClient/1.0",
        })

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def search_assignments(
        self,
        company_name: str,
        limit: int = 100,
    ) -> list[dict[str, str]]:
        """
        Return Stage1Record-compatible dicts for assignee name search.
        """
        if not company_name or not company_name.strip():
            return []

        patents = self._fetch_assignments(company_name.strip(), limit)
        return [self._to_row(p) for p in patents]

    # ──────────────────────────────────────────────────────────────────────────
    # FETCH LAYER
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_assignments(self, org_name: str, limit: int) -> list[dict]:
        """
        Query USPTO Assignment Search API.
        """

        params = {
            "q": f'assigneeName:("{org_name}")',
            "rows": min(limit, 1000),
        }

        url = f"{_BASE_URL}/search?{urlencode(params)}"

        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        results: list[dict] = []

        for doc in root.findall(".//doc"):
            patent_id = _xml_field(doc, "patentNumber")
            execution_date = _xml_field(doc, "executionDate")
            assignee = _xml_field(doc, "assigneeName")

            results.append({
                "patent_id": patent_id,
                "patent_date": execution_date,
                "assignees": [{
                    "assignee_organization": assignee
                }],
                "application": {},
                "inventors": [],
                "cpc_at_issue": [],
                "ipcr": [],
                "wipo": [],
            })

        return results[:limit]

    # ──────────────────────────────────────────────────────────────────────────
    # TRANSFORM
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_row(patent: dict) -> dict[str, str]:
        patent_id = str(patent.get("patent_id") or "").strip()
        patent_date = str(patent.get("patent_date") or "").strip()

        grant_year = patent_date[:4] if patent_date else ""

        source_url = (
            f"https://patents.google.com/patent/US{patent_id}"
            if patent_id else ""
        )

        raw_text = _build_raw_text(patent, patent_id, grant_year)

        return {
            "source_type": "USPTO",
            "raw_text": raw_text,
            "source_url": source_url,
            "date": grant_year,
        }


# ──────────────────────────────────────────────────────────────────────────────
# RAW TEXT BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def _build_raw_text(patent: dict, patent_id: str, grant_year: str) -> str:
    parts: list[str] = []

    # Patent ID
    parts.append(f"Patent {patent_id}.")

    # Assignees
    assignees: list[dict] = patent.get("assignees") or []
    orgs = [
        a.get("assignee_organization")
        for a in assignees
        if a.get("assignee_organization")
    ]
    if orgs:
        parts.append(f"Assignee: {'; '.join(orgs)}.")

    # Grant year (often missing or derived from execution date)
    if grant_year:
        parts.append(f"Grant year: {grant_year}.")

    # No application data in this API
    parts.append("Application number: nan.")
    parts.append("Application year: nan.")

    # No location data
    parts.append("Location: nan.")

    # No classification data available from this API
    parts.append("CPC sections: nan.")
    parts.append("IPC sections: nan.")

    # No WIPO data
    parts.append("WIPO field: nan. WIPO sector: nan.")

    # Inventors not available
    parts.append("Team size: 0.")

    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# XML HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _xml_field(doc: ET.Element, name: str) -> str:
    """
    Extract <str name="field">VALUE</str> safely.
    """
    el = doc.find(f"./str[@name='{name}']")
    return el.text.strip() if el is not None and el.text else ""