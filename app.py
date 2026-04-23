"""Stage 1: Collect raw text records for each company seed.

Reads a plain-text company list, queries configured data sources (USPTO, PubMed,
SEC, OpenCorporates), and writes a CSV of Stage1Record rows.

Usage:
    python app.py

Environment variables are read from .env (see config.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from config import Settings
from clients.uspto_client import USPTOClient
from clients.ncbi_client import NCBIClient
from clients.sec_client import SECClient
# from clients.opencorporates_client import OpenCorporatesClient
from src.preprocess import normalize_company_seed, parse_date_to_mon_yy
from schema import Stage1Record

FILE_PATH = "organizations.10-13-25.all.txt"
OUTPUT_PATH = "data/raw/stage1_records.csv"
CHECKPOINT_EVERY = 1   # save after every N companies; increase for speed

_LEADING_JUNK_RE = re.compile(r"^[\.\{\(\[\s]+")


def _lightly_clean(value: str) -> str:
    return _LEADING_JUNK_RE.sub("", value).strip()


def _is_invalid_line(value: str) -> bool:
    if not value:
        return True
    upper = value.upper()
    if upper in {"#NAME?", "N/A", "NA", "NULL", "NONE"}:
        return True
    if re.fullmatch(r"\d+", value):
        return True
    return False


def load_company_list(filepath: str) -> list[str]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Company list not found: {filepath}")

    seen: dict[str, None] = {}   # ordered-set via dict
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            raw = normalize_company_seed(line.strip())
            if not raw:
                continue
            raw = _lightly_clean(raw)
            if _is_invalid_line(raw):
                continue
            seen[raw] = None

    return list(seen)


def _save_checkpoint(records: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=[
        "source_id", "source_type", "raw_text",
        "source_url", "date", "company_seed",
    ]).to_csv(output_path, index=False)


def collect_for_company(
    seed: str,
    next_id: int,
    settings: Settings,
) -> tuple[list[Stage1Record], int]:
    """Query all enabled sources for *seed* and return a list of Stage1Records."""
    records: list[Stage1Record] = []

    clean_seed = normalize_company_seed(seed)
    if not clean_seed:
        return records, next_id

    lookup_seed = clean_seed

    # Optional: use OpenCorporates to resolve the canonical legal name first.
    # if settings.opencorporates_enabled:
    #     try:
    #         oc = OpenCorporatesClient(settings)
    #         match = oc.search_company(clean_seed)
    #         if match and match.get("matched_name"):
    #             lookup_seed = match["matched_name"]
    #     except Exception as exc:
    #         print(f"[WARN] OpenCorporates failed for {clean_seed}: {exc}")

    def _append(rows: list[dict]) -> None:
        nonlocal next_id
        for row in rows:
            records.append(Stage1Record(
                source_id=next_id,
                source_type=row["source_type"],
                raw_text=row["raw_text"],
                source_url=row["source_url"],
                date=parse_date_to_mon_yy(row["date"]) or "",
                company_seed=clean_seed,
            ))
            next_id += 1

    if settings.uspto_enabled:
        try:
            _append(USPTOClient(settings).search_assignments(
                lookup_seed, limit=settings.max_records_per_source
            ))
        except Exception as exc:
            print(f"[WARN] USPTO failed for {clean_seed}: {exc}")

    if settings.ncbi_enabled:
        try:
            ncbi = NCBIClient(settings)
            pmids = ncbi.search_pubmed_ids(
                lookup_seed, retmax=min(20, settings.max_records_per_source)
            )
            _append(ncbi.fetch_pubmed_records(pmids))
        except Exception as exc:
            print(f"[WARN] PubMed failed for {clean_seed}: {exc}")

    if settings.sec_enabled:
        try:
            _append(SECClient(settings).collect_company_records(
                lookup_seed,
                forms=["10-K", "8-K", "S-1"],
                limit=min(10, settings.max_records_per_source),
            ))
        except Exception as exc:
            print(f"[WARN] SEC failed for {clean_seed}: {type(exc).__name__}: {exc}")

    return records, next_id


def main() -> None:
    settings = Settings()
    seeds = load_company_list(FILE_PATH)
    print(f"Loaded {len(seeds)} valid company seeds from {FILE_PATH}")

    all_records: list[dict] = []
    next_id = 1

    for idx, seed in enumerate(seeds, start=1):
        print(f"[{idx}/{len(seeds)}] Processing: {seed}")
        batch, next_id = collect_for_company(seed, next_id, settings)
        all_records.extend(r.to_dict() for r in batch)

        if idx % CHECKPOINT_EVERY == 0:
            _save_checkpoint(all_records, OUTPUT_PATH)
            print(f"[CHECKPOINT] {len(all_records)} rows after {idx} companies")

    _save_checkpoint(all_records, OUTPUT_PATH)
    print(f"\nWrote {len(all_records)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
