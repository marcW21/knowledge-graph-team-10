from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

from config import Settings
from clients.uspto_client import USPTOClient
from clients.ncbi_client import NCBIClient
from clients.sec_client import SECClient
from clients.opencorporates_client import OpenCorporatesClient
from processing.preprocess import (
    normalize_company_seed,
    parse_date_to_mon_yy,
)
from processing.schema import Stage1Record


FILE_PATH = "organizations.10-13-25.all.txt"
OUTPUT_PATH = "outputs/stage1_records.csv"
CHECKPOINT_EVERY = 1


def lightly_clean_company_prefix(value: str) -> str:
    return re.sub(r"^[\.\{\(\[\s]+", "", value).strip()


def is_probably_invalid_company_line(value: str) -> bool:
    if not value:
        return True

    upper_v = value.upper()

    if upper_v in {"#NAME?", "N/A", "NA", "NULL", "NONE"}:
        return True

    if re.fullmatch(r"\d+", value):
        return True

    return False


def load_company_list_txt(filepath: str) -> list[str]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Company list not found: {filepath}")

    companies: list[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            cleaned = normalize_company_seed(raw)

            if not cleaned:
                continue

            cleaned = lightly_clean_company_prefix(cleaned)

            if is_probably_invalid_company_line(cleaned):
                continue

            companies.append(cleaned)

    return list(dict.fromkeys(companies))


def save_checkpoint(records: list[dict], output_path: str = OUTPUT_PATH) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=[
        "source_id",
        "source_type",
        "raw_text",
        "source_url",
        "date",
        "company_seed",
    ])
    df.to_csv(output_path, index=False)


def collect_stage1_for_company(
    company_seed: str,
    source_id_start: int,
    settings: Settings
) -> tuple[list[Stage1Record], int]:
    records: list[Stage1Record] = []
    next_id = source_id_start

    uspto = USPTOClient(settings)
    ncbi = NCBIClient(settings)
    sec = SECClient(settings)
    oc = OpenCorporatesClient(settings)

    clean_seed = normalize_company_seed(company_seed)
    if not clean_seed:
        return records, next_id

    lookup_seed = clean_seed

    # OpenCorporates
    if settings.opencorporates_enabled:
        try:
            oc_match = oc.search_company(clean_seed)
            if oc_match and oc_match.get("matched_name"):
                lookup_seed = oc_match["matched_name"]
        except Exception as e:
            print(f"[WARN] OpenCorporates failed for {clean_seed}: {e}")

    # USPTO
    if settings.uspto_enabled:
        try:
            for row in uspto.search_assignments(
                lookup_seed,
                rows=settings.max_records_per_source
            ):
                record = Stage1Record(
                    source_id=next_id,
                    source_type=row["source_type"],
                    raw_text=row["raw_text"],
                    source_url=row["source_url"],
                    date=parse_date_to_mon_yy(row["date"]) or "",
                    company_seed=clean_seed,
                )
                records.append(record)
                next_id += 1
        except Exception as e:
            print(f"[WARN] USPTO failed for {clean_seed}: {e}")

    # PUBMED
    if settings.ncbi_enabled:
        try:
            pmids = ncbi.search_pubmed_ids(
                lookup_seed,
                retmax=min(20, settings.max_records_per_source)
            )
            for row in ncbi.fetch_pubmed_records(pmids):
                record = Stage1Record(
                    source_id=next_id,
                    source_type=row["source_type"],
                    raw_text=row["raw_text"],
                    source_url=row["source_url"],
                    date=parse_date_to_mon_yy(row["date"]) or "",
                    company_seed=clean_seed,
                )
                records.append(record)
                next_id += 1
        except Exception as e:
            print(f"[WARN] PUBMED failed for {clean_seed}: {e}")

    # SEC
    if settings.sec_enabled:
        try:
            for row in sec.collect_company_records(
                lookup_seed,
                forms=["10-K", "8-K", "S-1"],
                limit=min(10, settings.max_records_per_source),
            ):
                record = Stage1Record(
                    source_id=next_id,
                    source_type=row["source_type"],
                    raw_text=row["raw_text"],
                    source_url=row["source_url"],
                    date=parse_date_to_mon_yy(row["date"]) or "",
                    company_seed=clean_seed,
                )
                records.append(record)
                next_id += 1
        except Exception as e:
            print(f"[WARN] SEC failed for {clean_seed}: {type(e).__name__}: {e}")
            if hasattr(e, "last_attempt"):
                try:
                    last_exc = e.last_attempt.exception()
                    print(f"[WARN] Last SEC exception for {clean_seed}: {type(last_exc).__name__}: {last_exc}")
                except Exception:
                    pass
    return records, next_id


def main():
    settings = Settings()

    company_seeds = load_company_list_txt(FILE_PATH)
    print(f"Loaded {len(company_seeds)} valid company seeds from {FILE_PATH}")

    all_records: list[dict] = []
    next_source_id = 1

    for idx, seed in enumerate(company_seeds, start=1):
        print(f"[{idx}/{len(company_seeds)}] Processing: {seed}")

        records, next_source_id = collect_stage1_for_company(
            company_seed=seed,
            source_id_start=next_source_id,
            settings=settings,
        )
        all_records.extend([r.to_dict() for r in records])

        if idx % CHECKPOINT_EVERY == 0:
            save_checkpoint(all_records, OUTPUT_PATH)
            print(f"[CHECKPOINT] Saved {len(all_records)} rows after {idx} companies")

    save_checkpoint(all_records, OUTPUT_PATH)

    out_df = pd.DataFrame(all_records, columns=[
        "source_id",
        "source_type",
        "raw_text",
        "source_url",
        "date",
        "company_seed",
    ])

    print(out_df.head(20))
    print(f"\nWrote {len(out_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()