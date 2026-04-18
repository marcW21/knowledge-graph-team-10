
#!/usr/bin/env python3
"""Preprocess input data before running NER.

This script:
1. Reads a CSV file containing source text and metadata.
2. Normalizes whitespace and source types.
3. Flags suspicious company_seed values such as Excel-like errors or date-like tokens.
4. Splits the dataset into:
   - cleaned_input.csv: rows that are safe to continue into NER
   - invalid_rows.csv: rows that should be reviewed or excluded for now

Expected input columns (recommended):
- source_id
- source_type
- raw_text
- source_url
- date
- company_seed

The script is conservative by design:
if company_seed looks broken or missing, the row is routed to invalid_rows.csv.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


REQUIRED_COLUMNS = {"source_id", "source_type", "raw_text"}
OPTIONAL_COLUMNS = ["source_url", "date", "company_seed"]

EXCEL_ERROR_TOKENS = {
    "#NAME?",
    "#VALUE!",
    "#REF!",
    "#DIV/0!",
    "#NUM!",
    "#N/A",
    "#NULL!",
}

MONTHS = {
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "SEPT", "OCT", "NOV", "DEC",
}

VALID_SOURCE_TYPES = {"SEC", "USPTO", "PUBMED", "NCBI", "OPENCORPORATES"}


def normalize_whitespace(value: object) -> str:
    """Convert a value to string and collapse repeated whitespace."""
    if pd.isna(value):
        return ""
    text = str(value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_source_type(value: object) -> str:
    """Normalize source type to upper-case with simple aliases."""
    text = normalize_whitespace(value).upper()
    aliases = {
        "NCBI E-UTILITIES": "PUBMED",
        "NCBI": "PUBMED",
        "PATENTSVIEW": "USPTO",
        "EDGAR": "SEC",
    }
    return aliases.get(text, text)


def looks_like_excel_error(text: str) -> bool:
    """Detect obvious spreadsheet error tokens."""
    cleaned = text.strip().upper()
    return cleaned in EXCEL_ERROR_TOKENS


def looks_like_date_like_token(text: str) -> bool:
    """Catch strings like '11-JUL' or '7-May' that often come from Excel coercion."""
    cleaned = text.strip().upper()
    if not cleaned:
        return False

    # Exact pattern like 11-JUL or 7-MAY
    match = re.fullmatch(r"(\d{1,2})-([A-Z]{3,4})", cleaned)
    if match:
        _, month = match.groups()
        return month in MONTHS

    # Pattern like JUL-11 or MAY-7
    match = re.fullmatch(r"([A-Z]{3,4})-(\d{1,2})", cleaned)
    if match:
        month, _ = match.groups()
        return month in MONTHS

    return False


def clean_company_seed(value: object) -> Tuple[str, bool, str]:
    """Return cleaned company seed, invalid flag, and invalid reason."""
    cleaned = normalize_whitespace(value)

    if not cleaned:
        return cleaned, True, "missing_company_seed"
    if looks_like_excel_error(cleaned):
        return cleaned, True, "excel_error_token"
    if looks_like_date_like_token(cleaned):
        return cleaned, True, "date_like_token"

    return cleaned, False, ""


def validate_columns(columns: Iterable[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"Input CSV is missing required column(s): {missing_str}. "
            f"Required columns are: {', '.join(sorted(REQUIRED_COLUMNS))}."
        )


def build_row_invalid_reasons(row: pd.Series) -> List[str]:
    """Return all reasons a row should be excluded from downstream NER."""
    reasons: List[str] = []

    if not row["raw_text_cleaned"]:
        reasons.append("missing_raw_text")
    if not row["source_type_cleaned"]:
        reasons.append("missing_source_type")
    elif row["source_type_cleaned"] not in VALID_SOURCE_TYPES:
        reasons.append("unknown_source_type")
    if row.get("company_seed_invalid", False):
        reasons.append(str(row.get("company_seed_invalid_reason", "invalid_company_seed")))

    return reasons


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the DataFrame and split it into valid and invalid subsets."""
    df = df.copy()

    # Ensure optional columns exist so later code is predictable.
    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df["source_type_cleaned"] = df["source_type"].apply(normalize_source_type)
    df["raw_text_cleaned"] = df["raw_text"].apply(normalize_whitespace)
    df["source_url_cleaned"] = df["source_url"].apply(normalize_whitespace)
    df["date_cleaned"] = df["date"].apply(normalize_whitespace)

    company_seed_results = df["company_seed"].apply(clean_company_seed)
    df["company_seed_cleaned"] = company_seed_results.apply(lambda x: x[0])
    df["company_seed_invalid"] = company_seed_results.apply(lambda x: x[1])
    df["company_seed_invalid_reason"] = company_seed_results.apply(lambda x: x[2])

    df["row_invalid_reasons"] = df.apply(build_row_invalid_reasons, axis=1)
    df["row_valid_for_ner"] = df["row_invalid_reasons"].apply(lambda reasons: len(reasons) == 0)

    valid_df = df[df["row_valid_for_ner"]].copy()
    invalid_df = df[~df["row_valid_for_ner"]].copy()

    if not invalid_df.empty:
        invalid_df["row_invalid_reasons"] = invalid_df["row_invalid_reasons"].apply(lambda x: ";".join(x))
    if not valid_df.empty:
        valid_df["row_invalid_reasons"] = valid_df["row_invalid_reasons"].apply(lambda x: ";".join(x))

    return valid_df, invalid_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Maker Day CSV input.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw input CSV (e.g. mock_input.csv).",
    )
    parser.add_argument(
        "--cleaned-output",
        required=True,
        help="Path to write cleaned rows that are safe for NER.",
    )
    parser.add_argument(
        "--invalid-output",
        required=True,
        help="Path to write invalid / review-needed rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    cleaned_output_path = Path(args.cleaned_output)
    invalid_output_path = Path(args.invalid_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df.columns)

    valid_df, invalid_df = preprocess_dataframe(df)

    cleaned_output_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_df.to_csv(cleaned_output_path, index=False)
    invalid_df.to_csv(invalid_output_path, index=False)

    print(f"[preprocess] Wrote {len(valid_df)} valid row(s) to: {cleaned_output_path}")
    print(f"[preprocess] Wrote {len(invalid_df)} invalid row(s) to: {invalid_output_path}")


if __name__ == "__main__":
    main()
