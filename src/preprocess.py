#!/usr/bin/env python3
"""Preprocessing utilities shared across Stage 1 (data collection) and Stage 2 (NER).

Stage 1 entry point  : normalize_company_seed(), parse_date_to_mon_yy(), build_raw_text()
Stage 2 entry point  : run via  python preprocess.py --input ... --cleaned-output ... --invalid-output ...

Design:
- Single source of truth for text-cleaning rules (see constants.py).
- Conservative: broken / ambiguous rows go to invalid_rows.csv, not into NER.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple

import ftfy
import pandas as pd
from dateutil import parser as dtparser

from constants import (
    ALL_LEGAL_SUFFIXES,
    CORRUPT_TOKENS,
    MONTH_ABBREVS,
    SOURCE_TYPE_ALIASES,
    VALID_SOURCE_TYPES,
)

# ─── Low-level text helpers ───────────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")
_DATE_LIKE_RE = re.compile(r"^(\d{1,2}-[A-Z]{3,4}|[A-Z]{3,4}-\d{1,2})$", re.IGNORECASE)
_NCT_RE = re.compile(r"^NCT\d+$", re.IGNORECASE)


def normalize_whitespace(value: object) -> str:
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return ""
    return _WS_RE.sub(" ", str(value).replace("\u00a0", " ")).strip()


def is_corrupt(value: str) -> bool:
    """Return True for values that should never be treated as real data."""
    upper = value.upper()
    if upper in CORRUPT_TOKENS:
        return True
    if value.startswith("#"):
        return True
    if upper in ALL_LEGAL_SUFFIXES:   # bare suffix with no base name
        return True
    if len(value) <= 1:
        return True
    if _NCT_RE.fullmatch(value):
        return True
    if _DATE_LIKE_RE.fullmatch(value):
        return True
    return False


# ─── Stage 1 helpers (used by app.py) ────────────────────────────────────────

def normalize_company_seed(value: str | None) -> str | None:
    """Clean a raw company seed from the input list.

    Returns None when the value is corrupt / empty.
    """
    cleaned = normalize_whitespace(value)
    if not cleaned or is_corrupt(cleaned):
        return None
    return normalize_whitespace(ftfy.fix_text(cleaned))


def parse_date_to_mon_yy(value: str | None) -> str | None:
    """Normalise an arbitrary date string to 'Mon-YY' (e.g. 'Jan-23').

    Returns None when parsing fails.
    """
    cleaned = normalize_whitespace(value)
    if not cleaned or is_corrupt(cleaned):
        return None

    for fmt in ("%b-%y", "%b-%Y", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%b-%y")
        except ValueError:
            pass

    try:
        return dtparser.parse(cleaned, fuzzy=True).strftime("%b-%y")
    except Exception:
        return None


def build_raw_text(*parts: str | None, max_len: int = 12_000) -> str:
    text = " ".join(normalize_whitespace(p) for p in parts if p)
    return normalize_whitespace(text)[:max_len]


# ─── Stage 2 helpers (used by the NER pipeline) ───────────────────────────────

def _is_date_like(text: str) -> bool:
    upper = text.strip().upper()
    m = re.fullmatch(r"(\d{1,2})-([A-Z]{3,4})", upper)
    if m and m.group(2) in MONTH_ABBREVS:
        return True
    m = re.fullmatch(r"([A-Z]{3,4})-(\d{1,2})", upper)
    if m and m.group(1) in MONTH_ABBREVS:
        return True
    return False


def normalize_source_type(value: object) -> str:
    text = normalize_whitespace(value).upper()
    return SOURCE_TYPE_ALIASES.get(text, text)


def clean_company_seed_for_ner(value: object) -> Tuple[str, bool, str]:
    """Return (cleaned_value, is_invalid, reason).

    Used during Stage 2 preprocessing to flag bad seeds before NER.
    """
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return cleaned, True, "missing_company_seed"
    upper = cleaned.upper()
    if upper in CORRUPT_TOKENS or cleaned.startswith("#"):
        return cleaned, True, "excel_error_token"
    if _is_date_like(cleaned):
        return cleaned, True, "date_like_token"
    return cleaned, False, ""


def _build_row_invalid_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if not row["raw_text_cleaned"]:
        reasons.append("missing_raw_text")
    st = row["source_type_cleaned"]
    if not st:
        reasons.append("missing_source_type")
    elif st not in VALID_SOURCE_TYPES:
        reasons.append("unknown_source_type")
    if row.get("company_seed_invalid", False):
        reasons.append(str(row.get("company_seed_invalid_reason", "invalid_company_seed")))
    return reasons


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a raw Stage-1 CSV into (valid_for_ner, invalid) DataFrames.

    Adds *_cleaned columns for every key field.
    """
    df = df.copy()

    for col in ["source_url", "date", "company_seed"]:
        if col not in df.columns:
            df[col] = ""

    df["source_type_cleaned"] = df["source_type"].apply(normalize_source_type)
    df["raw_text_cleaned"] = df["raw_text"].apply(normalize_whitespace)
    df["source_url_cleaned"] = df["source_url"].apply(normalize_whitespace)
    df["date_cleaned"] = df["date"].apply(normalize_whitespace)

    seed_results = df["company_seed"].apply(clean_company_seed_for_ner)
    df["company_seed_cleaned"] = seed_results.apply(lambda x: x[0])
    df["company_seed_invalid"] = seed_results.apply(lambda x: x[1])
    df["company_seed_invalid_reason"] = seed_results.apply(lambda x: x[2])

    df["_reasons"] = df.apply(_build_row_invalid_reasons, axis=1)
    df["row_valid_for_ner"] = df["_reasons"].apply(lambda r: len(r) == 0)

    valid_df = df[df["row_valid_for_ner"]].drop(columns=["_reasons"]).copy()
    invalid_df = df[~df["row_valid_for_ner"]].copy()
    invalid_df["row_invalid_reasons"] = invalid_df["_reasons"].apply(";".join)
    invalid_df = invalid_df.drop(columns=["_reasons"])

    return valid_df, invalid_df


# ─── CLI (Stage 2 mode) ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess Stage-1 CSV for NER.")
    p.add_argument("--input", required=True)
    p.add_argument("--cleaned-output", required=True)
    p.add_argument("--invalid-output", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    required = {"source_id", "source_type", "raw_text"}
    df = pd.read_csv(input_path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    valid_df, invalid_df = preprocess_dataframe(df)

    for path, frame, label in [
        (args.cleaned_output, valid_df, "valid"),
        (args.invalid_output, invalid_df, "invalid"),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"[preprocess] Wrote {len(frame)} {label} row(s) to: {path}")


if __name__ == "__main__":
    main()
