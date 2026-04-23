#!/usr/bin/env python3
"""spaCy + EntityRuler NER for company/ORG extraction.

Pipeline position:  preprocess.py → ner.py → resolve_alias.py → extract_candidate_relations.py

Strategy
--------
1. Load spaCy's en_core_web_sm (or a model passed via --spacy-model).
2. Inject an EntityRuler (before the NER component) built from the unique
   company_seed values in the input file, plus common legal-suffix variants.
   Matching is case-insensitive via phrase_matcher_attr="LOWER".
3. Run the full pipeline over raw_text_cleaned.
4. Keep only entities whose label is in --entity-labels (default: ORG).
5. Deduplicate by (source_id, start_char, end_char, mention_lower).

Input  : cleaned_input CSV from preprocess.py
Output : ner_results CSV with one row per mention.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

try:
    import spacy
    from spacy.language import Language
except ImportError as exc:
    raise ImportError(
        "spaCy is not installed. Run: pip install spacy && python -m spacy download en_core_web_sm"
    ) from exc

from constants import ALL_LEGAL_SUFFIXES

REQUIRED_COLUMNS: frozenset[str] = frozenset({"source_id", "source_type", "raw_text_cleaned"})

# Additional suffix tokens to append when building EntityRuler patterns.
# These supplement ALL_LEGAL_SUFFIXES with industry-specific variants.
EXTRA_SUFFIX_VARIANTS: list[str] = [
    "Inc.", "Corp.", "Co.", "Ltd.",
    "Therapeutics", "Therapeutics, Inc.",
    "Pharmaceuticals", "Pharmaceuticals, Inc.",
    "Biosciences", "Biosciences, Inc.",
    "Sciences", "Sciences, Inc.",
]

BAD_SEEDS: frozenset[str] = frozenset({
    "", "n/a", "na", "null", "none", "tbd", "#name?", "--", "unknown",
})

_WS_RE = re.compile(r"\s+")
_DATE_LIKE_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3,4}$")

OUTPUT_COLUMNS: list[str] = [
    "source_id", "source_type",
    "source_type_cleaned", "source_url", "source_url_cleaned",
    "date", "date_cleaned", "company_seed", "company_seed_cleaned",
    "raw_text", "raw_text_cleaned",
    "raw_mention", "entity_label", "start_char", "end_char",
    "mention_source", "mention_confidence",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _norm(value: object) -> str:
    if pd.isna(value):
        return ""
    return _WS_RE.sub(" ", str(value).replace("\u00a0", " ")).strip()


def _is_reasonable_seed(seed: str) -> bool:
    s = _norm(seed)
    if not s or s.lower() in BAD_SEEDS:
        return False
    if _DATE_LIKE_RE.match(s) or s.isdigit():
        return False
    return True


# ─── EntityRuler construction ─────────────────────────────────────────────────

def _build_patterns(df: pd.DataFrame) -> list[Dict]:
    """Build case-insensitive EntityRuler patterns from unique seeds."""
    seen: Set[str] = set()
    patterns: list[Dict] = []

    seeds: List[str] = []
    for col in ("company_seed_cleaned", "company_seed"):
        if col in df.columns:
            seeds.extend(df[col].fillna("").astype(str).unique().tolist())

    for seed in seeds:
        seed = _norm(seed)
        if not _is_reasonable_seed(seed):
            continue

        variants = {seed, seed.upper()}
        for suffix in EXTRA_SUFFIX_VARIANTS:
            variants.add(f"{seed} {suffix}")

        for v in variants:
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            patterns.append({"label": "ORG", "pattern": v})

    return patterns


def _add_entity_ruler(nlp: Language, df: pd.DataFrame) -> Language:
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner",
        config={"phrase_matcher_attr": "LOWER"},
    )
    patterns = _build_patterns(df)
    if patterns:
        ruler.add_patterns(patterns)
    return nlp


# ─── Extraction ───────────────────────────────────────────────────────────────

def _row_meta(row: pd.Series) -> dict:
    return {
        "source_id": row["source_id"],
        "source_type": row.get("source_type", ""),
        "source_type_cleaned": row.get("source_type_cleaned", row.get("source_type", "")),
        "source_url": row.get("source_url", ""),
        "source_url_cleaned": row.get("source_url_cleaned", row.get("source_url", "")),
        "date": row.get("date", ""),
        "date_cleaned": row.get("date_cleaned", row.get("date", "")),
        "company_seed": row.get("company_seed", ""),
        "company_seed_cleaned": row.get("company_seed_cleaned", row.get("company_seed", "")),
        "raw_text": row.get("raw_text", row.get("raw_text_cleaned", "")),
        "raw_text_cleaned": row.get("raw_text_cleaned", ""),
    }


def extract_mentions(
    df: pd.DataFrame,
    nlp: Language,
    keep_labels: Set[str],
    batch_size: int,
) -> list[dict]:
    records: list[dict] = []
    texts = df["raw_text_cleaned"].fillna("").astype(str).tolist()

    for (_, row), doc in zip(df.iterrows(), nlp.pipe(texts, batch_size=batch_size)):
        meta = _row_meta(row)
        for ent in doc.ents:
            if ent.label_.upper() not in keep_labels:
                continue
            records.append({
                **meta,
                "raw_mention": ent.text,
                "entity_label": ent.label_,
                "start_char": int(ent.start_char),
                "end_char": int(ent.end_char),
                "mention_source": "spacy_or_entityruler",
                "mention_confidence": "",
            })
    return records


def dedupe_mentions(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame(records)
    out["_key"] = out["raw_mention"].str.lower().str.strip()
    out = (
        out.sort_values(["source_id", "start_char", "end_char", "_key"])
           .drop_duplicates(subset=["source_id", "start_char", "end_char", "_key"], keep="first")
           .drop(columns=["_key"])
    )
    # Ensure all output columns exist even if no data arrived.
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[OUTPUT_COLUMNS]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="spaCy + EntityRuler NER.")
    p.add_argument("--input", required=True, help="cleaned_input CSV from preprocess.py")
    p.add_argument("--output", required=True, help="NER results CSV")
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--entity-labels", nargs="+", default=["ORG"])
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Fill optional columns so downstream schema is stable.
    for col in ["source_url", "date", "company_seed", "raw_text"]:
        # df.setdefault(col, "")
        df[col] = df[col].fillna("")
    for derived, base in [
        ("source_type_cleaned", "source_type"),
        ("source_url_cleaned", "source_url"),
        ("date_cleaned", "date"),
        ("company_seed_cleaned", "company_seed"),
    ]:
        if derived not in df.columns:
            df[derived] = df[base].fillna("").astype(str)

    try:
        nlp = spacy.load(args.spacy_model)
    except OSError as exc:
        raise OSError(
            f"Could not load '{args.spacy_model}'. "
            f"Run: python -m spacy download {args.spacy_model}"
        ) from exc

    nlp = _add_entity_ruler(nlp, df)
    keep_labels = {lbl.upper() for lbl in args.entity_labels}
    records = extract_mentions(df, nlp, keep_labels, args.batch_size)

    output_df = dedupe_mentions(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"[ner] Wrote {len(output_df)} mention(s) to: {output_path}")


if __name__ == "__main__":
    main()
