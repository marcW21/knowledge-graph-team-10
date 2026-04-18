\
#!/usr/bin/env python3
"""spaCy + EntityRuler NER for company/ORG extraction.

This version does NOT use any Hugging Face biomedical model.
It is designed for higher company-name recall than a plain spaCy baseline by:
1. running spaCy ORG NER
2. adding an EntityRuler built from company_seed values
3. adding simple legal-suffix variants for seed companies

Expected upstream flow:
    preprocess.py -> ner_spacy_entityruler.py -> resolve_alias.py -> extract_candidate_relations.py

Input:
    cleaned_input.csv from preprocess.py

Required columns:
    source_id, source_type, raw_text_cleaned

Recommended optional columns:
    source_url, date, company_seed, raw_text,
    source_type_cleaned, source_url_cleaned, date_cleaned, company_seed_cleaned

Output:
    ner_results CSV with one row per mention.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set

import pandas as pd

try:
    import spacy
    from spacy.language import Language
except ImportError as exc:
    raise ImportError("spaCy is not installed. In Colab run: pip install spacy") from exc

REQUIRED_COLUMNS = {"source_id", "source_type", "raw_text_cleaned"}

LEGAL_SUFFIXES = [
    "Inc.", "Inc", "Corporation", "Corp.", "Corp", "Co.", "Co", "Company",
    "Ltd.", "Ltd", "LLC", "PLC", "GmbH", "BV", "AB", "SE", "SA", "AG",
    "Therapeutics", "Therapeutics, Inc.", "Pharmaceuticals", "Pharmaceuticals, Inc.",
]

BAD_SEEDS = {"", "n/a", "na", "null", "none", "tbd", "#name?", "--", "unknown", "nocompany"}

OUTPUT_COLUMNS = [
    "source_id",
    "source_type",
    "source_type_cleaned",
    "source_url",
    "source_url_cleaned",
    "date",
    "date_cleaned",
    "company_seed",
    "company_seed_cleaned",
    "raw_text",
    "raw_text_cleaned",
    "raw_mention",
    "entity_label",
    "start_char",
    "end_char",
    "mention_source",
    "mention_confidence",
]

WS_RE = re.compile(r"\s+")
DATE_LIKE_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}$")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return WS_RE.sub(" ", str(value).replace("\u00a0", " ")).strip()


def validate_columns(columns: Sequence[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {', '.join(sorted(missing))}. "
            "Did you run preprocess.py first?"
        )


def load_spacy_model(model_name: str) -> Language:
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"Could not load spaCy model '{model_name}'. "
            f"In Colab run: python -m spacy download {model_name}"
        ) from exc


def is_reasonable_seed(seed: str) -> bool:
    seed = normalize_text(seed)
    if not seed:
        return False
    if seed.lower() in BAD_SEEDS:
        return False
    if DATE_LIKE_RE.match(seed):
        return False
    if seed.isdigit():
        return False
    return True


def build_entity_ruler_patterns(df: pd.DataFrame) -> List[Dict]:
    patterns: List[Dict] = []
    seen: Set[str] = set()

    candidate_seeds: List[str] = []
    if "company_seed_cleaned" in df.columns:
        candidate_seeds.extend(df["company_seed_cleaned"].fillna("").astype(str).tolist())
    if "company_seed" in df.columns:
        candidate_seeds.extend(df["company_seed"].fillna("").astype(str).tolist())

    for seed in candidate_seeds:
        seed = normalize_text(seed)
        if not is_reasonable_seed(seed):
            continue

        variants = {seed}
        # Add simple legal-suffix variants to improve recall.
        for suffix in LEGAL_SUFFIXES:
            variants.add(f"{seed} {suffix}".strip())

        # Add uppercase variant for common filing styles.
        variants.add(seed.upper())

        for variant in variants:
            norm_variant = variant.lower()
            if norm_variant in seen:
                continue
            seen.add(norm_variant)
            patterns.append({"label": "ORG", "pattern": variant})

    return patterns


def add_entity_ruler(nlp: Language, df: pd.DataFrame) -> Language:
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = build_entity_ruler_patterns(df)
    if patterns:
        ruler.add_patterns(patterns)
    return nlp


def row_metadata(row: pd.Series) -> Dict[str, object]:
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


def extract_mentions(df: pd.DataFrame, nlp: Language, keep_labels: Set[str], batch_size: int) -> List[Dict]:
    records: List[Dict] = []
    texts = df["raw_text_cleaned"].fillna("").astype(str).tolist()

    for (_, row), doc in zip(df.iterrows(), nlp.pipe(texts, batch_size=batch_size)):
        base = row_metadata(row)
        for ent in doc.ents:
            if ent.label_.upper() not in keep_labels:
                continue
            records.append(
                {
                    **base,
                    "raw_mention": ent.text,
                    "entity_label": ent.label_,
                    "start_char": int(ent.start_char),
                    "end_char": int(ent.end_char),
                    "mention_source": "spacy_or_entityruler",
                    "mention_confidence": "",
                }
            )
    return records


def dedupe_mentions(records: List[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(records)
    df["_mention_norm"] = df["raw_mention"].fillna("").astype(str).str.lower().str.strip()
    df = df.sort_values(by=["source_id", "start_char", "end_char", "_mention_norm"])
    df = df.drop_duplicates(
        subset=["source_id", "start_char", "end_char", "_mention_norm"],
        keep="first",
    )
    df = df.drop(columns=["_mention_norm"])
    return df[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="spaCy + EntityRuler NER for company/ORG extraction.")
    parser.add_argument("--input", required=True, help="Path to cleaned_input.csv from preprocess.py")
    parser.add_argument("--output", required=True, help="Path to write NER results CSV")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name")
    parser.add_argument(
        "--entity-labels",
        nargs="+",
        default=["ORG"],
        help="spaCy entity labels to keep. Default: ORG",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="spaCy batch size. Default: 32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df.columns)

    # Fill optional columns for stable downstream schema.
    for column in ["source_url", "date", "company_seed", "raw_text"]:
        if column not in df.columns:
            df[column] = ""
    if "source_type_cleaned" not in df.columns:
        df["source_type_cleaned"] = df["source_type"].fillna("").astype(str)
    if "source_url_cleaned" not in df.columns:
        df["source_url_cleaned"] = df["source_url"].fillna("").astype(str)
    if "date_cleaned" not in df.columns:
        df["date_cleaned"] = df["date"].fillna("").astype(str)
    if "company_seed_cleaned" not in df.columns:
        df["company_seed_cleaned"] = df["company_seed"].fillna("").astype(str)

    nlp = load_spacy_model(args.spacy_model)
    nlp = add_entity_ruler(nlp, df)

    keep_labels = {label.upper() for label in args.entity_labels}
    records = extract_mentions(df, nlp, keep_labels, args.batch_size)

    output_df = dedupe_mentions(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"[ner_spacy_entityruler] Wrote {len(output_df)} mention(s) to: {output_path}")


if __name__ == "__main__":
    main()
