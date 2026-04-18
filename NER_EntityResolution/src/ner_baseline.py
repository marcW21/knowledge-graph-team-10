
#!/usr/bin/env python3
"""Run a spaCy baseline NER pass over cleaned Maker Day input data.

This script:
1. Reads cleaned_input.csv from preprocess.py.
2. Loads a spaCy model (default: en_core_web_sm).
3. Extracts entities with selected labels (default: ORG).
4. Writes one row per detected mention to ner_results.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import pandas as pd

try:
    import spacy
except ImportError as exc:
    raise ImportError(
        "spaCy is not installed. In Colab, run: pip install spacy"
    ) from exc


REQUIRED_COLUMNS = {"source_id", "source_type", "raw_text_cleaned"}


def validate_columns(columns: Sequence[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"Input CSV is missing required column(s): {missing_str}. "
            "Did you run preprocess.py first?"
        )


def load_spacy_model(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"Could not load spaCy model '{model_name}'. "
            f"In Colab, run: python -m spacy download {model_name}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spaCy baseline NER.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to cleaned_input.csv from preprocess.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write ner_results.csv.",
    )
    parser.add_argument(
        "--model",
        default="en_core_web_sm",
        help="spaCy model name to load. Default: en_core_web_sm",
    )
    parser.add_argument(
        "--entity-labels",
        nargs="+",
        default=["ORG"],
        help="Entity labels to keep. Default: ORG",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="spaCy pipe batch size. Default: 32",
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

    # Fill missing optional columns so downstream output stays stable.
    optional_columns = ["source_url", "date", "company_seed", "raw_text"]
    for column in optional_columns:
        if column not in df.columns:
            df[column] = ""
    if "source_url_cleaned" not in df.columns:
        df["source_url_cleaned"] = df["source_url"].fillna("").astype(str)
    if "date_cleaned" not in df.columns:
        df["date_cleaned"] = df["date"].fillna("").astype(str)
    if "company_seed_cleaned" not in df.columns:
        df["company_seed_cleaned"] = df["company_seed"].fillna("").astype(str)

    nlp = load_spacy_model(args.model)
    keep_labels = {label.upper() for label in args.entity_labels}

    records: List[dict] = []
    texts = df["raw_text_cleaned"].fillna("").astype(str).tolist()

    for (_, row), doc in zip(df.iterrows(), nlp.pipe(texts, batch_size=args.batch_size)):
        for ent in doc.ents:
            if ent.label_.upper() not in keep_labels:
                continue

            records.append(
                {
                    "source_id": row["source_id"],
                    "source_type": row["source_type"],
                    "source_type_cleaned": row.get("source_type_cleaned", row["source_type"]),
                    "source_url": row.get("source_url", ""),
                    "source_url_cleaned": row.get("source_url_cleaned", ""),
                    "date": row.get("date", ""),
                    "date_cleaned": row.get("date_cleaned", ""),
                    "company_seed": row.get("company_seed", ""),
                    "company_seed_cleaned": row.get("company_seed_cleaned", ""),
                    "raw_text": row.get("raw_text", row["raw_text_cleaned"]),
                    "raw_text_cleaned": row["raw_text_cleaned"],
                    "raw_mention": ent.text,
                    "entity_label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(records)

    if output_df.empty:
        output_df = pd.DataFrame(
            columns=[
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
            ]
        )

    output_df.to_csv(output_path, index=False)
    print(f"[ner] Wrote {len(output_df)} entity mention(s) to: {output_path}")


if __name__ == "__main__":
    main()
