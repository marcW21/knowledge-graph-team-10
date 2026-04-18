\
#!/usr/bin/env python3
"""Extract simple candidate company-company relationships from resolved entity rows.

This is a first-pass heuristic extractor meant for integration with validation/confidence-scoring stage.

Expected upstream flow:
    preprocess.py -> ner_baseline.py -> resolve_alias.py -> extract_candidate_relations.py

Input:
    resolved_entities.csv from resolve_alias.py

Output:
    candidate_relations.csv with one row per candidate relationship, including:
    - entity_a / entity_b
    - candidate_relation
    - evidence_text
    - source metadata
    - trigger phrase
    - extraction method

Current scope:
    - company-company relationships only
    - heuristic sentence-level extraction using trigger words
    - conservative defaults to avoid over-claiming
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

REQUIRED_COLUMNS = {
    "source_id",
    "source_type",
    "source_url",
    "date",
    "raw_text",
    "raw_mention",
    "canonical_name",
    "start_char",
    "end_char",
}

RELATION_PATTERNS = [
    {
        "relation": "ACQUIRED",
        "trigger_regex": r"\b(acquired|acquire|acquires|buy|bought|purchased|merged with|acquisition of)\b",
        "priority": 1,
    },
    {
        "relation": "LICENSED_FROM",
        "trigger_regex": r"\b(licensed from|licensed to|licensed|licensing agreement|license agreement|entered into a license)\b",
        "priority": 2,
    },
    {
        "relation": "FILED_PATENT_WITH",
        "trigger_regex": r"\b(filed patent with|co-filed|joint patent|patent with|co-assignee|co assigned)\b",
        "priority": 3,
    },
    {
        "relation": "COLLABORATED_WITH",
        "trigger_regex": r"\b(collaborated with|collaboration with|partnered with|partnership with|jointly developed|worked with)\b",
        "priority": 4,
    },
]

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_REGEX = re.compile(r"\s+")


def normalize_whitespace(value: object) -> str:
    if pd.isna(value):
        return ""
    return WHITESPACE_REGEX.sub(" ", str(value).replace("\u00a0", " ")).strip()


def validate_columns(columns: Sequence[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"Input CSV is missing required column(s): {missing_str}. "
            "Did you run resolve_alias.py first?"
        )


def sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return sentence spans as (start, end, sentence_text)."""
    text = normalize_whitespace(text)
    if not text:
        return []

    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for part in SENTENCE_SPLIT_REGEX.split(text):
        part = part.strip()
        if not part:
            continue
        start = text.find(part, cursor)
        end = start + len(part)
        spans.append((start, end, part))
        cursor = end
    if not spans:
        spans.append((0, len(text), text))
    return spans


def assign_sentence(raw_text: str, start_char: object, raw_mention: str) -> str:
    """Attach a mention to the sentence that contains it; fall back safely."""
    text = normalize_whitespace(raw_text)
    if not text:
        return ""

    try:
        mention_start = int(start_char)
    except Exception:
        mention_start = -1

    for sent_start, sent_end, sent_text in sentence_spans(text):
        if mention_start >= 0 and sent_start <= mention_start < sent_end:
            return sent_text

    # Fallback: first sentence containing the raw mention text
    mention = normalize_whitespace(raw_mention)
    if mention:
        for _, _, sent_text in sentence_spans(text):
            if mention.lower() in sent_text.lower():
                return sent_text

    # Last fallback: the whole text
    return text


def detect_relation(sentence: str) -> Tuple[str, str]:
    """Return (relation_label, matched_trigger)."""
    lower_sentence = sentence.lower()
    for spec in sorted(RELATION_PATTERNS, key=lambda x: x["priority"]):
        match = re.search(spec["trigger_regex"], lower_sentence, flags=re.IGNORECASE)
        if match:
            return spec["relation"], match.group(0)
    return "", ""


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def collect_sentence_level_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Add a sentence_text column for each resolved mention row."""
    df = df.copy()
    df["sentence_text"] = df.apply(
        lambda row: assign_sentence(
            row.get("raw_text", ""),
            row.get("start_char", -1),
            row.get("raw_mention", ""),
        ),
        axis=1,
    )
    return df


def build_candidate_rows(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Extract candidate company-company relations from shared sentences."""
    candidate_rows: List[Dict[str, object]] = []

    group_cols = [
        "source_id",
        "source_type",
        "source_url",
        "date",
        "sentence_text",
    ]

    for group_key, group_df in df.groupby(group_cols, dropna=False):
        source_id, source_type, source_url, date, sentence_text = group_key

        sentence_text = normalize_whitespace(sentence_text)
        if not sentence_text:
            continue

        relation_label, trigger = detect_relation(sentence_text)
        if not relation_label:
            continue

        canonical_entities = dedupe_preserve_order(
            normalize_whitespace(x) for x in group_df["canonical_name"].tolist() if normalize_whitespace(x)
        )

        # Need at least 2 distinct companies in the sentence to form a company-company relation.
        if len(canonical_entities) < 2:
            continue

        # Create one candidate per pair of entities in this sentence.
        for entity_a, entity_b in itertools.combinations(canonical_entities, 2):
            candidate_rows.append(
                {
                    "source_id": source_id,
                    "source_type": source_type,
                    "source_url": source_url,
                    "date": date,
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "candidate_relation": relation_label,
                    "evidence_text": sentence_text,
                    "trigger_phrase": trigger,
                    "extraction_method": "heuristic_sentence_trigger",
                    "num_entities_in_sentence": len(canonical_entities),
                }
            )

    return candidate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract candidate company-company relationships.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to resolved_entities.csv from resolve_alias.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write candidate_relations.csv",
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
    df = collect_sentence_level_entities(df)

    candidate_rows = build_candidate_rows(df)
    candidate_df = pd.DataFrame(candidate_rows)

    # Always write a CSV with headers, even if there are no rows.
    if candidate_df.empty:
        candidate_df = pd.DataFrame(
            columns=[
                "source_id",
                "source_type",
                "source_url",
                "date",
                "entity_a",
                "entity_b",
                "candidate_relation",
                "evidence_text",
                "trigger_phrase",
                "extraction_method",
                "num_entities_in_sentence",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(output_path, index=False)

    print(f"[extract_candidate_relations] Wrote {len(candidate_df)} candidate row(s) to: {output_path}")


if __name__ == "__main__":
    main()
