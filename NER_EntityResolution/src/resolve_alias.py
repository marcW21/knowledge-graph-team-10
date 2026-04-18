
#!/usr/bin/env python3
"""Resolve company aliases conservatively from NER output.

This script:
1. Reads ner_results.csv.
2. Normalizes company mentions.
3. Builds conservative auto-merge groups.
4. Routes ambiguous cases to review_queue.csv instead of force-merging them.
5. Writes resolved_entities.csv and review_queue.csv.

Design principle:
Favor missed matches over false merges.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


REQUIRED_COLUMNS = {"source_id", "raw_mention"}

GENERIC_SUFFIXES = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
}

REGION_OR_ENTITY_SUFFIXES = {
    "LLC", "LTD", "LIMITED", "PLC", "LP", "LLP",
    "BV", "AB", "GMBH", "AG", "SA", "SAS", "NV", "SPA", "SARL", "KK",
}

ALL_SUFFIXES = GENERIC_SUFFIXES | REGION_OR_ENTITY_SUFFIXES

MONTHS = {
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "SEPT", "OCT", "NOV", "DEC",
}

EXCEL_ERROR_TOKENS = {
    "#NAME?",
    "#VALUE!",
    "#REF!",
    "#DIV/0!",
    "#NUM!",
    "#N/A",
    "#NULL!",
}


def normalize_whitespace(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u00a0", " ")).strip()


def looks_like_excel_error(text: str) -> bool:
    return text.strip().upper() in EXCEL_ERROR_TOKENS


def looks_like_date_like_token(text: str) -> bool:
    cleaned = text.strip().upper()
    match = re.fullmatch(r"(\d{1,2})-([A-Z]{3,4})", cleaned)
    if match:
        return match.group(2) in MONTHS
    match = re.fullmatch(r"([A-Z]{3,4})-(\d{1,2})", cleaned)
    if match:
        return match.group(1) in MONTHS
    return False


def normalize_company_name(text: str) -> str:
    """Upper-case, remove punctuation, and collapse whitespace."""
    cleaned = normalize_whitespace(text).upper()
    cleaned = cleaned.replace("&", " AND ")
    cleaned = re.sub(r"[^A-Z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_suffixes(normalized_name: str) -> Tuple[str, List[str]]:
    """Split trailing legal suffixes from a normalized company name."""
    if not normalized_name:
        return "", []

    tokens = normalized_name.split()
    suffixes: List[str] = []

    while tokens and tokens[-1] in ALL_SUFFIXES:
        suffixes.insert(0, tokens.pop())

    base_name = " ".join(tokens).strip()
    return base_name, suffixes


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def classify_pair(name_a: str, name_b: str) -> Tuple[str, float, str]:
    """Return (decision, confidence, reason) for a pair of normalized names."""
    if not name_a or not name_b:
        return "do_not_merge", 0.0, "empty_name"

    if name_a == name_b:
        return "auto_merge", 0.99, "exact_match_after_normalization"

    if looks_like_excel_error(name_a) or looks_like_excel_error(name_b):
        return "do_not_merge", 0.0, "excel_error_token"
    if looks_like_date_like_token(name_a) or looks_like_date_like_token(name_b):
        return "do_not_merge", 0.0, "date_like_token"

    base_a, suffixes_a = split_suffixes(name_a)
    base_b, suffixes_b = split_suffixes(name_b)

    if base_a and base_a == base_b:
        if not suffixes_a and suffixes_b and set(suffixes_b).issubset(GENERIC_SUFFIXES):
            return "auto_merge", 0.88, "same_base_name_plus_generic_suffix"
        if not suffixes_b and suffixes_a and set(suffixes_a).issubset(GENERIC_SUFFIXES):
            return "auto_merge", 0.88, "same_base_name_plus_generic_suffix"

        if suffixes_a != suffixes_b:
            return "review_needed", 0.55, "same_base_name_but_suffix_differs"

    sim = similarity(name_a, name_b)
    if sim >= 0.92:
        return "review_needed", 0.60, "very_high_string_similarity_but_not_exact"
    if sim >= 0.85:
        return "review_needed", 0.50, "high_string_similarity"

    return "do_not_merge", max(0.10, round(sim * 0.30, 2)), "low_similarity"


def choose_canonical_name(raw_mentions: Sequence[str]) -> str:
    """Choose one mention string to represent the group."""
    counts = Counter(raw_mentions)

    def sort_key(name: str) -> Tuple[int, int, str]:
        # Prefer frequent + longer + lexicographically stable
        return (counts[name], len(name), name)

    return sorted(counts.keys(), key=sort_key, reverse=True)[0]


def connected_components(nodes: Iterable[str], edges: Iterable[Tuple[str, str]]) -> List[Set[str]]:
    """Find connected components in an undirected graph."""
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    all_nodes = set(nodes)

    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
        all_nodes.add(a)
        all_nodes.add(b)

    seen: Set[str] = set()
    components: List[Set[str]] = []

    for node in all_nodes:
        if node in seen:
            continue
        stack = [node]
        component: Set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(adjacency[current] - seen)
        components.append(component)

    return components


def validate_columns(columns: Sequence[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"Input CSV is missing required column(s): {missing_str}. "
            "Did you run ner_baseline.py first?"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve company aliases conservatively.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to ner_results.csv from ner_baseline.py",
    )
    parser.add_argument(
        "--resolved-output",
        required=True,
        help="Path to write resolved_entities.csv",
    )
    parser.add_argument(
        "--review-output",
        required=True,
        help="Path to write review_queue.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    resolved_output_path = Path(args.resolved_output)
    review_output_path = Path(args.review_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df.columns)
    df = df.copy()

    # Fill missing optional columns.
    for column in [
        "source_type", "source_url", "date", "company_seed",
        "raw_text", "entity_label", "start_char", "end_char"
    ]:
        if column not in df.columns:
            df[column] = ""

    df["raw_mention"] = df["raw_mention"].fillna("").astype(str)
    df["normalized_name"] = df["raw_mention"].apply(normalize_company_name)
    split_results = df["normalized_name"].apply(split_suffixes)
    df["base_name"] = split_results.apply(lambda x: x[0])
    df["legal_suffixes"] = split_results.apply(lambda x: ";".join(x[1]))

    unique_mentions = (
        df[["raw_mention", "normalized_name", "base_name", "legal_suffixes"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    normalized_names = unique_mentions["normalized_name"].tolist()

    auto_merge_edges: List[Tuple[str, str]] = []
    review_records: List[dict] = []
    review_summary: Dict[str, Tuple[float, str]] = {}

    # Only compare mentions that share the same base_name or are very similar.
    for a, b in itertools.combinations(normalized_names, 2):
        decision, confidence, reason = classify_pair(a, b)
        if decision == "auto_merge":
            auto_merge_edges.append((a, b))
        elif decision == "review_needed":
            raw_a = unique_mentions.loc[unique_mentions["normalized_name"] == a, "raw_mention"].iloc[0]
            raw_b = unique_mentions.loc[unique_mentions["normalized_name"] == b, "raw_mention"].iloc[0]
            review_records.append(
                {
                    "candidate_a": raw_a,
                    "candidate_b": raw_b,
                    "normalized_a": a,
                    "normalized_b": b,
                    "similarity_score": round(similarity(a, b), 4),
                    "review_confidence": confidence,
                    "reason": reason,
                }
            )

            for normalized_name in (a, b):
                previous = review_summary.get(normalized_name)
                if previous is None or confidence > previous[0]:
                    review_summary[normalized_name] = (confidence, reason)

    components = connected_components(normalized_names, auto_merge_edges)
    normalized_to_group: Dict[str, int] = {}
    group_to_members: Dict[int, Set[str]] = {}

    for idx, component in enumerate(components, start=1):
        group_to_members[idx] = component
        for normalized_name in component:
            normalized_to_group[normalized_name] = idx

    group_to_canonical: Dict[int, str] = {}
    group_to_confidence: Dict[int, float] = {}
    group_to_reason: Dict[int, str] = {}

    for group_id, members in group_to_members.items():
        raw_mentions_in_group = df.loc[df["normalized_name"].isin(members), "raw_mention"].tolist()
        group_to_canonical[group_id] = choose_canonical_name(raw_mentions_in_group)

        if len(members) == 1:
            only_name = next(iter(members))
            if only_name in review_summary:
                group_to_confidence[group_id] = review_summary[only_name][0]
                group_to_reason[group_id] = review_summary[only_name][1]
            else:
                group_to_confidence[group_id] = 1.00
                group_to_reason[group_id] = "singleton_no_merge_needed"
        else:
            # Conservative default for auto-merged groups.
            group_to_confidence[group_id] = 0.88
            group_to_reason[group_id] = "auto_merge_component"

    def decision_for_group(group_id: int) -> str:
        members = group_to_members[group_id]
        if len(members) > 1:
            return "auto_merge"
        only_name = next(iter(members))
        if only_name in review_summary:
            return "review_needed"
        return "singleton"

    df["canonical_group_id"] = df["normalized_name"].map(normalized_to_group)
    df["canonical_name"] = df["canonical_group_id"].map(group_to_canonical)
    df["merge_confidence"] = df["canonical_group_id"].map(group_to_confidence)
    df["merge_decision"] = df["canonical_group_id"].map(decision_for_group)
    df["evidence_note"] = df["canonical_group_id"].map(group_to_reason)

    resolved_columns = [
        "source_id",
        "source_type",
        "source_url",
        "date",
        "company_seed",
        "raw_text",
        "raw_mention",
        "entity_label",
        "start_char",
        "end_char",
        "normalized_name",
        "base_name",
        "legal_suffixes",
        "canonical_group_id",
        "canonical_name",
        "merge_confidence",
        "merge_decision",
        "evidence_note",
    ]
    resolved_df = df[resolved_columns].copy()
    review_df = pd.DataFrame(review_records)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    review_output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_df.to_csv(resolved_output_path, index=False)
    review_df.to_csv(review_output_path, index=False)

    print(f"[resolve_alias] Wrote {len(resolved_df)} resolved row(s) to: {resolved_output_path}")
    print(f"[resolve_alias] Wrote {len(review_df)} review pair(s) to: {review_output_path}")


if __name__ == "__main__":
    main()
