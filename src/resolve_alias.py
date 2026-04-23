#!/usr/bin/env python3
"""Resolve company aliases conservatively from NER output.

Pipeline position:  ner.py → resolve_alias.py → extract_candidate_relations.py

Design principle: favour missed matches over false merges.

Bug fix vs original
-------------------
The original code called `itertools.combinations` over ALL normalized names
(O(n²)) even after grouping by base_name, because it built `pairs_to_compare`
from `normalized_names` rather than from within each group's member list.
This caused spurious cross-company review entries and wasted CPU on completely
unrelated pairs. The fix: iterate combinations only within each base_name group.
"""

from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import pandas as pd

from constants import ALL_LEGAL_SUFFIXES, GENERIC_SUFFIXES

REQUIRED_COLUMNS: frozenset[str] = frozenset({"source_id", "raw_mention"})

_WS_RE = re.compile(r"\s+")


# ─── Text helpers ─────────────────────────────────────────────────────────────

def _norm_ws(value: object) -> str:
    if pd.isna(value):
        return ""
    return _WS_RE.sub(" ", str(value).replace("\u00a0", " ")).strip()


def normalize_company_name(text: str) -> str:
    """Upper-case, replace & → AND, strip non-alphanumeric, collapse spaces."""
    s = _norm_ws(text).upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_suffixes(normalized: str) -> Tuple[str, List[str]]:
    """Return (base_name, [trailing_legal_suffixes])."""
    if not normalized:
        return "", []
    tokens = normalized.split()
    suffixes: List[str] = []
    while tokens and tokens[-1] in ALL_LEGAL_SUFFIXES:
        suffixes.insert(0, tokens.pop())
    return " ".join(tokens).strip(), suffixes


# ─── Pair classification ──────────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def classify_pair(norm_a: str, norm_b: str) -> Tuple[str, float, str]:
    """Return (decision, confidence, reason).

    Decisions: "auto_merge" | "review_needed" | "do_not_merge"
    """
    if not norm_a or not norm_b:
        return "do_not_merge", 0.0, "empty_name"
    if norm_a == norm_b:
        return "auto_merge", 0.99, "exact_match_after_normalization"

    base_a, suf_a = split_suffixes(norm_a)
    base_b, suf_b = split_suffixes(norm_b)

    if base_a and base_a == base_b:
        # One side has no suffix: safe to merge only when the suffix is generic.
        if not suf_a and suf_b and set(suf_b).issubset(GENERIC_SUFFIXES):
            return "auto_merge", 0.88, "same_base_name_plus_generic_suffix"
        if not suf_b and suf_a and set(suf_a).issubset(GENERIC_SUFFIXES):
            return "auto_merge", 0.88, "same_base_name_plus_generic_suffix"
        if suf_a != suf_b:
            return "review_needed", 0.55, "same_base_name_but_suffix_differs"

    sim = _similarity(norm_a, norm_b)
    if sim >= 0.92:
        return "review_needed", 0.60, "very_high_string_similarity_but_not_exact"
    if sim >= 0.85:
        return "review_needed", 0.50, "high_string_similarity"
    return "do_not_merge", max(0.10, round(sim * 0.30, 2)), "low_similarity"


# ─── Graph helpers ────────────────────────────────────────────────────────────

def connected_components(
    nodes: Set[str],
    edges: List[Tuple[str, str]],
) -> List[Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
        nodes.add(a)
        nodes.add(b)

    seen: Set[str] = set()
    components: List[Set[str]] = []
    for node in nodes:
        if node in seen:
            continue
        stack, component = [node], set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            component.add(cur)
            stack.extend(adj[cur] - seen)
        components.append(component)
    return components


# ─── Canonical name selection ─────────────────────────────────────────────────

def choose_canonical(raw_mentions: List[str]) -> str:
    """Prefer title-case, then most frequent, then longest, then lexicographic."""
    counts = Counter(raw_mentions)

    def _key(name: str) -> Tuple[int, int, int, str]:
        words = name.split()
        is_title = bool(words) and all(
            w[0].isupper() if w[0].isalpha() else True for w in words
        ) and not name.isupper()
        return (int(is_title), counts[name], len(name), name)

    return max(counts, key=_key)


# ─── Within-group pair iteration (the bug-fix) ────────────────────────────────

def _within_group_pairs(
    base_to_norms: Dict[str, List[str]],
) -> Iterator[Tuple[str, str]]:
    """Yield (a, b) pairs only within the same base-name group.

    This is the key fix: the original iterated combinations over the full
    unique-names list, accidentally comparing names across completely different
    companies and flooding review_queue with spurious low-confidence pairs.
    """
    for members in base_to_norms.values():
        if len(members) < 2:
            continue
        for a, b in itertools.combinations(sorted(set(members)), 2):
            yield min(a, b), max(a, b)


# ─── Main resolution logic ────────────────────────────────────────────────────

def resolve(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (resolved_df, review_df)."""
    df = df.copy()

    for col in ["source_type", "source_url", "date", "company_seed",
                "raw_text", "entity_label", "start_char", "end_char"]:
        if col not in df.columns:
            df[col] = ""

    df["raw_mention"] = df["raw_mention"].fillna("").astype(str)
    df["normalized_name"] = df["raw_mention"].apply(normalize_company_name)

    split_result = df["normalized_name"].apply(split_suffixes)
    df["base_name"] = split_result.apply(lambda x: x[0])
    df["legal_suffixes"] = split_result.apply(lambda x: ";".join(x[1]))

    # Map normalized name → first raw mention (for review output labelling).
    norm_to_raw: Dict[str, str] = (
        df[["normalized_name", "raw_mention"]]
        .drop_duplicates(subset=["normalized_name"])
        .set_index("normalized_name")["raw_mention"]
        .to_dict()
    )

    # Build base_name → [normalized_names] index.
    base_to_norms: Dict[str, List[str]] = defaultdict(list)
    for norm in df["normalized_name"].unique():
        base, _ = split_suffixes(norm)
        base_to_norms[base or norm].append(norm)

    merge_edges: List[Tuple[str, str]] = []
    review_records: List[dict] = []
    # norm → (best_confidence, reason) for singletons that appeared in a review pair
    review_summary: Dict[str, Tuple[float, str]] = {}

    seen_pairs: Set[Tuple[str, str]] = set()
    for a, b in _within_group_pairs(base_to_norms):
        if (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))

        decision, confidence, reason = classify_pair(a, b)
        if decision == "auto_merge":
            merge_edges.append((a, b))
        elif decision == "review_needed":
            review_records.append({
                "candidate_a": norm_to_raw.get(a, a),
                "candidate_b": norm_to_raw.get(b, b),
                "normalized_a": a,
                "normalized_b": b,
                "similarity_score": round(_similarity(a, b), 4),
                "review_confidence": confidence,
                "reason": reason,
            })
            for norm in (a, b):
                prev = review_summary.get(norm)
                if prev is None or confidence > prev[0]:
                    review_summary[norm] = (confidence, reason)

    all_norms: Set[str] = set(df["normalized_name"].unique())
    components = connected_components(all_norms, merge_edges)

    norm_to_group: Dict[str, int] = {}
    group_members: Dict[int, Set[str]] = {}
    for gid, comp in enumerate(components, start=1):
        group_members[gid] = comp
        for n in comp:
            norm_to_group[n] = gid

    group_canonical: Dict[int, str] = {}
    group_confidence: Dict[int, float] = {}
    group_reason: Dict[int, str] = {}
    group_decision: Dict[int, str] = {}

    for gid, members in group_members.items():
        raw_in_group = df.loc[df["normalized_name"].isin(members), "raw_mention"].tolist()
        group_canonical[gid] = choose_canonical(raw_in_group)

        if len(members) > 1:
            group_confidence[gid] = 0.88
            group_reason[gid] = "auto_merge_component"
            group_decision[gid] = "auto_merge"
        else:
            only = next(iter(members))
            if only in review_summary:
                conf, reason = review_summary[only]
                group_confidence[gid] = conf
                group_reason[gid] = reason
                group_decision[gid] = "review_needed"
            else:
                group_confidence[gid] = 1.00
                group_reason[gid] = "singleton_no_merge_needed"
                group_decision[gid] = "singleton"

    df["canonical_group_id"] = df["normalized_name"].map(norm_to_group)
    df["canonical_name"] = df["canonical_group_id"].map(group_canonical)
    df["merge_confidence"] = df["canonical_group_id"].map(group_confidence)
    df["merge_decision"] = df["canonical_group_id"].map(group_decision)
    df["evidence_note"] = df["canonical_group_id"].map(group_reason)

    resolved_cols = [
        "source_id", "source_type", "source_url", "date", "company_seed",
        "raw_text", "raw_mention", "entity_label", "start_char", "end_char",
        "normalized_name", "base_name", "legal_suffixes",
        "canonical_group_id", "canonical_name",
        "merge_confidence", "merge_decision", "evidence_note",
    ]
    resolved_df = df[[c for c in resolved_cols if c in df.columns]].copy()
    review_df = pd.DataFrame(review_records)
    return resolved_df, review_df


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conservative company alias resolution.")
    p.add_argument("--input", required=True, help="ner_results CSV from ner.py")
    p.add_argument("--resolved-output", required=True)
    p.add_argument("--review-output", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    resolved_df, review_df = resolve(df)

    for path, frame, label in [
        (args.resolved_output, resolved_df, "resolved"),
        (args.review_output, review_df, "review"),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"[resolve_alias] Wrote {len(frame)} {label} row(s) to: {path}")


if __name__ == "__main__":
    main()
