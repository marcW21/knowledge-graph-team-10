#!/usr/bin/env python3
"""Source-aware candidate relation extraction from resolved entity rows.

Relaxed version to keep recall healthier while avoiding the noisiest metadata explosions.
Main changes:
1. keeps original source-aware trigger extraction
2. broadens trigger patterns moderately
3. adds a controlled record-level fallback when raw text contains relation evidence
4. adds optional CO_OCCURS_WITH fallback for SEC/PUBMED sentences with >=2 plausible orgs
5. still blocks obvious metadata/product/inventor noise, especially for USPTO
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

REQUIRED_COLUMNS = {
    "source_id", "source_type", "source_url", "date", "raw_text", "canonical_name", "start_char", "end_char",
}

SEC_PATTERNS = [
    ("ACQUIRED", r"\b(acquired|acquire|acquires|buy|bought|purchased|acquisition of|merged with|merger with|acquisition agreement)\b"),
    ("LICENSED_FROM", r"\b(licensed from|licensed to|licensed|licensing agreement|license agreement|entered into a license|license deal|exclusive license|non-exclusive license)\b"),
    ("PARTNERED_WITH", r"\b(partnership|partnered with|strategic partnership|collaboration agreement|entered into an agreement with|agreement with)\b"),
    ("COLLABORATED_WITH", r"\b(collaborated with|collaboration with|co-develop(?:ed)? with|jointly developed|working with|worked with)\b"),
    ("FUNDED_WITH", r"\b(funded by .* and .*|co-funded by|supported by .* and .*|sponsored by .* and .*)\b"),
]

PUBMED_PATTERNS = [
    ("PARTNERED_WITH", r"\b(partnership|partnered with|in partnership with|agreement with)\b"),
    ("COLLABORATED_WITH", r"\b(collaborated with|collaboration with|in collaboration with|co-develop(?:ed)? with|jointly developed|worked with|developed with)\b"),
    ("FUNDED_WITH", r"\b(funded by .* and .*|co-funded by|supported by .* and .*|sponsored by .* and .*)\b"),
]

USPTO_PATTERNS = [
    ("FILED_PATENT_WITH", r"\b(co-assignee|co-assigned|co assigned|co-filed|joint patent|filed patent with|patent with|joint assignee|assignees)\b"),
]

NEGATIVE_TRIGGER_PATTERNS = [r"\bcommunity-acquired\b", r"\bhospital-acquired\b", r"\bventilator-associated\b"]
SEC_BOILERPLATE_PATTERNS = [
    r"\bcurrent report on form 8-k\b", r"\bforward-looking statements\b",
    r"\bprivate securities litigation reform act\b", r"\bcommission file\b",
    r"\bcommon stock\b", r"\bsecurities exchange act\b",
]
ENTITY_BLOCKLIST_EXACT = {
    "COMPANY", "REGISTRANT", "COMMON STOCK", "COMMISSION FILE", "DISTRICT COURT",
    "THE SECURITIES AND EXCHANGE COMMISSION", "SEC", "FDA", "NIH",
    "THE LICENSE AGREEMENT", "LICENSE AGREEMENT", "THE MERGER AGREEMENT", "MERGER AGREEMENT",
    "AGREEMENT", "TERRITORY", "THE LICENSED PRODUCT", "LICENSED PRODUCT",
    "PURCHASER", "SHARES", "CVR", "CI", "CAPITA", "DSO", "NJ",
    "CPC", "IPC", "USPC", "INVENTOR", "INVENTORS", "ASSIGNEE", "PATENT", "GRANT", "APPLICATION",
}
ENTITY_BLOCKLIST_SUBSTRINGS = [
    "&#", "FORM 8-K", "FORM 10-K", "FORM 10-Q", "COMMON STOCK", "COMMISSION FILE", "DISTRICT COURT",
    "SECURITIES EXCHANGE ACT", "LICENSE AGREEMENT", "MERGER AGREEMENT", "LICENSED PRODUCT", "TERRITORY",
    "CURRENT REPORT", "GRANT YEAR", "PUBLICATION", "APPLICATION NO", "PATENT NO", "INVENTOR", "INVENTORS",
    "CPC", "IPC", "USPC", "ASSIGNEE:",
]
PRODUCTISH_PATTERNS = [r"\bBAQSIMI\b", r"\bPRIMATENE\b", r"\bMIST\b", r"\bGLUCAGON\b", r"\bPCV\d+\b"]
USPTO_METADATA_PATTERNS = [
    r"^\s*CPC\s*$", r"^\s*IPC\s*$", r"^\s*USPC\s*$", r"^\s*PATENT\s*$", r"^\s*ASSIGNEE\s*$",
    r"^\s*INVENTOR[S]?\s*$", r"^\s*PUBLICATION\s*$", r"^\s*APPLICATION\s*$",
]
EXACT_ALLOWLIST = {"National Institutes of Health", "University of Pennsylvania"}

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_REGEX = re.compile(r"\s+")
RECORD_LEVEL_RELATION_PATTERNS = SEC_PATTERNS + PUBMED_PATTERNS + USPTO_PATTERNS


def normalize_whitespace(value: object) -> str:
    if pd.isna(value):
        return ""
    return WHITESPACE_REGEX.sub(" ", str(value).replace("\u00a0", " ")).strip()


def validate_columns(columns: Sequence[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {', '.join(sorted(missing))}. Did you run resolve_alias.py first?")


def sentence_spans(text: str) -> List[Tuple[int, int, str]]:
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


def assign_sentence(raw_text: str, start_char: object) -> str:
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
    return text


def contains_negative_trigger(sentence: str) -> bool:
    lower_sentence = sentence.lower()
    return any(re.search(pat, lower_sentence) for pat in NEGATIVE_TRIGGER_PATTERNS)


def looks_like_sec_boilerplate(sentence: str) -> bool:
    lower_sentence = sentence.lower()
    return any(re.search(pat, lower_sentence) for pat in SEC_BOILERPLATE_PATTERNS)


def patterns_for_source(source_type: str):
    source = str(source_type).upper()
    if source == "SEC":
        return SEC_PATTERNS
    if source == "PUBMED":
        return PUBMED_PATTERNS
    if source == "USPTO":
        return USPTO_PATTERNS
    return RECORD_LEVEL_RELATION_PATTERNS


def detect_relation(text: str, source_type: str) -> Tuple[str, str]:
    lower_text = text.lower()
    if contains_negative_trigger(lower_text):
        return "", ""
    for relation, regex in patterns_for_source(source_type):
        match = re.search(regex, lower_text, flags=re.IGNORECASE)
        if match:
            return relation, match.group(0)
    return "", ""


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def looks_like_company_entity(text: str, source_type: str = "") -> bool:
    entity = normalize_whitespace(text)
    if not entity:
        return False
    upper = entity.upper()
    if entity in EXACT_ALLOWLIST or upper in {x.upper() for x in EXACT_ALLOWLIST}:
        return True
    if upper in ENTITY_BLOCKLIST_EXACT:
        return False
    if any(bad in upper for bad in ENTITY_BLOCKLIST_SUBSTRINGS):
        return False
    if any(re.search(pat, upper) for pat in PRODUCTISH_PATTERNS):
        return False
    if any(re.search(pat, upper) for pat in USPTO_METADATA_PATTERNS):
        return False
    if entity.isdigit() or not re.search(r"[A-Za-z]", entity):
        return False
    bare = re.sub(r"[^A-Za-z0-9&]", "", entity)
    if len(bare) <= 2:
        return False
    if str(source_type).upper() == "USPTO":
        if re.fullmatch(r"[A-Z0-9\-]{2,8}", upper) and upper not in {"BMS", "GSK", "JNJ", "CSL"}:
            return False
    return True


def collect_sentence_level_entities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentence_text"] = df.apply(lambda row: assign_sentence(row.get("raw_text", ""), row.get("start_char", -1)), axis=1)
    return df


def patent_like_text(text: str) -> bool:
    lower = text.lower()
    return ("patent" in lower) or ("assignee" in lower) or ("inventor" in lower)


def has_explicit_uspto_joint_signal(text: str) -> bool:
    lower = text.lower()
    joint_patterns = [
        r"\bco-assignee\b", r"\bco-assigned\b", r"\bco assigned\b", r"\bjoint patent\b",
        r"\bpatent with\b", r"\bfiled patent with\b", r"\bco-filed\b", r"\bjoint assignee\b",
        r"\band\b.*\bassignee\b", r"\bassignees?\b.*\band\b",
    ]
    return any(re.search(pat, lower) for pat in joint_patterns)


def add_pair_rows(candidate_rows: List[Dict[str, object]], entities: List[str], row_base: Dict[str, object]) -> None:
    for entity_a, entity_b in itertools.combinations(entities, 2):
        if entity_a == entity_b:
            continue
        candidate_rows.append({**row_base, "entity_a": entity_a, "entity_b": entity_b})


def build_candidate_rows(df: pd.DataFrame, enable_cooccur_fallback: bool = True) -> List[Dict[str, object]]:
    candidate_rows: List[Dict[str, object]] = []
    group_cols = ["source_id", "source_type", "source_url", "date", "sentence_text"]

    for group_key, group_df in df.groupby(group_cols, dropna=False):
        source_id, source_type, source_url, date, sentence_text = group_key
        source_type = str(source_type).upper()
        sentence_text = normalize_whitespace(sentence_text)
        if not sentence_text:
            continue
        if source_type == "SEC" and looks_like_sec_boilerplate(sentence_text):
            continue

        canonical_entities = dedupe_preserve_order(
            normalize_whitespace(x)
            for x in group_df["canonical_name"].tolist()
            if normalize_whitespace(x) and looks_like_company_entity(x, source_type)
        )
        if len(canonical_entities) < 2:
            continue

        relation_label, trigger = detect_relation(sentence_text, source_type)
        if relation_label:
            add_pair_rows(candidate_rows, canonical_entities, {
                "source_id": source_id,
                "source_type": source_type,
                "source_url": source_url,
                "date": date,
                "candidate_relation": relation_label,
                "evidence_text": sentence_text,
                "trigger_phrase": trigger,
                "extraction_method": "source_aware_sentence_trigger_relaxed",
                "num_entities_in_sentence": len(canonical_entities),
            })
            continue

        # controlled fallback: co-occurrence only for SEC/PUBMED when sentence is not too crowded
        if enable_cooccur_fallback and source_type in {"SEC", "PUBMED"} and 2 <= len(canonical_entities) <= 4:
            lower_sentence = sentence_text.lower()
            soft_relation_words = [
                "agreement", "collaboration", "partner", "license", "licensed", "acquired", "merger",
                "joint", "together", "with", "between",
            ]
            if any(word in lower_sentence for word in soft_relation_words):
                add_pair_rows(candidate_rows, canonical_entities, {
                    "source_id": source_id,
                    "source_type": source_type,
                    "source_url": source_url,
                    "date": date,
                    "candidate_relation": "CO_OCCURS_WITH",
                    "evidence_text": sentence_text,
                    "trigger_phrase": "soft_relation_context",
                    "extraction_method": "sentence_cooccurrence_fallback_relaxed",
                    "num_entities_in_sentence": len(canonical_entities),
                })

    # record-level fallback when sentence split misses trigger but raw record still has clear evidence
    record_cols = ["source_id", "source_type", "source_url", "date", "raw_text"]
    for group_key, group_df in df.groupby(record_cols, dropna=False):
        source_id, source_type, source_url, date, raw_text = group_key
        source_type = str(source_type).upper()
        raw_text = normalize_whitespace(raw_text)
        if not raw_text:
            continue

        canonical_entities = dedupe_preserve_order(
            normalize_whitespace(x)
            for x in group_df["canonical_name"].tolist()
            if normalize_whitespace(x) and looks_like_company_entity(x, source_type)
        )
        if len(canonical_entities) < 2:
            continue

        if source_type == "USPTO":
            if patent_like_text(raw_text) and has_explicit_uspto_joint_signal(raw_text):
                add_pair_rows(candidate_rows, canonical_entities, {
                    "source_id": source_id,
                    "source_type": "USPTO",
                    "source_url": source_url,
                    "date": date,
                    "candidate_relation": "FILED_PATENT_WITH",
                    "evidence_text": raw_text[:500],
                    "trigger_phrase": "explicit_joint_patent_signal",
                    "extraction_method": "source_aware_record_trigger_relaxed",
                    "num_entities_in_sentence": len(canonical_entities),
                })
            continue

        relation_label, trigger = detect_relation(raw_text[:2000], source_type)
        if relation_label and len(canonical_entities) <= 5:
            add_pair_rows(candidate_rows, canonical_entities, {
                "source_id": source_id,
                "source_type": source_type,
                "source_url": source_url,
                "date": date,
                "candidate_relation": relation_label,
                "evidence_text": raw_text[:500],
                "trigger_phrase": trigger,
                "extraction_method": "record_trigger_fallback_relaxed",
                "num_entities_in_sentence": len(canonical_entities),
            })

    return candidate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract candidate company-company relationships.")
    parser.add_argument("--input", required=True, help="Path to resolved_entities.csv from resolve_alias.py")
    parser.add_argument("--output", required=True, help="Path to write candidate_relations.csv")
    parser.add_argument("--disable-cooccur-fallback", action="store_true", help="Disable relaxed SEC/PubMed co-occurrence fallback")
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
    candidate_rows = build_candidate_rows(df, enable_cooccur_fallback=not args.disable_cooccur_fallback)
    candidate_df = pd.DataFrame(candidate_rows)

    if not candidate_df.empty:
        # make pair order stable for better dedupe
        candidate_df[["entity_a", "entity_b"]] = candidate_df.apply(
            lambda row: pd.Series(sorted([row["entity_a"], row["entity_b"]])), axis=1
        )
        candidate_df = candidate_df.drop_duplicates(
            subset=[
                "source_id", "source_type", "source_url", "date",
                "entity_a", "entity_b", "candidate_relation",
                "evidence_text", "trigger_phrase", "extraction_method",
            ]
        ).reset_index(drop=True)
    else:
        candidate_df = pd.DataFrame(columns=[
            "source_id", "source_type", "source_url", "date", "entity_a", "entity_b",
            "candidate_relation", "evidence_text", "trigger_phrase", "extraction_method",
            "num_entities_in_sentence",
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(output_path, index=False)
    print(f"[extract_candidate_relations_relaxed] Wrote {len(candidate_df)} candidate row(s) to: {output_path}")


if __name__ == "__main__":
    main()
