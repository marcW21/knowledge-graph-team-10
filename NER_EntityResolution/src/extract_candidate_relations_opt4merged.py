\
#!/usr/bin/env python3
"""Source-aware candidate relation extraction from resolved entity rows.

Revised to avoid USPTO metadata explosions while still supporting merged stage 1 inputs.

Key changes vs prior source-aware version:
1. USPTO company-company edges are ONLY created when there is explicit co-assignee / joint-patent style evidence.
2. Stronger filtering of metadata-like entities (CPC, IPC, inventors, filing fields, etc.).
3. Moderate SEC and PubMed filtering retained.
4. Keeps output schema compatible with downstream Andrew validation.

Expected upstream flow:
    preprocess.py -> ner_*.py -> resolve_alias.py -> extract_candidate_relations_stage1_v2.py
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
    "canonical_name",
    "start_char",
    "end_char",
}

SEC_PATTERNS = [
    ("ACQUIRED", r"\b(acquired|acquire|acquires|buy|bought|purchased|acquisition of|merged with|merger with)\b"),
    ("LICENSED_FROM", r"\b(licensed from|licensed to|licensed|licensing agreement|license agreement|entered into a license|license deal)\b"),
    ("PARTNERED_WITH", r"\b(partnership|partnered with|strategic partnership|collaboration agreement)\b"),
    ("COLLABORATED_WITH", r"\b(collaborated with|collaboration with|co-develop(?:ed)? with|jointly developed)\b"),
]

PUBMED_PATTERNS = [
    ("PARTNERED_WITH", r"\b(partnership|partnered with|in partnership with)\b"),
    ("COLLABORATED_WITH", r"\b(collaborated with|collaboration with|in collaboration with|co-develop(?:ed)? with|jointly developed|worked with)\b"),
    ("FUNDED_WITH", r"\b(funded by .* and .*|co-funded by|supported by .* and .*|sponsored by .* and .*)\b"),
]

USPTO_PATTERNS = [
    ("FILED_PATENT_WITH", r"\b(co-assignee|co-assigned|co assigned|co-filed|joint patent|filed patent with|patent with)\b"),
]

NEGATIVE_TRIGGER_PATTERNS = [
    r"\bcommunity-acquired\b",
    r"\bhospital-acquired\b",
    r"\bventilator-associated\b",
]

SEC_BOILERPLATE_PATTERNS = [
    r"\bcurrent report on form 8-k\b",
    r"\bforward-looking statements\b",
    r"\bprivate securities litigation reform act\b",
    r"\bcommission file\b",
    r"\bcommon stock\b",
    r"\bsecurities exchange act\b",
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
    "&#",
    "FORM 8-K", "FORM 10-K", "FORM 10-Q", "COMMON STOCK",
    "COMMISSION FILE", "DISTRICT COURT", "SECURITIES EXCHANGE ACT",
    "LICENSE AGREEMENT", "MERGER AGREEMENT", "LICENSED PRODUCT", "TERRITORY",
    "CURRENT REPORT", "GRANT YEAR", "PUBLICATION", "APPLICATION NO", "PATENT NO",
    "INVENTOR", "INVENTORS", "CPC", "IPC", "USPC", "ASSIGNEE:",
]

PRODUCTISH_PATTERNS = [
    r"\bBAQSIMI\b",
    r"\bPRIMATENE\b",
    r"\bMIST\b",
    r"\bGLUCAGON\b",
    r"\bPCV\d+\b",
]

USPTO_METADATA_PATTERNS = [
    r"^\s*CPC\s*$",
    r"^\s*IPC\s*$",
    r"^\s*USPC\s*$",
    r"^\s*PATENT\s*$",
    r"^\s*ASSIGNEE\s*$",
    r"^\s*INVENTOR[S]?\s*$",
    r"^\s*PUBLICATION\s*$",
    r"^\s*APPLICATION\s*$",
]

EXACT_ALLOWLIST = {
    "National Institutes of Health", "University of Pennsylvania",
}

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


def detect_relation(sentence: str, source_type: str) -> Tuple[str, str]:
    lower_sentence = sentence.lower()

    if contains_negative_trigger(lower_sentence):
        return "", ""

    source = str(source_type).upper()
    if source == "SEC":
        patterns = SEC_PATTERNS
    elif source == "PUBMED":
        patterns = PUBMED_PATTERNS
    elif source == "USPTO":
        patterns = USPTO_PATTERNS
    else:
        patterns = SEC_PATTERNS + PUBMED_PATTERNS + USPTO_PATTERNS

    for relation, regex in patterns:
        match = re.search(regex, lower_sentence, flags=re.IGNORECASE)
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
    if entity.isdigit():
        return False
    if not re.search(r"[A-Za-z]", entity):
        return False

    bare = re.sub(r"[^A-Za-z0-9&]", "", entity)
    if len(bare) <= 2:
        return False

    # Block most all-caps short metadata-like tokens in USPTO.
    if str(source_type).upper() == "USPTO":
        if re.fullmatch(r"[A-Z0-9\-]{2,8}", upper) and upper not in {"BMS", "GSK", "JNJ"}:
            return False

    if entity.lower() == entity and " " in entity:
        return False

    return True


def collect_sentence_level_entities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentence_text"] = df.apply(
        lambda row: assign_sentence(
            row.get("raw_text", ""),
            row.get("start_char", -1),
        ),
        axis=1,
    )
    return df


def patent_like_text(text: str) -> bool:
    lower = text.lower()
    return ("patent" in lower) or ("assignee" in lower) or ("inventor" in lower)


def has_explicit_uspto_joint_signal(text: str) -> bool:
    lower = text.lower()
    joint_patterns = [
        r"\bco-assignee\b",
        r"\bco-assigned\b",
        r"\bco assigned\b",
        r"\bjoint patent\b",
        r"\bpatent with\b",
        r"\bfiled patent with\b",
        r"\bco-filed\b",
        r"\band\b.*\bassignee\b",
        r"\bassignees?\b.*\band\b",
    ]
    return any(re.search(pat, lower) for pat in joint_patterns)


def build_candidate_rows(df: pd.DataFrame) -> List[Dict[str, object]]:
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
        if not relation_label:
            continue

        for entity_a, entity_b in itertools.combinations(canonical_entities, 2):
            if entity_a == entity_b:
                continue
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
                    "extraction_method": "source_aware_sentence_trigger_v2",
                    "num_entities_in_sentence": len(canonical_entities),
                }
            )

    # Additional record-level pass for USPTO ONLY when there is explicit joint/co-assignee evidence.
    uspto_cols = ["source_id", "source_type", "source_url", "date", "raw_text"]
    for group_key, group_df in df.groupby(uspto_cols, dropna=False):
        source_id, source_type, source_url, date, raw_text = group_key
        if str(source_type).upper() != "USPTO":
            continue

        raw_text = normalize_whitespace(raw_text)
        if not patent_like_text(raw_text):
            continue
        if not has_explicit_uspto_joint_signal(raw_text):
            continue

        canonical_entities = dedupe_preserve_order(
            normalize_whitespace(x)
            for x in group_df["canonical_name"].tolist()
            if normalize_whitespace(x) and looks_like_company_entity(x, "USPTO")
        )

        if len(canonical_entities) < 2:
            continue

        for entity_a, entity_b in itertools.combinations(canonical_entities, 2):
            if entity_a == entity_b:
                continue
            candidate_rows.append(
                {
                    "source_id": source_id,
                    "source_type": "USPTO",
                    "source_url": source_url,
                    "date": date,
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "candidate_relation": "FILED_PATENT_WITH",
                    "evidence_text": raw_text[:500],
                    "trigger_phrase": "explicit_joint_patent_signal",
                    "extraction_method": "source_aware_record_trigger_v2",
                    "num_entities_in_sentence": len(canonical_entities),
                }
            )

    return candidate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract candidate company-company relationships.")
    parser.add_argument("--input", required=True, help="Path to resolved_entities.csv from resolve_alias.py")
    parser.add_argument("--output", required=True, help="Path to write candidate_relations.csv")
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

    if not candidate_df.empty:
        candidate_df = candidate_df.drop_duplicates(
            subset=[
                "source_id", "source_type", "source_url", "date",
                "entity_a", "entity_b", "candidate_relation",
                "evidence_text", "trigger_phrase", "extraction_method",
            ]
        ).reset_index(drop=True)

    if candidate_df.empty:
        candidate_df = pd.DataFrame(
            columns=[
                "source_id", "source_type", "source_url", "date",
                "entity_a", "entity_b", "candidate_relation",
                "evidence_text", "trigger_phrase",
                "extraction_method", "num_entities_in_sentence",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(output_path, index=False)

    print(f"[extract_candidate_relations_stage1_v2] Wrote {len(candidate_df)} candidate row(s) to: {output_path}")


if __name__ == "__main__":
    main()
