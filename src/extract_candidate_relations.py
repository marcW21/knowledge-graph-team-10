#!/usr/bin/env python3
"""Source-aware candidate relation extraction from resolved entity rows.

Pipeline position:  resolve_alias.py → extract_candidate_relations.py

Relation extraction strategy per source
----------------------------------------
SEC    : sentence-level trigger + document-level fallback (acquirer/target often
         in separate sentences).
PubMed : sentence-level trigger only.
USPTO  : record-level trigger ONLY when there is explicit co-assignee/joint-patent
         signal (avoids combinatorial explosion from single-assignee patent metadata).
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "source_id", "source_type", "source_url", "date",
    "raw_text", "canonical_name", "start_char", "end_char",
})

# ─── Relation trigger patterns ────────────────────────────────────────────────

_SEC_PATTERNS: list[Tuple[str, str]] = [
    ("ACQUIRED",        r"\b(acquired|acquires|buy|bought|purchased|acquisition of|merged with|"
                        r"merger with|definitive agreement to acquire|agreement to purchase|"
                        r"tender offer)\b"),
    ("LICENSED_FROM",   r"\b(licensed from|licensed to|exclusive license|license granted|"
                        r"granted\s+\w*\s*license|licensing agreement|license agreement|"
                        r"license deal|entered into a license|sublicense|royalt(?:y|ies))\b"),
    ("PARTNERED_WITH",  r"\b(partnership|partnered with|strategic partnership|"
                        r"collaboration agreement)\b"),
    ("COLLABORATED_WITH", r"\b(collaborated with|collaboration with|co-develop(?:ed)? with|"
                          r"jointly developed)\b"),
]

_PUBMED_PATTERNS: list[Tuple[str, str]] = [
    ("PARTNERED_WITH",    r"\b(partnership|partnered with|in partnership with)\b"),
    ("COLLABORATED_WITH", r"\b(collaborated with|collaboration with|in collaboration with|"
                          r"co-develop(?:ed)? with|jointly developed|worked with)\b"),
    # Only match co-funding when two named entities appear in the same clause.
    ("FUNDED_WITH",       r"\b(co-funded by|funded by \S[\w\s&,\.]{1,60} and "
                          r"\S[\w\s&,\.]{1,40}(?=\s*[;,\.]|\s*$))\b"),
]

_USPTO_PATTERNS: list[Tuple[str, str]] = [
    ("FILED_PATENT_WITH", r"\b(co-assignee|co-assigned|co assigned|co-filed|"
                          r"joint patent|filed patent with|patent with)\b"),
]

# Merged fallback for unknown source types.
_ALL_PATTERNS = _SEC_PATTERNS + _PUBMED_PATTERNS + _USPTO_PATTERNS

_SOURCE_PATTERNS: Dict[str, list[Tuple[str, str]]] = {
    "SEC": _SEC_PATTERNS,
    "PUBMED": _PUBMED_PATTERNS,
    "USPTO": _USPTO_PATTERNS,
}

# ─── Filtering ────────────────────────────────────────────────────────────────

_NEGATIVE_TRIGGERS = [
    r"\bcommunity-acquired\b",
    r"\bhospital-acquired\b",
    r"\bventilator-associated\b",
]

_SEC_BOILERPLATE = [
    r"\bcurrent report on form 8-k\b",
    r"\bforward-looking statements\b",
    r"\bprivate securities litigation reform act\b",
    r"\bcommission file\b",
]

_ENTITY_BLOCKLIST_EXACT: Set[str] = {
    "COMPANY", "REGISTRANT", "COMMON STOCK", "COMMISSION FILE", "DISTRICT COURT",
    "THE SECURITIES AND EXCHANGE COMMISSION", "SEC", "FDA", "NIH",
    "THE LICENSE AGREEMENT", "LICENSE AGREEMENT",
    "THE MERGER AGREEMENT", "MERGER AGREEMENT", "AGREEMENT",
    "TERRITORY", "THE LICENSED PRODUCT", "LICENSED PRODUCT",
    "PURCHASER", "SHARES", "CVR", "CI", "CAPITA", "DSO", "NJ",
    "CPC", "IPC", "USPC", "INVENTOR", "INVENTORS", "ASSIGNEE",
    "PATENT", "GRANT", "APPLICATION",
    "INC", "INC.", "CORP", "CORP.", "LTD", "LLC", "CO", "CO.",
    "SHARP", "DOHME", "SHARP & DOHME",
    "OTHERS", "AND OTHERS",
}

_ENTITY_BLOCKLIST_SUBSTRINGS: list[str] = [
    "&#",
    "FORM 8-K", "FORM 10-K", "FORM 10-Q", "COMMON STOCK",
    "COMMISSION FILE", "DISTRICT COURT", "SECURITIES EXCHANGE ACT",
    "LICENSE AGREEMENT", "MERGER AGREEMENT", "LICENSED PRODUCT", "TERRITORY",
    "CURRENT REPORT", "GRANT YEAR", "PUBLICATION", "APPLICATION NO", "PATENT NO",
    "INVENTOR", "INVENTORS", "CPC", "IPC", "USPC", "ASSIGNEE:",
    "FUNDED BY", "ABBREVIATED NEW DRUG",
    "COLLABORATION AGREEMENT", "AND COLLABORATION",
]

_PRODUCTISH_PATTERNS: list[str] = [
    r"\bBAQSIMI\b", r"\bPRIMATENE\b", r"\bMIST\b",
    r"\bGLUCAGON\b", r"\bPCV\d+\b",
]

_USPTO_METADATA_PATTERNS: list[str] = [
    r"^\s*(CPC|IPC|USPC|PATENT|ASSIGNEE|INVENTOR[S]?|PUBLICATION|APPLICATION)\s*$",
]

# Moved out of hot loop — was re-created on every call in the original.
_KNOWN_PHARMA_ABBREVS: Set[str] = {
    "BMS", "GSK", "JNJ", "AZ", "MSD", "MRK", "LLY", "PFE", "ABBV", "AMGN",
}

_EXACT_ALLOWLIST: Set[str] = {
    "National Institutes of Health",
    "University of Pennsylvania",
}

_NCT_RE = re.compile(r"^NCT\d{6,}$", re.IGNORECASE)
_TRIAL_CODENAME_RE = re.compile(r"^[A-Z][A-Z0-9\-]{2,15}$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _norm(value: object) -> str:
    if pd.isna(value):
        return ""
    return _WS_RE.sub(" ", str(value).replace("\u00a0", " ")).strip()


def _dedupe_ordered(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    text = _norm(text)
    if not text:
        return []
    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for part in _SENTENCE_SPLIT_RE.split(text):
        part = part.strip()
        if not part:
            continue
        start = text.find(part, cursor)
        end = start + len(part)
        spans.append((start, end, part))
        cursor = end
    return spans or [(0, len(text), text)]


def _assign_sentence(raw_text: str, start_char: object) -> str:
    text = _norm(raw_text)
    if not text:
        return ""
    try:
        ms = int(start_char)
    except Exception:
        return text
    for s, e, sent in _sentence_spans(text):
        if s <= ms < e:
            return sent
    return text


# ─── Entity filtering ─────────────────────────────────────────────────────────

def _looks_like_company(text: str, source_type: str) -> bool:
    entity = _norm(text)
    if not entity:
        return False

    upper = entity.upper()

    if entity in _EXACT_ALLOWLIST or upper in {x.upper() for x in _EXACT_ALLOWLIST}:
        return True
    if upper in _ENTITY_BLOCKLIST_EXACT:
        return False
    if any(bad in upper for bad in _ENTITY_BLOCKLIST_SUBSTRINGS):
        return False
    if any(re.search(p, upper) for p in _PRODUCTISH_PATTERNS):
        return False
    if any(re.search(p, upper) for p in _USPTO_METADATA_PATTERNS):
        return False
    if not re.search(r"[A-Za-z]", entity):
        return False
    if _NCT_RE.match(entity.strip()):
        return False
    if len(re.sub(r"[^A-Za-z0-9&]", "", entity)) <= 2:
        return False

    src = source_type.upper()
    if src == "USPTO" and re.fullmatch(r"[A-Z0-9\-]{2,8}", upper):
        if upper not in _KNOWN_PHARMA_ABBREVS:
            return False
    if src == "PUBMED" and _TRIAL_CODENAME_RE.match(entity.strip()):
        if upper not in _KNOWN_PHARMA_ABBREVS:
            return False

    # Reject all-lowercase multi-word strings (usually sentence fragments).
    if entity.lower() == entity and " " in entity:
        return False

    return True


# ─── Relation detection ───────────────────────────────────────────────────────

def _detect_relation(sentence: str, source_type: str) -> Tuple[str, str]:
    lower = sentence.lower()
    if any(re.search(p, lower) for p in _NEGATIVE_TRIGGERS):
        return "", ""
    patterns = _SOURCE_PATTERNS.get(source_type.upper(), _ALL_PATTERNS)
    for label, regex in patterns:
        m = re.search(regex, lower, flags=re.IGNORECASE)
        if m:
            return label, m.group(0)
    return "", ""


def _is_sec_boilerplate(sentence: str) -> bool:
    lower = sentence.lower()
    return any(re.search(p, lower) for p in _SEC_BOILERPLATE)


def _has_joint_patent_signal(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in [
        r"\bco-assignee\b", r"\bco-assigned\b", r"\bco assigned\b",
        r"\bjoint patent\b", r"\bpatent with\b", r"\bfiled patent with\b",
        r"\bco-filed\b", r"\band\b.*\bassignee\b", r"\bassignees?\b.*\band\b",
    ])


# ─── Candidate row builders ───────────────────────────────────────────────────

def _emit_pairs(
    entities: List[str],
    base: dict,
    relation: str,
    trigger: str,
    method: str,
) -> List[dict]:
    rows: List[dict] = []
    for a, b in itertools.combinations(entities, 2):
        if a == b:
            continue
        rows.append({**base, "entity_a": a, "entity_b": b,
                     "candidate_relation": relation, "trigger_phrase": trigger,
                     "extraction_method": method,
                     "num_entities_in_sentence": len(entities)})
    return rows


def build_candidate_rows(df: pd.DataFrame) -> List[dict]:
    rows: List[dict] = []

    # ── Sentence-level pass (SEC + PubMed) ────────────────────────────────────
    sent_groups = df.groupby(
        ["source_id", "source_type", "source_url", "date", "sentence_text"],
        dropna=False,
    )
    for (sid, stype, surl, date, sent), grp in sent_groups:
        stype = str(stype).upper()
        sent = _norm(sent)
        if not sent:
            continue
        if stype == "SEC" and _is_sec_boilerplate(sent):
            continue

        entities = _dedupe_ordered(
            _norm(x) for x in grp["canonical_name"]
            if _norm(x) and _looks_like_company(x, stype)
        )
        if len(entities) < 2:
            continue

        relation, trigger = _detect_relation(sent, stype)
        if not relation:
            continue

        base = {"source_id": sid, "source_type": stype, "source_url": surl,
                "date": date, "evidence_text": sent}
        rows.extend(_emit_pairs(entities, base, relation, trigger,
                                "source_aware_sentence_trigger"))

    # ── Document-level fallback for SEC ───────────────────────────────────────
    for (sid, stype, surl, date, raw), grp in df.groupby(
        ["source_id", "source_type", "source_url", "date", "raw_text"], dropna=False
    ):
        if str(stype).upper() != "SEC":
            continue
        raw = _norm(raw)
        if not raw:
            continue
        relation, trigger = _detect_relation(raw, "SEC")
        if not relation:
            continue

        entities = _dedupe_ordered(
            _norm(x) for x in grp["canonical_name"]
            if _norm(x) and _looks_like_company(x, "SEC")
        )
        if len(entities) < 2:
            continue

        base = {"source_id": sid, "source_type": "SEC", "source_url": surl,
                "date": date, "evidence_text": raw[:500]}
        rows.extend(_emit_pairs(entities, base, relation, trigger,
                                "sec_document_level_trigger"))

    # ── Record-level joint-signal pass for USPTO ──────────────────────────────
    for (sid, stype, surl, date, raw), grp in df.groupby(
        ["source_id", "source_type", "source_url", "date", "raw_text"], dropna=False
    ):
        if str(stype).upper() != "USPTO":
            continue
        raw = _norm(raw)
        if not raw or not _has_joint_patent_signal(raw):
            continue

        entities = _dedupe_ordered(
            _norm(x) for x in grp["canonical_name"]
            if _norm(x) and _looks_like_company(x, "USPTO")
        )
        if len(entities) < 2:
            continue

        base = {"source_id": sid, "source_type": "USPTO", "source_url": surl,
                "date": date, "evidence_text": raw[:500]}
        rows.extend(_emit_pairs(entities, base,
                                "FILED_PATENT_WITH",
                                "explicit_joint_patent_signal",
                                "source_aware_record_trigger"))

    return rows


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract candidate company-company relationships.")
    p.add_argument("--input", required=True, help="resolved_entities CSV from resolve_alias.py")
    p.add_argument("--output", required=True, help="candidate_relations CSV")
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

    # Attach sentence text to each row.
    df["sentence_text"] = df.apply(
        lambda r: _assign_sentence(r.get("raw_text", ""), r.get("start_char", -1)),
        axis=1,
    )

    candidate_rows = build_candidate_rows(df)

    output_cols = [
        "source_id", "source_type", "source_url", "date",
        "entity_a", "entity_b", "candidate_relation",
        "evidence_text", "trigger_phrase",
        "extraction_method", "num_entities_in_sentence",
    ]

    if candidate_rows:
        out = (
            pd.DataFrame(candidate_rows)
              .drop_duplicates(subset=[
                  "source_id", "source_type", "source_url", "date",
                  "entity_a", "entity_b", "candidate_relation",
                  "evidence_text", "trigger_phrase", "extraction_method",
              ])
              .reset_index(drop=True)
        )
    else:
        out = pd.DataFrame(columns=output_cols)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out[output_cols].to_csv(args.output, index=False)
    print(f"[extract_candidate_relations] Wrote {len(out)} candidate row(s) to: {args.output}")


if __name__ == "__main__":
    main()
