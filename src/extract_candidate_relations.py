#!/usr/bin/env python3
"""Source-aware candidate relation extraction from resolved entity rows.

Pipeline position:  resolve_alias.py → extract_candidate_relations.py

Relation extraction strategy per source
----------------------------------------
SEC    : sentence-level trigger (primary). Paragraph-scoped fallback only —
         pairs entities within the same paragraph as the trigger, not the whole
         document. Paragraph window is capped at MAX_PARAGRAPH_CHARS.
PubMed : sentence-level trigger only. Tighter patterns than SEC — "worked with"
         removed, "co-funded by" requires exactly two ORG spans in the sentence.
USPTO  : record-level trigger ONLY when there is an explicit co-assignee /
         joint-patent signal. Combinatorial pairs capped at MAX_PAIRS_PER_RECORD.

Key changes from v1
--------------------
- SEC document-level fallback replaced with paragraph-scoped fallback.
  v1 paired every entity in a document whenever any trigger appeared anywhere
  in that document. A single "acquisition" in a risk-factor section caused
  every company mentioned in the filing to be paired. Now the fallback only
  fires within the paragraph containing the trigger, and only if that paragraph
  is under MAX_PARAGRAPH_CHARS (longer paragraphs are likely boilerplate dumps).

- _emit_pairs now accepts trigger_offset + entity_offsets and skips pairs where
  neither entity is within MAX_ENTITY_DISTANCE chars of the trigger. Prevents
  the paragraph fallback from pairing distant co-located names.

- "worked with" removed from COLLABORATED_WITH PubMed pattern (too broad for
  academic methods sections).

- FUNDED_WITH PubMed now requires exactly 2 ORG entities in the sentence. The
  co-funding relation is binary by definition; more entities means the trigger
  fired in a list context and the pair is underspecified.

- _looks_like_company adds a solo-all-caps guard: a single token that is
  entirely uppercase and not in the pharma abbreviation allowlist is rejected.
  This catches drug names, form numbers, and metadata header tokens that slip
  through the blocklist.

- Sentence-level pairs are tracked to avoid re-emitting them in the paragraph
  fallback pass, keeping duplicate counts clean.

- MAX_PAIRS_PER_RECORD hard cap prevents combinatorial explosion for records
  with many entity mentions.
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "source_id", "source_type", "source_url", "date",
    "raw_text", "canonical_name", "start_char", "end_char",
})

# ── Tuneable limits ────────────────────────────────────────────────────────────

# Max chars between an entity's offset and the trigger to count as "proximate".
# Only applied in the paragraph fallback pass, not sentence-level.
MAX_ENTITY_DISTANCE: int = 400

# Paragraph fallback: skip paragraphs longer than this (likely full-doc dumps).
MAX_PARAGRAPH_CHARS: int = 1_500

# Hard cap on pairs emitted per source_id / record.
MAX_PAIRS_PER_RECORD: int = 20

# ─── Relation trigger patterns ────────────────────────────────────────────────

_SEC_PATTERNS: list[Tuple[str, str]] = [
    ("ACQUIRED",          r"\b(acquired|acquires|buy|bought|purchased|acquisition of|merged with|"
                          r"merger with|definitive agreement to acquire|agreement to purchase|"
                          r"tender offer)\b"),
    ("LICENSED_FROM",     r"\b(licensed from|licensed to|exclusive license|license granted|"
                          r"granted\s+\w*\s*license|licensing agreement|license agreement|"
                          r"license deal|entered into a license|sublicense|royalt(?:y|ies))\b"),
    ("PARTNERED_WITH",    r"\b(partnership|partnered with|strategic partnership|"
                          r"collaboration agreement)\b"),
    ("COLLABORATED_WITH", r"\b(collaborated with|collaboration with|co-develop(?:ed)? with|"
                          r"jointly developed)\b"),
]

_PUBMED_PATTERNS: list[Tuple[str, str]] = [
    ("PARTNERED_WITH",    r"\b(partnership|partnered with|in partnership with)\b"),
    # "worked with" removed — fires on almost every academic methods section.
    ("COLLABORATED_WITH", r"\b(collaborated with|collaboration with|in collaboration with|"
                          r"co-develop(?:ed)? with|jointly developed)\b"),
    # Require the explicit co-funding form. "funded by" alone is too common.
    # Two-entity requirement enforced downstream in build_candidate_rows.
    ("FUNDED_WITH",       r"\b(co-funded by|jointly funded by)\b"),
]

_USPTO_PATTERNS: list[Tuple[str, str]] = [
    ("FILED_PATENT_WITH", r"\b(co-assignee|co-assigned|co assigned|co-filed|"
                          r"joint patent|filed patent with|patent with)\b"),
]

_ALL_PATTERNS = _SEC_PATTERNS + _PUBMED_PATTERNS + _USPTO_PATTERNS

_SOURCE_PATTERNS: Dict[str, list[Tuple[str, str]]] = {
    "SEC":    _SEC_PATTERNS,
    "PUBMED": _PUBMED_PATTERNS,
    "USPTO":  _USPTO_PATTERNS,
}

# ─── Filtering constants ───────────────────────────────────────────────────────

_NEGATIVE_TRIGGERS: list[str] = [
    r"\bcommunity-acquired\b",
    r"\bhospital-acquired\b",
    r"\bventilator-associated\b",
    r"\bdevice-associated\b",
    r"\bhealthcare-associated\b",
]

_SEC_BOILERPLATE: list[str] = [
    r"\bcurrent report on form 8-k\b",
    r"\bforward-looking statements\b",
    r"\bprivate securities litigation reform act\b",
    r"\bcommission file\b",
    r"\bsafe harbor\b",
    r"\bunaudited condensed consolidated\b",
    r"\bnotes to (consolidated )?financial statements\b",
]

_ENTITY_BLOCKLIST_EXACT: Set[str] = {
    "COMPANY", "REGISTRANT", "COMMON STOCK", "COMMISSION FILE", "DISTRICT COURT",
    "THE SECURITIES AND EXCHANGE COMMISSION", "SEC", "FDA", "NIH", "EMA", "WHO",
    "THE LICENSE AGREEMENT", "LICENSE AGREEMENT",
    "THE MERGER AGREEMENT", "MERGER AGREEMENT", "AGREEMENT",
    "TERRITORY", "THE LICENSED PRODUCT", "LICENSED PRODUCT",
    "PURCHASER", "SHARES", "CVR", "CI", "CAPITA", "DSO", "NJ",
    "CPC", "IPC", "USPC", "INVENTOR", "INVENTORS", "ASSIGNEE",
    "PATENT", "GRANT", "APPLICATION",
    "INC", "INC.", "CORP", "CORP.", "LTD", "LLC", "CO", "CO.",
    "SHARP", "DOHME", "SHARP & DOHME",
    "OTHERS", "AND OTHERS",
    "BOARD OF DIRECTORS", "AUDIT COMMITTEE", "EXECUTIVE COMMITTEE",
    "THE COMPANY", "THE REGISTRANT", "THE ISSUER",
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
    "BOARD OF DIRECTOR", "AUDIT COMMITTEE",
]

_PRODUCTISH_PATTERNS: list[str] = [
    r"\bBAQSIMI\b", r"\bPRIMATENE\b", r"\bMIST\b",
    r"\bGLUCAGON\b", r"\bPCV\d+\b",
    # Generic drug-name suffix patterns (mAbs, small-molecule inhibitors, etc.)
    r"\b[A-Z]{4,10}(MAB|NIB|TINIB|ZUMAB|LUMAB|CICEPT)\b",
]

_USPTO_METADATA_PATTERNS: list[str] = [
    r"^\s*(CPC|IPC|USPC|PATENT|ASSIGNEE|INVENTOR[S]?|PUBLICATION|APPLICATION)\s*$",
]

_KNOWN_PHARMA_ABBREVS: Set[str] = {
    "BMS", "GSK", "JNJ", "AZ", "MSD", "MRK", "LLY", "PFE", "ABBV", "AMGN",
    "RHHBY", "NVS", "SNY", "AZN", "BMY",
}

_EXACT_ALLOWLIST: Set[str] = {
    "National Institutes of Health",
    "University of Pennsylvania",
    "National Cancer Institute",
}

_NCT_RE             = re.compile(r"^NCT\d{6,}$", re.IGNORECASE)
_TRIAL_CODENAME_RE  = re.compile(r"^[A-Z][A-Z0-9\-]{2,15}$")
_SENTENCE_SPLIT_RE  = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}|\r\n{2,}")
_WS_RE              = re.compile(r"\s+")
# Matches a single token that is entirely uppercase (no lowercase letters).
_SOLO_CAPS_RE       = re.compile(r"^[A-Z0-9&\-\.]{2,20}$")


# ─── Text helpers ─────────────────────────────────────────────────────────────

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


def _paragraph_spans(text: str) -> List[Tuple[int, int, str]]:
    """Split on blank lines, returning (start, end, paragraph_text) tuples."""
    text = _norm(text)
    if not text:
        return []
    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for part in _PARAGRAPH_SPLIT_RE.split(text):
        part = part.strip()
        if not part:
            continue
        start = text.find(part, cursor)
        end = start + len(part)
        spans.append((start, end, part))
        cursor = end
    return spans or [(0, len(text), text)]


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

    tokens = entity.split()
    src = source_type.upper()

    # Solo all-caps token: reject unless it's a known pharma abbreviation.
    # Catches drug names, header tokens (PATENT, IPC, etc.) not in the blocklist.
    if len(tokens) == 1 and _SOLO_CAPS_RE.match(upper):
        if upper not in _KNOWN_PHARMA_ABBREVS:
            return False

    if src == "USPTO" and re.fullmatch(r"[A-Z0-9\-]{2,8}", upper):
        if upper not in _KNOWN_PHARMA_ABBREVS:
            return False
    if src == "PUBMED" and _TRIAL_CODENAME_RE.match(entity.strip()):
        if upper not in _KNOWN_PHARMA_ABBREVS:
            return False

    # Reject all-lowercase multi-word strings (sentence fragments).
    if entity.lower() == entity and " " in entity:
        return False

    return True


# ─── Relation detection ───────────────────────────────────────────────────────

def _detect_relation(text: str, source_type: str) -> Tuple[str, str, int]:
    """Return (relation_label, trigger_phrase, trigger_char_offset).

    Returns ("", "", -1) when no trigger fires or a negative trigger cancels it.
    """
    lower = text.lower()
    if any(re.search(p, lower) for p in _NEGATIVE_TRIGGERS):
        return "", "", -1
    patterns = _SOURCE_PATTERNS.get(source_type.upper(), _ALL_PATTERNS)
    for label, regex in patterns:
        m = re.search(regex, lower, flags=re.IGNORECASE)
        if m:
            return label, m.group(0), m.start()
    return "", "", -1


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


# ─── Pair emission ────────────────────────────────────────────────────────────

def _emit_pairs(
    entities: List[str],
    base: dict,
    relation: str,
    trigger: str,
    method: str,
    trigger_offset: int = -1,
    entity_offsets: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """Emit (entity_a, entity_b) pairs for a detected relation.

    Proximity guard (paragraph fallback only):
        When trigger_offset >= 0 and entity_offsets is supplied, a pair is only
        emitted if at least one of the two entities sits within MAX_ENTITY_DISTANCE
        chars of the trigger. This prevents distant co-located names from being
        paired just because they share a paragraph with a trigger phrase.

    Pair cap:
        Stops after MAX_PAIRS_PER_RECORD rows to prevent combinatorial explosion
        in documents with many entity mentions.
    """
    rows: List[dict] = []
    use_proximity = trigger_offset >= 0 and entity_offsets is not None

    for a, b in itertools.combinations(entities, 2):
        if a == b:
            continue
        if use_proximity:
            off_a = entity_offsets.get(a, -1)  # type: ignore[union-attr]
            off_b = entity_offsets.get(b, -1)  # type: ignore[union-attr]
            close_a = off_a >= 0 and abs(off_a - trigger_offset) <= MAX_ENTITY_DISTANCE
            close_b = off_b >= 0 and abs(off_b - trigger_offset) <= MAX_ENTITY_DISTANCE
            if not (close_a or close_b):
                continue
        rows.append({
            **base,
            "entity_a": a,
            "entity_b": b,
            "candidate_relation": relation,
            "trigger_phrase": trigger,
            "extraction_method": method,
            "num_entities_in_sentence": len(entities),
        })
        if len(rows) >= MAX_PAIRS_PER_RECORD:
            break
    return rows


# ─── Main extraction logic ────────────────────────────────────────────────────

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

        relation, trigger, _ = _detect_relation(sent, stype)
        if not relation:
            continue

        # PubMed FUNDED_WITH is a binary relation by definition.
        # More than 2 entities means the trigger fired in a list context.
        if stype == "PUBMED" and relation == "FUNDED_WITH" and len(entities) != 2:
            continue

        base = {"source_id": sid, "source_type": stype, "source_url": surl,
                "date": date, "evidence_text": sent}
        rows.extend(_emit_pairs(entities, base, relation, trigger,
                                "source_aware_sentence_trigger"))

    # Track what the sentence pass already found to avoid double-counting in
    # the paragraph fallback. Store both directions so (A,B) and (B,A) match.
    sentence_level_seen: Set[Tuple] = set()
    for r in rows:
        sid = r["source_id"]
        rel = r["candidate_relation"]
        sentence_level_seen.add((sid, r["entity_a"], r["entity_b"], rel))
        sentence_level_seen.add((sid, r["entity_b"], r["entity_a"], rel))

    # ── Paragraph-scoped fallback for SEC ─────────────────────────────────────
    # Replaces the v1 document-level fallback that paired all entities in a
    # document whenever any trigger appeared anywhere in it.
    #
    # Strategy:
    #   1. Split the document into paragraphs on blank lines.
    #   2. Skip paragraphs over MAX_PARAGRAPH_CHARS (likely boilerplate dumps).
    #   3. Run trigger detection on each paragraph.
    #   4. Restrict entity candidates to those whose start_char falls inside
    #      the paragraph's character range.
    #   5. Apply the proximity guard so entities far from the trigger are skipped.
    #   6. Skip pairs already captured by the sentence-level pass.
    for (sid, stype, surl, date, raw), grp in df.groupby(
        ["source_id", "source_type", "source_url", "date", "raw_text"], dropna=False
    ):
        if str(stype).upper() != "SEC":
            continue
        raw = _norm(raw)
        if not raw:
            continue

        # Build canonical_name → start_char map for this document.
        entity_offsets: Dict[str, int] = {}
        for _, erow in grp.iterrows():
            name = _norm(erow.get("canonical_name", ""))
            if name and name not in entity_offsets:
                try:
                    entity_offsets[name] = int(erow.get("start_char", -1))
                except (ValueError, TypeError):
                    entity_offsets[name] = -1

        for p_start, p_end, para in _paragraph_spans(raw):
            if len(para) > MAX_PARAGRAPH_CHARS:
                continue
            if _is_sec_boilerplate(para):
                continue

            relation, trigger, trigger_rel_offset = _detect_relation(para, "SEC")
            if not relation:
                continue

            trigger_abs_offset = p_start + trigger_rel_offset

            # Only entities whose char offset sits within this paragraph.
            para_entities = _dedupe_ordered(
                name for name, off in entity_offsets.items()
                if p_start <= off < p_end and _looks_like_company(name, "SEC")
            )
            if len(para_entities) < 2:
                continue

            base = {"source_id": sid, "source_type": "SEC", "source_url": surl,
                    "date": date, "evidence_text": para[:500]}

            new_pairs = _emit_pairs(
                para_entities, base, relation, trigger,
                "sec_paragraph_fallback",
                trigger_offset=trigger_abs_offset,
                entity_offsets=entity_offsets,
            )

            for pair in new_pairs:
                key = (sid, pair["entity_a"], pair["entity_b"], pair["candidate_relation"])
                if key not in sentence_level_seen:
                    rows.append(pair)
                    sentence_level_seen.add(key)
                    sentence_level_seen.add(
                        (sid, pair["entity_b"], pair["entity_a"], pair["candidate_relation"])
                    )

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
        rows.extend(_emit_pairs(
            entities, base,
            "FILED_PATENT_WITH",
            "explicit_joint_patent_signal",
            "source_aware_record_trigger",
        ))

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