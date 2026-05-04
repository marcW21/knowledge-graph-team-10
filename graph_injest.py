#!/usr/bin/env python3
"""
graph_ingest_v2.py

Drop-in upgrade of graph_ingest.py.

Changes from v1:
  - Adds COLLABORATED_WITH and FUNDED_WITH upsert methods (present in CSV but
    not handled in v1 — fell through to generic upsert which requires APOC).
  - Adds entity sanitization: strips NER artifacts like "Funded by X", bare
    trigger phrases, and clinical trial numbers (NCTxxxxxxxx) so they don't
    create junk Company nodes.
  - ClinicalTrial numbers are upserted as :ClinicalTrial nodes and linked via
    FUNDED_TRIAL edges instead of FUNDED_WITH Company→Company edges.
  - audit_flag=True rows are tagged on the relationship for SME review.
  - decision_bucket column is respected: 'auto_reject' rows are hard-skipped
    even if final_confidence >= MIN_CONFIDENCE (catches edge cases in the CSV).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

# ==========================
# CONFIG
# ==========================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

MIN_CONFIDENCE = 0.50

# ==========================
# LOGGING
# ==========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ==========================
# ENTITY SANITIZATION
# ==========================

# Regex for bare ClinicalTrials.gov IDs
_NCT_RE = re.compile(r"^NCT\d{7,8}$", re.IGNORECASE)

# Prefixes that indicate the NER grabbed a phrase fragment instead of an org
_JUNK_PREFIXES = (
    "funded by",
    "equitable access",
    "partnership by",
    "sponsored by",
    "supported by",
    "grant from",
)


def _is_clinical_trial(name: str) -> bool:
    return bool(_NCT_RE.match(name.strip()))


def _is_junk_entity(name: str) -> bool:
    """Return True if the entity name is clearly an NER extraction error."""
    lower = name.strip().lower()
    if lower in ("", "none", "n/a", "nan"):
        return True
    for prefix in _JUNK_PREFIXES:
        if lower.startswith(prefix):
            return True
    return False


def sanitize_entity(name: str) -> str:
    """
    Strip common NER prefix artifacts.
    e.g. "Funded by Eisai" -> "Eisai"
    Returns the cleaned name, or empty string if unsalvageable.
    """
    s = name.strip()
    for prefix in _JUNK_PREFIXES:
        # case-insensitive prefix strip
        if s.lower().startswith(prefix):
            s = s[len(prefix):].lstrip(" ,;").strip()
    return s


# ==========================
# UTILS
# ==========================

def generate_rel_uid(*args) -> str:
    raw = "|".join(str(a) for a in args)
    return hashlib.sha256(raw.encode()).hexdigest()


def _row_is_valid(row: pd.Series) -> bool:
    # Hard-skip anything the pipeline already rejected
    if str(row.get("decision_bucket", "")).strip().lower() == "auto_reject":
        return False
    if row["final_confidence"] < MIN_CONFIDENCE:
        return False
    gpt_ok = row.get("gpt4o_valid") is True
    claude_ok = row.get("claude_valid") is True
    return gpt_ok or claude_ok


def _rel_data_from_row(row: pd.Series, uid: str = "") -> dict:
    return {
        "uid": uid,
        "date": row.get("date", ""),
        "confidence": float(row["final_confidence"]),
        "source_type": row.get("source_type", "UNKNOWN"),
        "source_url": row.get("source_url", ""),
        "source_date": row.get("date", ""),
        "extracted_by": "NER_pipeline",
        "validated_by": _validators(row),
        "model_agreement_ratio": float(row.get("model_agreement_ratio", 0)),
        "gpt4o_reasoning": str(row.get("gpt4o_reasoning", "")),
        "claude_reasoning": str(row.get("claude_reasoning", "")),
        "audit_flag": bool(row.get("audit_flag", False)),
        "decision_bucket": str(row.get("decision_bucket", "")),
        "evidence_text": str(row.get("evidence_text", "")),
    }


def _validators(row: pd.Series) -> str:
    v = []
    if row.get("gpt4o_valid") is True:
        v.append("GPT-4o")
    if row.get("claude_valid") is True:
        v.append("Claude")
    return "+".join(v) if v else "none"


# ==========================
# GRAPH MANAGER
# ==========================

class KnowledgeGraph:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # -----------------------------------
    # SCHEMA
    # -----------------------------------

    def create_constraints(self):
        constraints = [
            "CREATE CONSTRAINT company_name_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT technology_name_unique IF NOT EXISTS FOR (t:Technology) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT jurisdiction_name_unique IF NOT EXISTS FOR (j:Jurisdiction) REQUIRE j.name IS UNIQUE",
            "CREATE CONSTRAINT personnel_uid_unique IF NOT EXISTS FOR (p:Personnel) REQUIRE p.uid IS UNIQUE",
            # New in v2
            "CREATE CONSTRAINT clinical_trial_id_unique IF NOT EXISTS FOR (ct:ClinicalTrial) REQUIRE ct.trial_id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for c in constraints:
                session.run(c)
        log.info("Constraints ensured.")

    # -----------------------------------
    # NODE UPSERTS
    # -----------------------------------

    def upsert_company(self, name: str):
        query = """
        MERGE (c:Company {name: $name})
        ON CREATE SET c.created_at = datetime()
        ON MATCH SET  c.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(query, name=name)

    def upsert_clinical_trial(self, trial_id: str):
        """ClinicalTrials.gov numbers get their own node type, not Company."""
        query = """
        MERGE (ct:ClinicalTrial {trial_id: $trial_id})
        ON CREATE SET ct.created_at = datetime()
        ON MATCH SET  ct.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(query, trial_id=trial_id.upper())

    # -----------------------------------
    # SHARED RELATIONSHIP PROPERTY SETTER
    # (extracted to avoid repetition across upsert methods)
    # -----------------------------------

    @staticmethod
    def _on_create_props() -> str:
        """Cypher fragment: SET all rel properties on CREATE."""
        return """
            r.date                 = $date,
            r.confidence           = $confidence,
            r.source_type          = $source_type,
            r.source_url           = $source_url,
            r.source_date          = $source_date,
            r.extracted_by         = $extracted_by,
            r.validated_by         = $validated_by,
            r.model_agreement      = $model_agreement_ratio,
            r.gpt4o_reasoning      = $gpt4o_reasoning,
            r.claude_reasoning     = $claude_reasoning,
            r.audit_flag           = $audit_flag,
            r.decision_bucket      = $decision_bucket,
            r.evidence_text        = $evidence_text,
            r.created_at           = datetime()
        """

    @staticmethod
    def _on_match_props() -> str:
        """Cypher fragment: update confidence and timestamp on MATCH."""
        return """
            r.confidence =
                CASE WHEN $confidence > r.confidence
                     THEN $confidence ELSE r.confidence END,
            r.updated_at = datetime()
        """

    # -----------------------------------
    # RELATIONSHIP UPSERTS
    # -----------------------------------

    def upsert_acquired(self, acquirer: str, target: str, rel_data: dict):
        uid = generate_rel_uid(acquirer, target, "ACQUIRED", rel_data["source_url"])
        rel_data["uid"] = uid
        query = f"""
        MATCH (a:Company {{name: $acquirer}})
        MATCH (t:Company {{name: $target}})
        MERGE (a)-[r:ACQUIRED {{uid: $uid}}]->(t)
        ON CREATE SET {self._on_create_props()}
        ON MATCH SET  {self._on_match_props()}
        """
        with self.driver.session() as session:
            session.run(query, acquirer=acquirer, target=target, **rel_data)

    def upsert_licensed_to(self, licensor: str, licensee: str, rel_data: dict):
        uid = generate_rel_uid(licensor, licensee, "LICENSED_TO", rel_data["source_url"])
        rel_data["uid"] = uid
        query = f"""
        MATCH (l:Company {{name: $licensor}})
        MATCH (e:Company {{name: $licensee}})
        MERGE (l)-[r:LICENSED_TO {{uid: $uid}}]->(e)
        ON CREATE SET {self._on_create_props()}
        ON MATCH SET  {self._on_match_props()}
        """
        with self.driver.session() as session:
            session.run(query, licensor=licensor, licensee=licensee, **rel_data)

    def upsert_filed_patent_with(self, c1: str, c2: str, rel_data: dict):
        patent_id = rel_data.get("patent_id", rel_data["source_url"])
        uid = generate_rel_uid(c1, c2, "FILED_PATENT_WITH", patent_id)
        rel_data["uid"] = uid
        query = f"""
        MATCH (a:Company {{name: $c1}})
        MATCH (b:Company {{name: $c2}})
        MERGE (a)-[r:FILED_PATENT_WITH {{uid: $uid}}]->(b)
        ON CREATE SET {self._on_create_props()}
        ON MATCH SET  {self._on_match_props()}
        """
        with self.driver.session() as session:
            session.run(query, c1=c1, c2=c2, **rel_data)

    def upsert_partnered_with(self, c1: str, c2: str, rel_data: dict):
        uid = generate_rel_uid(c1, c2, "PARTNERED_WITH", rel_data["source_url"])
        rel_data["uid"] = uid
        query = f"""
        MATCH (a:Company {{name: $c1}})
        MATCH (b:Company {{name: $c2}})
        MERGE (a)-[r:PARTNERED_WITH {{uid: $uid}}]->(b)
        ON CREATE SET {self._on_create_props()}
        ON MATCH SET  {self._on_match_props()}
        """
        with self.driver.session() as session:
            session.run(query, c1=c1, c2=c2, **rel_data)

    def upsert_collaborated_with(self, c1: str, c2: str, rel_data: dict):
        """
        COLLABORATED_WITH is symmetric — store canonical direction by
        alphabetical sort so duplicate edges don't accumulate from both
        directions.
        """
        a, b = sorted([c1, c2])
        uid = generate_rel_uid(a, b, "COLLABORATED_WITH", rel_data["source_url"])
        rel_data["uid"] = uid
        query = f"""
        MATCH (x:Company {{name: $a}})
        MATCH (y:Company {{name: $b}})
        MERGE (x)-[r:COLLABORATED_WITH {{uid: $uid}}]->(y)
        ON CREATE SET {self._on_create_props()}
        ON MATCH SET  {self._on_match_props()}
        """
        with self.driver.session() as session:
            session.run(query, a=a, b=b, **rel_data)

    def upsert_funded_with(self, funder: str, funded: str, rel_data: dict):
        """
        FUNDED_WITH between two Company nodes.
        Direction: funder → funded.
        """
        uid = generate_rel_uid(funder, funded, "FUNDED_WITH", rel_data["source_url"])
        rel_data["uid"] = uid
        query = f"""
        MATCH (f:Company {{name: $funder}})
        MATCH (e:Company {{name: $funded}})
        MERGE (f)-[r:FUNDED_WITH {{uid: $uid}}]->(e)
        ON CREATE SET {self._on_create_props()}
        ON MATCH SET  {self._on_match_props()}
        """
        with self.driver.session() as session:
            session.run(query, funder=funder, funded=funded, **rel_data)

    def upsert_funded_trial(self, funder: str, trial_id: str, rel_data: dict):
        """
        Company → ClinicalTrial edge for rows where entity_b is an NCT number.
        """
        tid = trial_id.upper()
        uid = generate_rel_uid(funder, tid, "FUNDED_TRIAL", rel_data["source_url"])
        rel_data["uid"] = uid
        query = f"""
        MATCH  (f:Company      {{name:     $funder}})
        MATCH  (t:ClinicalTrial {{trial_id: $trial_id}})
        MERGE  (f)-[r:FUNDED_TRIAL {{uid: $uid}}]->(t)
        ON CREATE SET {self._on_create_props()}
        ON MATCH SET  {self._on_match_props()}
        """
        with self.driver.session() as session:
            session.run(query, funder=funder, trial_id=tid, **rel_data)

    def upsert_generic_relation(self, entity_a: str, rel_type: str, entity_b: str, rel_data: dict):
        """Fallback for unmapped relation types. Backtick-quoted dynamic label."""
        uid = generate_rel_uid(entity_a, rel_type, entity_b, rel_data["source_url"])
        rel_data["uid"] = uid
        query = f"""
        MATCH (a:Company {{name: $entity_a}})
        MATCH (b:Company {{name: $entity_b}})
        MERGE (a)-[r:`{rel_type}` {{uid: $uid}}]->(b)
        ON CREATE SET {self._on_create_props()}
        ON MATCH SET  {self._on_match_props()}
        """
        with self.driver.session() as session:
            session.run(query, entity_a=entity_a, entity_b=entity_b, **rel_data)


# ==========================
# RELATION ROUTER
# ==========================

RELATION_HANDLERS = {
    "ACQUIRED":           lambda kg, a, b, d: kg.upsert_acquired(a, b, d),
    "ACQUIRED_BY":        lambda kg, a, b, d: kg.upsert_acquired(b, a, d),
    "LICENSED_TO":        lambda kg, a, b, d: kg.upsert_licensed_to(a, b, d),
    "LICENSED_FROM":      lambda kg, a, b, d: kg.upsert_licensed_to(b, a, d),
    "FILED_PATENT_WITH":  lambda kg, a, b, d: kg.upsert_filed_patent_with(a, b, d),
    "PARTNERED_WITH":     lambda kg, a, b, d: kg.upsert_partnered_with(a, b, d),
    "CO_DEVELOPED_WITH":  lambda kg, a, b, d: kg.upsert_partnered_with(a, b, d),
    # New in v2
    "COLLABORATED_WITH":  lambda kg, a, b, d: kg.upsert_collaborated_with(a, b, d),
    "FUNDED_WITH":        lambda kg, a, b, d: _route_funded_with(kg, a, b, d),
}


def _route_funded_with(kg: KnowledgeGraph, a: str, b: str, rel_data: dict):
    """
    FUNDED_WITH rows sometimes have a clinical trial ID as entity_b.
    Route to the correct edge type based on whether b is an NCT number.
    """
    if _is_clinical_trial(b):
        kg.upsert_clinical_trial(b)
        kg.upsert_funded_trial(a, b, rel_data)
    else:
        kg.upsert_funded_with(a, b, rel_data)


def route_and_upsert(kg: KnowledgeGraph, row: pd.Series) -> bool:
    raw_a = str(row["entity_a"]).strip()
    raw_b = str(row["entity_b"]).strip()
    rel_type = str(row["candidate_relation"]).strip().upper().replace(" ", "_")

    # --- Sanitize entity_a ---
    if _is_junk_entity(raw_a):
        entity_a = sanitize_entity(raw_a)
        if not entity_a:
            log.warning("Unsalvageable entity_a '%s' — skipping row.", raw_a)
            return False
        log.debug("Sanitized entity_a: '%s' → '%s'", raw_a, entity_a)
    else:
        entity_a = raw_a

    # --- Sanitize entity_b ---
    if _is_junk_entity(raw_b):
        entity_b = sanitize_entity(raw_b)
        if not entity_b:
            log.warning("Unsalvageable entity_b '%s' — skipping row.", raw_b)
            return False
        log.debug("Sanitized entity_b: '%s' → '%s'", raw_b, entity_b)
    else:
        entity_b = raw_b

    # Upsert Company nodes (clinical trial nodes handled inside _route_funded_with)
    kg.upsert_company(entity_a)
    if not _is_clinical_trial(entity_b):
        kg.upsert_company(entity_b)

    rel_data = _rel_data_from_row(row)

    handler = RELATION_HANDLERS.get(rel_type)
    try:
        if handler:
            handler(kg, entity_a, entity_b, rel_data)
        else:
            log.warning("Unknown relation type '%s' — using generic upsert.", rel_type)
            kg.upsert_generic_relation(entity_a, rel_type, entity_b, rel_data)
        return True
    except Exception as exc:
        log.error("Failed to upsert [%s → %s → %s]: %s", entity_a, rel_type, entity_b, exc)
        return False


# ==========================
# MAIN
# ==========================

def run(csv_path: Path, min_confidence: float = MIN_CONFIDENCE):
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path.name)

    for col in ("gpt4o_valid", "claude_valid"):
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: True  if str(v).strip().lower() == "true"  else
                          False if str(v).strip().lower() == "false" else None
            )

    # audit_flag column may be bool or string
    if "audit_flag" in df.columns:
        df["audit_flag"] = df["audit_flag"].map(
            lambda v: True if str(v).strip().lower() == "true" else False
        )

    kg = KnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    kg.create_constraints()

    total = len(df)
    skipped = ingested = failed = sanitized = 0

    for _, row in df.iterrows():
        if not _row_is_valid(row):
            skipped += 1
            continue

        # Count rows that needed entity sanitization for the summary
        raw_a = str(row["entity_a"]).strip()
        raw_b = str(row["entity_b"]).strip()
        if _is_junk_entity(raw_a) or _is_junk_entity(raw_b):
            sanitized += 1

        success = route_and_upsert(kg, row)
        if success:
            ingested += 1
        else:
            failed += 1

    kg.close()

    log.info("─── Ingest Summary ──────────────────────────────────────────")
    log.info("  Total rows                   : %d", total)
    log.info("  Skipped (low conf / rejected): %d", skipped)
    log.info("  Sanitized entity names       : %d", sanitized)
    log.info("  Ingested                     : %d", ingested)
    log.info("  Failed                       : %d", failed)
    log.info("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest validated relations CSV into Neo4j.")
    parser.add_argument("csv", type=Path, help="Path to validated_relations_*.csv")
    parser.add_argument(
        "--min-confidence", type=float, default=MIN_CONFIDENCE,
        help=f"Minimum final_confidence to ingest (default: {MIN_CONFIDENCE})"
    )
    args = parser.parse_args()

    if not args.csv.exists():
        log.error("File not found: %s", args.csv)
        sys.exit(1)

    run(args.csv, args.min_confidence)