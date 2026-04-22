#!/usr/bin/env python3

# we can improve this model by editting the hyper paramters and use more models
# right now the confidence scoring is just based on fix value
from __future__ import annotations
import json
import logging
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
import openai
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "NER_EntityResolution/outputs/relations"
    / "candidate_relations_mock2_entityruler_preprocess.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# same source would scoree diff since we use curretn date
REFERENCE_DATE = datetime.now()
# change this based on needed
RECENCY_HALF_LIFE_MONTHS = 48
# change this as well
HIGH_CONFIDENCE_THRESHOLD = 0.70
# we change we this rate based on what company want
AUDIT_RATE = 0.07
CLAUDE_MODEL = "claude-sonnet-4-6"
GPT_MODEL = "gpt-4o"

# just some hyper params
SOURCE_AUTHORITY: dict[str, float] = {
    "SEC": 1.0,
    "USPTO": 0.9,
    "OPENCORPORATES": 0.85,
    "PUBMED": 0.8,
    "UNKNOWN": 0.5,
}

SCORE_WEIGHTS = {
    "model_agreement": 0.40,
    "source_authority": 0.25,
    "corroboration": 0.20,
    "recency": 0.15,
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a biotech and pharmaceutical document analyst. "
    "Your task is to validate whether a company relationship extracted by an NER "
    "pipeline is actually supported by the evidence text provided. "
    "Be precise and conservative — only mark as valid if the text clearly supports "
    "the relationship."
)


def _build_user_prompt(row: pd.Series) -> str:
    return (
        f"Source type: {row['source_type']}\n"
        f"Evidence text: \"{row['evidence_text']}\"\n\n"
        f"Extracted relationship:\n"
        f"  Entity A : {row['entity_a']}\n"
        f"  Relation : {row['candidate_relation']}\n"
        f"  Entity B : {row['entity_b']}\n\n"
        "Based solely on the evidence text above, does this relationship hold?\n"
        "Reply with ONLY a JSON object — no markdown, no extra text:\n"
        '{"valid": true or false, "confidence": 0.0-1.0, "reasoning": "one sentence"}'
    )


def _parse_response(text: str) -> dict:
    """Parse JSON from model response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def validate_with_gpt4o(
    row: pd.Series, client: openai.OpenAI, retries: int = 2
) -> dict:
    prompt = _build_user_prompt(row)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=256,
            )
            return _parse_response(resp.choices[0].message.content)
        except Exception as exc:
            if attempt == retries:
                log.warning("GPT-4o failed (row %s): %s", row.get("source_id"), exc)
                return {"valid": None, "confidence": 0.0, "reasoning": f"error: {exc}"}
            time.sleep(2**attempt)


def validate_with_claude(
    row: pd.Series, client: anthropic.Anthropic, retries: int = 2
) -> dict:
    prompt = _build_user_prompt(row)
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_response(resp.content[0].text)
        except Exception as exc:
            if attempt == retries:
                log.warning("Claude failed (row %s): %s", row.get("source_id"), exc)
                return {"valid": None, "confidence": 0.0, "reasoning": f"error: {exc}"}
            time.sleep(2**attempt)



def _recency_score(date_str: str) -> float:
    """Exponential decay: newer sources score higher. Unknown date → 0.5 neutral."""
    try:
        source_dt = datetime.strptime(str(date_str).strip(), "%Y-%m")
    except (ValueError, TypeError):
        return 0.5
    months_elapsed = max(
        (REFERENCE_DATE.year - source_dt.year) * 12
        + (REFERENCE_DATE.month - source_dt.month),
        0,
    )
    return math.exp(-math.log(2) * months_elapsed / RECENCY_HALF_LIFE_MONTHS)


def _add_corroboration_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count distinct source_ids that share the same (entity_a, relation, entity_b) triple."""
    df = df.copy()
    df["_key_a"] = df["entity_a"].str.strip().str.upper()
    df["_key_rel"] = df["candidate_relation"].str.strip().str.upper()
    df["_key_b"] = df["entity_b"].str.strip().str.upper()
    df["corroboration_count"] = df.groupby(["_key_a", "_key_rel", "_key_b"])[
        "source_id"
    ].transform("nunique")
    return df.drop(columns=["_key_a", "_key_rel", "_key_b"])


def _model_agreement_ratio(
    gpt_valid: Optional[bool], claude_valid: Optional[bool]
) -> float:
    """Fraction of non-errored LLMs that marked the relation as valid."""
    votes = [v for v in (gpt_valid, claude_valid) if v is not None]
    if not votes:
        return 0.0
    return sum(bool(v) for v in votes) / len(votes)


def _weighted_confidence(
    source_auth: float,
    corroboration_count: int,
    model_agreement: float,
    rec_score: float,
) -> float:
    corroboration_norm = min(corroboration_count / 3.0, 1.0)
    return round(
        SCORE_WEIGHTS["model_agreement"] * model_agreement
        + SCORE_WEIGHTS["source_authority"] * source_auth
        + SCORE_WEIGHTS["corroboration"] * corroboration_norm
        + SCORE_WEIGHTS["recency"] * rec_score,
        4,
    )


# ── Audit Sampling ────────────────────────────────────────────────────────────

def _mark_audit_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Randomly flag AUDIT_RATE fraction of high-confidence rows for human review."""
    high_conf_idx = df.index[df["final_confidence"] >= HIGH_CONFIDENCE_THRESHOLD].tolist()
    n_sample = max(1, round(len(high_conf_idx) * AUDIT_RATE))
    audit_idx = set(random.sample(high_conf_idx, min(n_sample, len(high_conf_idx))))
    df["audit_flag"] = df.index.isin(audit_idx)
    return df



def run(
    input_csv: Path = DEFAULT_INPUT,
    output_dir: Path = OUTPUT_DIR,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    random_seed: Optional[int] = 42,
) -> pd.DataFrame:
    if random_seed is not None:
        random.seed(random_seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    log.info("Loaded %d candidate relations from %s", len(df), input_csv.name)

    df = _add_corroboration_counts(df)

    gpt_client = openai.OpenAI(api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
    claude_client = anthropic.Anthropic(
        api_key=anthropic_api_key or os.environ["ANTHROPIC_API_KEY"]
    )

    rows_out = []
    for idx, row in df.iterrows():
        log.info(
            "[%d/%d] %s → %s → %s",
            idx + 1, len(df),
            row["entity_a"], row["candidate_relation"], row["entity_b"],
        )

        gpt = validate_with_gpt4o(row, gpt_client)
        claude = validate_with_claude(row, claude_client)

        src_auth = SOURCE_AUTHORITY.get(str(row["source_type"]).strip().upper(), 0.5)
        rec = _recency_score(row["date"])
        agreement = _model_agreement_ratio(gpt["valid"], claude["valid"])
        final_conf = _weighted_confidence(
            src_auth, int(row["corroboration_count"]), agreement, rec
        )

        rows_out.append(
            {
                **row.to_dict(),
                # GPT-4o results
                "gpt4o_valid": gpt["valid"],
                "gpt4o_confidence": gpt["confidence"],
                "gpt4o_reasoning": gpt["reasoning"],
                # Claude results
                "claude_valid": claude["valid"],
                "claude_confidence": claude["confidence"],
                "claude_reasoning": claude["reasoning"],
                # Confidence components
                "source_authority_score": src_auth,
                "recency_score": round(rec, 4),
                "model_agreement_ratio": round(agreement, 4),
                # Final score
                "final_confidence": final_conf,
            }
        )

    result_df = pd.DataFrame(rows_out)
    result_df = _mark_audit_sample(result_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"validated_relations_mock2_{ts}.csv"
    result_df.to_csv(out_path, index=False)
    log.info("Saved results → %s", out_path)

    _print_summary(result_df)
    return result_df


def _print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    both_valid = ((df["gpt4o_valid"] == True) & (df["claude_valid"] == True)).sum()
    both_invalid = ((df["gpt4o_valid"] == False) & (df["claude_valid"] == False)).sum()
    disagree = total - both_valid - both_invalid
    high_conf = (df["final_confidence"] >= HIGH_CONFIDENCE_THRESHOLD).sum()
    audited = int(df["audit_flag"].sum())

    log.info("─── Validation Summary ──────────────────────────────────────")
    log.info("  Total relations          : %d", total)
    log.info("  Both models VALID        : %d", both_valid)
    log.info("  Both models INVALID      : %d", both_invalid)
    log.info("  Disagreement             : %d", disagree)
    log.info("  High-confidence (≥%.2f)  : %d", HIGH_CONFIDENCE_THRESHOLD, high_conf)
    log.info(
        "  Flagged for audit        : %d  (~%.0f%% of high-conf)",
        audited,
        100 * audited / max(high_conf, 1),
    )
    log.info("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    run(
        openai_api_key="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",
        anthropic_api_key="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",
    )
