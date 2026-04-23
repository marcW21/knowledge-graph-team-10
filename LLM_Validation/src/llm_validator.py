#!/usr/bin/env python3

# we can improve this model by editting the hyper paramters and use more models
# right now the confidence scoring is just based on fix value

# several problems:
# since lack of ground truth, I fixed the strength map and confidence score weight
# but after getting enough data, we can atually train it by using logistic regresion to get better weight
# Short sentences lose context 
# since NER extractions are fragments but the model needs surrounding paragraphs
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
    / "candidate_relations_opt_entityruler_stage1_merged.csv"
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

# Instead of asking models to self-report a free-form confidence score (0.0–1.0),
# we ask them to classify evidence strength into four ordinal categories and map
# those categories to fixed numeric scores ourselves. This eliminates calibration
# mismatch between GPT-4o and Claude each model has its own internal scale for
# what "0.8 confidence" means, making raw averages mathematically meaningless.
# Equal-interval spacing (0, 0.33, 0.67, 1.0) is used as the default because we
# have no labeled ground truth to fit a more precise mapping.

# but the probelem is the number here has no meaning,
# we just choose it by default
# to make it better, we have to use logistic regression to get tehse value
# by doing train and test
STRENGTH_MAP: dict[str, float] = {
    "strong":   1.0,
    "moderate": 0.67,
    "weak":     0.33,
    "none":     0.0,
}

_SYSTEM_PROMPT = (
    "You are a biotech and pharmaceutical document analyst. "
    "Your task is to validate whether a company relationship extracted by an NER "
    "pipeline is actually supported by the evidence text provided. "
    "Be precise and conservative — only mark as valid if the text clearly supports "
    "the relationship."
)


def _build_batch_prompt(rows: list[pd.Series]) -> str:
    items = ""
    for i, row in enumerate(rows):
        items += (
            f"\n[{i}]\n"
            f"Source type: {row['source_type']}\n"
            f"Evidence text: \"{row['evidence_text']}\"\n"
            f"Entity A: {row['entity_a']}  |  Relation: {row['candidate_relation']}  |  Entity B: {row['entity_b']}\n"
        )
    return (
        f"Validate each of the following {len(rows)} extracted relationships independently.\n"
        "For each, reply based solely on its own evidence text — do not compare across items.\n\n"
        + items
        + "\nReply with ONLY a JSON array — one object per item, same order, no markdown:\n"
        '[{"valid": true or false, "evidence_strength": "strong"|"moderate"|"weak"|"none", "reasoning": "one sentence"}, ...]'
    )


_ERROR_RESULT = {"valid": None, "evidence_strength": "none", "reasoning": "error"}


def _parse_batch_response(text: str, expected: int) -> list[dict]:
    """Parse JSON array from model response, fall back to per-item errors."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1][4:] if parts[1].startswith("json") else parts[1]
    try:
        results = json.loads(text.strip())
        if isinstance(results, list) and len(results) == expected:
            return results
    except Exception:
        pass
    return [_ERROR_RESULT.copy() for _ in range(expected)]


def validate_batch_gpt4o(
    rows: list[pd.Series], client: openai.OpenAI, retries: int = 2
) -> list[dict]:
    prompt = _build_batch_prompt(rows)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=512,
            )
            return _parse_batch_response(resp.choices[0].message.content, len(rows))
        except Exception as exc:
            if attempt == retries:
                log.warning("GPT-4o batch failed: %s", exc)
                return [_ERROR_RESULT.copy() for _ in rows]
            time.sleep(2**attempt)


def validate_batch_claude(
    rows: list[pd.Series], client: anthropic.Anthropic, retries: int = 2
) -> list[dict]:
    prompt = _build_batch_prompt(rows)
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_batch_response(resp.content[0].text, len(rows))
        except Exception as exc:
            if attempt == retries:
                log.warning("Claude batch failed: %s", exc)
                return [_ERROR_RESULT.copy() for _ in rows]
            time.sleep(2**attempt)


def _recency_score(date_str: str) -> float:
    """Exponential decay: newer sources score higher. Unknown date → 0.5 neutral."""
    source_dt = None
    for fmt in ("%Y-%m", "%b-%y"):
        try:
            source_dt = datetime.strptime(str(date_str).strip(), fmt)
            break
        except (ValueError, TypeError):
            continue
    if source_dt is None:
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


def _model_valid_agreement(
    gpt_valid: Optional[bool], claude_valid: Optional[bool]
) -> float:
    """Do the models agree on valid/invalid? 1.0=both agree valid, 0.5=disagree, 0.0=both agree invalid."""
    if gpt_valid is None and claude_valid is None:
        return 0.0
    if gpt_valid is None or claude_valid is None:
        return 0.5
    if gpt_valid is True and claude_valid is True:
        return 1.0
    if gpt_valid is False and claude_valid is False:
        return 0.0
    return 0.5  # disagree


def _model_strength_score(
    gpt_strength: Optional[str], claude_strength: Optional[str]
) -> float:
    """Average mapped evidence strength across non-errored models."""
    scores = [
        STRENGTH_MAP.get(s, 0.0)
        for s in (gpt_strength, claude_strength)
        if s is not None
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


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


def _rejection_confidence(
    source_auth: float,
    corroboration_count: int,
    valid_agreement: float,
    rec_score: float,
) -> float:
    # invert valid_agreement: both invalid (0.0) → rejection score 1.0
    # models saying "none" evidence strength is the signal, not strength itself
    corroboration_norm = min(corroboration_count / 3.0, 1.0)
    return round(
        SCORE_WEIGHTS["model_agreement"] * (1.0 - valid_agreement)
        + SCORE_WEIGHTS["source_authority"] * source_auth
        + SCORE_WEIGHTS["corroboration"] * corroboration_norm
        + SCORE_WEIGHTS["recency"] * rec_score,
        4,
    )


# ── Decision Bucketing ────────────────────────────────────────────────────────

def _assign_bucket(row: pd.Series) -> str:
    """
    Three-bucket triage:
      auto_accept  — both models valid AND final_confidence ≥ threshold
      auto_reject  — both models invalid AND final_confidence ≥ threshold
      review       — models disagree, errors, or low confidence (review if budget allows)
    """
    gpt_valid = row["gpt4o_valid"]
    claude_valid = row["claude_valid"]
    high_conf = row["final_confidence"] >= HIGH_CONFIDENCE_THRESHOLD

    high_reject_conf = row["rejection_confidence"] >= HIGH_CONFIDENCE_THRESHOLD

    if gpt_valid is True and claude_valid is True and high_conf:
        return "auto_accept"
    if gpt_valid is False and claude_valid is False and high_reject_conf:
        return "auto_reject"
    return "review"


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

    BATCH_SIZE = 5
    all_rows = list(df.iterrows())
    batches = [all_rows[i:i + BATCH_SIZE] for i in range(0, len(all_rows), BATCH_SIZE)]
    rows_out = [None] * len(df)

    for batch_num, batch in enumerate(batches):
        batch_rows = [row for _, row in batch]
        log.info("Batch %d/%d", batch_num + 1, len(batches))

        gpt_results = validate_batch_gpt4o(batch_rows, gpt_client)
        claude_results = validate_batch_claude(batch_rows, claude_client)

        for i, (idx, row) in enumerate(batch):
            gpt = gpt_results[i]
            claude = claude_results[i]
            src_auth = SOURCE_AUTHORITY.get(str(row["source_type"]).strip().upper(), 0.5)
            rec = _recency_score(row["date"])
            valid_agreement = _model_valid_agreement(gpt["valid"], claude["valid"])
            strength_score = _model_strength_score(gpt.get("evidence_strength"), claude.get("evidence_strength"))
            model_component = valid_agreement * strength_score
            final_conf = _weighted_confidence(src_auth, int(row["corroboration_count"]), model_component, rec)
            reject_conf = _rejection_confidence(src_auth, int(row["corroboration_count"]), valid_agreement, rec)
            rows_out[idx] = {
                **row.to_dict(),
                "gpt4o_valid": gpt["valid"],
                "gpt4o_evidence_strength": gpt.get("evidence_strength"),
                "gpt4o_reasoning": gpt["reasoning"],
                "claude_valid": claude["valid"],
                "claude_evidence_strength": claude.get("evidence_strength"),
                "claude_reasoning": claude["reasoning"],
                "source_authority_score": src_auth,
                "recency_score": round(rec, 4),
                "model_valid_agreement": round(valid_agreement, 4),
                "model_strength_score": round(strength_score, 4),
                "model_component": round(model_component, 4),
                "final_confidence": final_conf,
                "rejection_confidence": reject_conf,
            }

    result_df = pd.DataFrame(rows_out)
    result_df["decision_bucket"] = result_df.apply(_assign_bucket, axis=1)
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

    auto_accept = (df["decision_bucket"] == "auto_accept").sum()
    auto_reject = (df["decision_bucket"] == "auto_reject").sum()
    review = (df["decision_bucket"] == "review").sum()

    log.info("─── Validation Summary ──────────────────────────────────────")
    log.info("  Total relations          : %d", total)
    log.info("  Both models VALID        : %d", both_valid)
    log.info("  Both models INVALID      : %d", both_invalid)
    log.info("  Disagreement             : %d", disagree)
    log.info("  High-confidence (≥%.2f)  : %d", HIGH_CONFIDENCE_THRESHOLD, high_conf)
    log.info("  ── Decision Buckets ─────────────────────────────────────")
    log.info("  auto_accept              : %d", auto_accept)
    log.info("  auto_reject              : %d", auto_reject)
    log.info("  review                   : %d", review)
    log.info(
        "  Flagged for audit        : %d  (~%.0f%% of high-conf)",
        audited,
        100 * audited / max(high_conf, 1),
    )
    log.info("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    run(
        #key
    )
