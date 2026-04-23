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

from dotenv import load_dotenv
load_dotenv()


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "outputs/relations"
    / "candidate_relations_run.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

RECENCY_HALF_LIFE_MONTHS = 48
HIGH_CONFIDENCE_THRESHOLD = 0.70
AUDIT_RATE = 0.07
CLAUDE_MODEL = "claude-sonnet-4-6"
GPT_MODEL = "gpt-4o"

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STRENGTH_MAP: dict[str, float] = {
    "strong":   1.0,
    "moderate": 0.67,
    "weak":     0.33,
    "none":     0.0,
}

_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m", "%b-%y", "%B %Y", "%Y")

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
        '{"valid": true or false, '
        '"evidence_strength": "strong" or "moderate" or "weak" or "none", '
        '"reasoning": "one sentence"}'
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
                return {"valid": None, "evidence_strength": "none", "reasoning": f"error: {exc}"}
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
                return {"valid": None, "evidence_strength": "none", "reasoning": f"error: {exc}"}
            time.sleep(2**attempt)


def _recency_score(date_str: str, reference_date: datetime) -> float:
    """Exponential decay: newer sources score higher. Unknown date → 0.5 neutral."""
    source_dt = None
    for fmt in _DATE_FORMATS:
        try:
            source_dt = datetime.strptime(str(date_str).strip(), fmt)
            break
        except (ValueError, TypeError):
            continue
    if source_dt is None:
        return 0.5
    months_elapsed = max(
        (reference_date.year - source_dt.year) * 12
        + (reference_date.month - source_dt.month),
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


# returns a tuple (agreement_score, had_error) so callers can
# distinguish "one model errored" from "models disagreed"
def _model_valid_agreement(
    gpt_valid: Optional[bool], claude_valid: Optional[bool]
) -> tuple[float, bool]:
    """
    Returns (agreement_score, had_error).
    agreement_score: 1.0=both valid, 0.0=both invalid, 0.5=disagree or error
    had_error: True if at least one model returned None (API failure)
    """
    had_error = gpt_valid is None or claude_valid is None
    if gpt_valid is None and claude_valid is None:
        return 0.0, had_error
    if gpt_valid is None or claude_valid is None:
        return 0.5, had_error
    if gpt_valid is True and claude_valid is True:
        return 1.0, had_error
    if gpt_valid is False and claude_valid is False:
        return 0.0, had_error
    return 0.5, had_error  # genuine disagreement, no error


def _model_strength_score(
    gpt_strength: Optional[str], claude_strength: Optional[str]
) -> float:
    """Average mapped evidence strength across non-errored models."""
    scores = [
        STRENGTH_MAP.get(s.lower(), 0.0)
        for s in (gpt_strength, claude_strength)
        if s is not None
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _weighted_confidence(
    source_auth: float,
    corroboration_count: int,
    valid_agreement: float,
    strength_score: float,
    rec_score: float,
) -> float:
    # model_agreement uses strength_score only when models agree as valid.
    # Previously valid_agreement * strength_score collapsed to 0 when both models
    # agreed a relation was INVALID, making final_confidence useless for rejection.
    # Now acceptance and rejection paths use agreement independently of
    # strength — strength only boosts the acceptance score.
    if valid_agreement == 1.0:
        # Both valid: weight by how strongly they agreed
        model_component = strength_score
    elif valid_agreement == 0.0:
        # Both invalid: full agreement signal, strength not relevant
        model_component = 0.0
    else:
        # Disagreement or error: partial credit
        model_component = valid_agreement * strength_score

    corroboration_norm = min(corroboration_count / 3.0, 1.0)
    return round(
        SCORE_WEIGHTS["model_agreement"] * model_component
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
    corroboration_norm = min(corroboration_count / 3.0, 1.0)
    return round(
        SCORE_WEIGHTS["model_agreement"] * (1.0 - valid_agreement)
        + SCORE_WEIGHTS["source_authority"] * source_auth
        + SCORE_WEIGHTS["corroboration"] * corroboration_norm
        + SCORE_WEIGHTS["recency"] * rec_score,
        4,
    )


def _assign_bucket(row: pd.Series) -> str:
    """
    Three-bucket triage:
      auto_accept  — both models valid AND final_confidence ≥ threshold
      auto_reject  — both models invalid AND final_confidence ≥ threshold
      review       — models disagree, errors, or low confidence
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
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY"),
    random_seed: Optional[int] = 42,
) -> pd.DataFrame:
    if random_seed is not None:
        random.seed(random_seed)

    # reference date captured at run() time
    reference_date = datetime.now()

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
        rec = _recency_score(row["date"], reference_date)
        # unpack tuple — valid_agreement score + whether an API error occurred
        valid_agreement, had_error = _model_valid_agreement(gpt["valid"], claude["valid"])
        strength_score = _model_strength_score(gpt.get("evidence_strength"), claude.get("evidence_strength"))

        final_conf = _weighted_confidence(
            src_auth, int(row["corroboration_count"]), valid_agreement, strength_score, rec
        )
        reject_conf = _rejection_confidence(
            src_auth, int(row["corroboration_count"]), valid_agreement, rec
        )

        rows_out.append(
            {
                **row.to_dict(),
                # GPT-4o results
                "gpt4o_valid": gpt["valid"],
                "gpt4o_evidence_strength": gpt.get("evidence_strength"),
                "gpt4o_reasoning": gpt["reasoning"],
                # Claude results
                "claude_valid": claude["valid"],
                "claude_evidence_strength": claude.get("evidence_strength"),
                "claude_reasoning": claude["reasoning"],
                # Confidence components
                "source_authority_score": src_auth,
                "recency_score": round(rec, 4),
                "model_valid_agreement": round(valid_agreement, 4),
                "model_strength_score": round(strength_score, 4),
                # had_error surfaces whether disagreement was due to API failure
                "model_had_error": had_error,
                # Final scores
                "final_confidence": final_conf,
                "rejection_confidence": reject_conf,
            }
        )

    result_df = pd.DataFrame(rows_out)
    result_df["decision_bucket"] = result_df.apply(_assign_bucket, axis=1)
    result_df = _mark_audit_sample(result_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"validated_relations_{ts}.csv"
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
    errored = int(df["model_had_error"].sum())

    auto_accept = (df["decision_bucket"] == "auto_accept").sum()
    auto_reject = (df["decision_bucket"] == "auto_reject").sum()
    review = (df["decision_bucket"] == "review").sum()

    log.info("─── Validation Summary ──────────────────────────────────────")
    log.info("  Total relations          : %d", total)
    log.info("  Both models VALID        : %d", both_valid)
    log.info("  Both models INVALID      : %d", both_invalid)
    log.info("  Disagreement             : %d", disagree)
    log.info("  Rows with API errors     : %d", errored)
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
    run()