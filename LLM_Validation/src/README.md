# Multi-LLM Relation Validation Pipeline

Validates candidate company relationships extracted by an NER pipeline using dual-model LLM verification (GPT-4o + Claude) and a weighted confidence scoring system. Designed for the CSL Behring knowledge graph project, where entities and relations are sourced from USPTO, SEC EDGAR, PubMed, and OpenCorporates.

---

## What It Does

1. **Loads candidate relations** from a CSV produced by the upstream NER/entity-resolution pipeline. Each row represents a proposed triple: `entity_a → relation → entity_b`, along with an evidence text snippet and source metadata.

2. **Validates each relation with two LLMs independently** — GPT-4o and Claude — using the same structured prompt. Each model returns:
   - `valid`: whether the relation is supported by the evidence text
   - `evidence_strength`: one of `strong`, `moderate`, `weak`, or `none`
   - `reasoning`: a one-sentence explanation

3. **Computes a weighted confidence score** (0.0–1.0) for each relation from four components:

   | Component | Weight | Description |
   |---|---|---|
   | Model agreement | 40% | Average mapped strength score from both models |
   | Source authority | 25% | Fixed trust score per source type (SEC=1.0, USPTO=0.9, etc.) |
   | Corroboration | 20% | How many distinct sources support the same triple (capped at 3) |
   | Recency | 15% | Exponential decay based on source date (48-month half-life) |

4. **Flags a random audit sample** (~7%) of high-confidence rows (≥0.70) for human review.

5. **Saves output** to a timestamped CSV with all model outputs, component scores, and audit flags appended.

---

## Key Design Decisions

**Ordinal strength categories instead of raw confidence floats**
Rather than asking each model to self-report a numeric confidence score, the prompt asks for a categorical strength label (`strong`, `moderate`, `weak`, `none`). These are then mapped to fixed values (1.0, 0.67, 0.33, 0.0). This avoids the calibration mismatch where GPT-4o and Claude use different internal scales for what "0.8 confidence" means.

**Fixed hyperparameters (known limitation)**
The strength-to-score mapping, source authority values, and SCORE_WEIGHTS are manually set due to the absence of labeled ground truth. The comments in the code note that logistic regression on labeled data would yield more defensible weights once enough validated examples are available.

**Short evidence text limitation**
NER extractions are sentence fragments. The model's judgment is constrained by whatever snippet is in `evidence_text`, without surrounding paragraph context. This can cause valid relationships to appear weakly supported.

---

## Configuration

All tunable parameters are at the top of the file:

```python
RECENCY_HALF_LIFE_MONTHS = 48     # How quickly older sources decay in score
HIGH_CONFIDENCE_THRESHOLD = 0.70  # Cutoff for "high confidence" classification
AUDIT_RATE = 0.07                 # Fraction of high-conf rows flagged for review
CLAUDE_MODEL = "claude-sonnet-4-6"
GPT_MODEL = "gpt-4o"
```

Source authority scores and score component weights (`SCORE_WEIGHTS`) are also defined near the top and can be adjusted.

---

## Input Format

CSV with at minimum these columns:

| Column | Description |
|---|---|
| `entity_a` | First entity in the relation |
| `candidate_relation` | Relation type (e.g., `PARTNER_OF`, `ACQUIRED_BY`) |
| `entity_b` | Second entity in the relation |
| `evidence_text` | Text snippet used to support the relation |
| `source_type` | One of `SEC`, `USPTO`, `PUBMED`, `OPENCORPORATES`, `UNKNOWN` |
| `source_id` | Unique identifier for the source document |
| `date` | Source date in `YYYY-MM` format |

---

## Output Format

Same columns as input, with these appended:

- `gpt4o_valid`, `gpt4o_evidence_strength`, `gpt4o_reasoning`
- `claude_valid`, `claude_evidence_strength`, `claude_reasoning`
- `source_authority_score`, `recency_score`, `model_agreement_ratio`
- `final_confidence`
- `audit_flag`

---

## Usage

```python
from pathlib import Path
from validation_pipeline import run

results = run(
    input_csv=Path("path/to/candidate_relations.csv"),
    output_dir=Path("outputs/"),
    openai_api_key="...",
    anthropic_api_key="...",
)
```

API keys can also be set via environment variables `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

---

## Known Limitations & Future Work

- **No ground truth**: confidence weights are heuristic. Training a logistic regression classifier on human-labeled triples would significantly improve score calibration.
- **Context window**: evidence snippets lack surrounding context, which can under-inform the models on valid relations.
- **Single reference date**: `REFERENCE_DATE` is set at import time, so recency scores shift slightly depending on when the script runs.
