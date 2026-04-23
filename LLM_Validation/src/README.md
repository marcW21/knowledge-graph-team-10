# Multi-LLM Relation Validator

Validates candidate company relationships extracted by an NER pipeline using dual-model cross-validation (GPT-4o + Claude Sonnet). Each relation is scored for confidence, bucketed into an auto-accept / auto-reject / review queue, and a random sample of high-confidence rows is flagged for human audit.

---

## Overview

```
candidate_relations.csv
        │
        ▼
┌───────────────────┐     ┌───────────────────┐
│     GPT-4o        │     │  Claude Sonnet     │
│  (validate_with_  │     │  (validate_with_   │
│    gpt4o)         │     │    claude)         │
└────────┬──────────┘     └────────┬───────────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
         Confidence Scoring
         (model agreement × 0.40
          source authority × 0.25
          corroboration   × 0.20
          recency         × 0.15)
                    │
                    ▼
        ┌───────────────────────┐
        │  auto_accept          │  both valid + high confidence
        │  auto_reject          │  both invalid + high confidence
        │  review               │  disagree / error / low confidence
        └───────────────────────┘
                    │
                    ▼
        validated_relations_<timestamp>.csv
```

---

## Confidence Scoring

Each relation receives a `final_confidence` score in [0, 1]:

| Component | Weight | Description |
|---|---|---|
| `model_agreement` | 0.40 | Whether GPT-4o and Claude agree the relation is valid; scaled by average evidence strength |
| `source_authority` | 0.25 | Trust score by source type (SEC=1.0, USPTO=0.9, PubMed=0.8, …) |
| `corroboration` | 0.20 | Number of distinct source docs supporting the same (A, relation, B) triple, normalized at 3 |
| `recency` | 0.15 | Exponential decay with 48-month half-life from the document date |

A separate `rejection_confidence` score is computed for the auto-reject path, replacing the strength-weighted model component with `(1 - valid_agreement)`.

### Decision Buckets

| Bucket | Condition |
|---|---|
| `auto_accept` | Both models valid **and** `final_confidence ≥ 0.70` |
| `auto_reject` | Both models invalid **and** `rejection_confidence ≥ 0.70` |
| `review` | Disagreement, API error, or low confidence |

7% of high-confidence rows are randomly flagged (`audit_flag = True`) for quality sampling.

---

## Known Limitations

- **Fixed weights**: Score weights and the strength map are manually tuned. With enough labeled ground truth, these can be learned via logistic regression for better calibration.
- **Short sentences lose context**: NER extractions are often sentence fragments. The models perform better when the surrounding paragraph is included as evidence.
- **No ground truth**: The current confidence formula is a heuristic. Precision/recall cannot be measured without labeled data.

---

## Setup

```bash
pip install anthropic openai pandas
```

Set API keys as environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

**Default run** (reads from the pre-configured input path):

```bash
python validate_relations.py
```

**Programmatic use:**

```python
from validate_relations import run
from pathlib import Path

df = run(
    input_csv=Path("my_candidates.csv"),
    output_dir=Path("outputs/"),
    random_seed=42,
)
```

### Input CSV

Required columns:

| Column | Description |
|---|---|
| `entity_a` | First entity in the candidate relation |
| `candidate_relation` | Relation type (e.g., `ACQUIRED`, `LICENSED_FROM`) |
| `entity_b` | Second entity |
| `evidence_text` | Source text the relation was extracted from |
| `source_type` | One of `SEC`, `USPTO`, `PUBMED`, `OPENCORPORATES`, `UNKNOWN` |
| `source_id` | Unique identifier for the source document |
| `date` | Document date (accepts `YYYY-MM-DD`, `YYYY-MM`, `Mon-YY`, `Month YYYY`, `YYYY`) |

### Output CSV

All input columns plus:

| Column | Description |
|---|---|
| `gpt4o_valid` | Boolean (or `None` on API error) |
| `gpt4o_evidence_strength` | `strong` / `moderate` / `weak` / `none` |
| `gpt4o_reasoning` | One-sentence explanation |
| `claude_valid` | Boolean (or `None` on API error) |
| `claude_evidence_strength` | Same scale |
| `claude_reasoning` | One-sentence explanation |
| `source_authority_score` | Float from SOURCE_AUTHORITY map |
| `recency_score` | Exponential decay score |
| `model_valid_agreement` | 1.0 / 0.5 / 0.0 |
| `model_strength_score` | Average mapped evidence strength |
| `model_had_error` | True if at least one model API call failed |
| `final_confidence` | Weighted acceptance confidence |
| `rejection_confidence` | Weighted rejection confidence |
| `decision_bucket` | `auto_accept` / `auto_reject` / `review` |
| `audit_flag` | Boolean — row selected for random audit |
| `corroboration_count` | Number of distinct sources for this (A, rel, B) triple |

---

## Configuration

All tunable constants are at the top of the file:

```python
RECENCY_HALF_LIFE_MONTHS = 48       # Decay rate for document age
HIGH_CONFIDENCE_THRESHOLD = 0.70    # Minimum score for auto-accept/reject
AUDIT_RATE = 0.07                   # Fraction of high-conf rows flagged for audit
CLAUDE_MODEL = "claude-sonnet-4-6"
GPT_MODEL = "gpt-4o"

SOURCE_AUTHORITY = {
    "SEC": 1.0, "USPTO": 0.9, "OPENCORPORATES": 0.85,
    "PUBMED": 0.8, "UNKNOWN": 0.5,
}

SCORE_WEIGHTS = {
    "model_agreement": 0.40,
    "source_authority": 0.25,
    "corroboration": 0.20,
    "recency": 0.15,
}
```

---

## Future Improvements

- Replace fixed weights with logistic regression trained on labeled ground truth
- Include surrounding paragraph context in evidence prompts to improve model accuracy on short extractions
- Add a batch/async mode to reduce wall-clock time for large CSVs
- Integrate directly into the Neo4j ingestion pipeline to write accepted relations in one pass