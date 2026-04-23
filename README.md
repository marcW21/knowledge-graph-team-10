# KG Pipeline — Stage 1 (Data Collection) + Stage 2 (NER / Entity Resolution)

## Overview

This repo covers two stages of a knowledge-graph pipeline for biotech/pharma companies:

| Stage | Entry point | What it does |
|-------|-------------|--------------|
| 1 | `app.py` | Queries USPTO, PubMed, SEC, OpenCorporates for each company seed; writes `stage1_records.csv` |
| 2 | `run_pipeline.py` | Preprocess → NER → Alias resolution → Candidate relation extraction |

## File layout

```
├── app.py                        Stage 1 data collection
├── config.py                     Settings (reads from .env)
├── schema.py                     Stage1Record dataclass
├── run_pipeline.py               Stage 2 end-to-end runner
├── src/
│   ├── constants.py              Shared constants (suffixes, corrupt tokens, …)
│   ├── preprocess.py             Stage 1 seed cleaning + Stage 2 NER input cleaning
│   ├── ner.py                    spaCy + EntityRuler NER
│   ├── resolve_alias.py          Conservative alias resolution
│   └── extract_candidate_relations.py  Source-aware relation extraction
├── data/
│   ├── raw/                      Stage 1 output CSVs go here
│   └── processed/                Cleaned / invalid split outputs
└── outputs/
    ├── ner/
    ├── entity_resolution/
    └── relations/
```

## Quick start

```bash
pip install pandas spacy python-dotenv ftfy python-dateutil
python -m spacy download en_core_web_sm

# Stage 1 (data collection — needs API keys in .env)
python app.py

# Stage 2 (NER pipeline)
python run_pipeline.py \
    --input  data/raw/stage1_records.csv \
    --run-id run
```

## Input / Output schema

### Stage 1 output / Stage 2 input (`stage1_records.csv`)
`source_id`, `source_type`, `raw_text`, `source_url`, `date`, `company_seed`

### Final outputs for downstream LLM validation
| File | Purpose |
|------|---------|
| `outputs/ner/ner_results_{run_id}.csv` | One row per entity mention |
| `outputs/entity_resolution/resolved_entities_{run_id}.csv` | Canonical entity layer |
| `outputs/entity_resolution/review_queue_{run_id}.csv` | Ambiguous merge pairs for manual review |
| `outputs/relations/candidate_relations_{run_id}.csv` | Company–company relation candidates |

## NER strategy (Stage 2)

**Why spaCy + EntityRuler over BioBERT?**  
The bottleneck is *company-name* recall, not biomedical entity tagging. The EntityRuler is seeded directly from the `company_seed` list and its legal-suffix variants, which recovers mentions that spaCy's statistical NER misses (all-caps filings, partial names, etc.).

## Key design decisions

### Alias resolution (`resolve_alias.py`)
- **Bug fixed from original:** the original code built `pairs_to_compare` from the full unique-name list, accidentally comparing names across completely different companies (O(n²)). The fix groups candidates by `base_name` first and only evaluates pairs *within* each group.
- Decision ladder: exact match → same base + generic suffix → high string similarity → do-not-merge.
- Ambiguous pairs go to `review_queue` instead of being force-merged.

### Relation extraction (`extract_candidate_relations.py`)
- USPTO relations only fire when there is an explicit co-assignee / joint-patent signal — avoids combinatorial explosion from single-assignee patent metadata.
- SEC has a document-level fallback pass because acquirer and target often appear in separate sentences.
- `KNOWN_PHARMA_ABBREVS` and all filter sets are module-level constants (not recreated per call).
