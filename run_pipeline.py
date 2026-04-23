#!/usr/bin/env python3
"""End-to-end NER + entity resolution + relation extraction runner.

Equivalent to run_entityruler_opt_stage1_merged.ipynb but runnable anywhere,
not just Colab. Run from the project root:

    python run_pipeline.py \
        --input  data/raw/stage1_records_merged.csv \
        --run-id run

Outputs land in:
    data/processed/cleaned_input_{run_id}.csv
    data/processed/invalid_rows_{run_id}.csv
    outputs/ner/ner_results_{run_id}.csv
    outputs/entity_resolution/resolved_entities_{run_id}.csv
    outputs/entity_resolution/review_queue_{run_id}.csv
    outputs/relations/candidate_relations_{run_id}.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Raw stage-1 CSV")
    p.add_argument("--run-id", default="run", help="Label appended to all output filenames")
    p.add_argument("--spacy-model", default="en_core_web_sm")
    args = p.parse_args()

    rid = args.run_id
    src = "src"

    cleaned   = f"data/processed/cleaned_input_{rid}.csv"
    invalid   = f"data/processed/invalid_rows_{rid}.csv"
    ner_out   = f"outputs/ner/ner_results_{rid}.csv"
    resolved  = f"outputs/entity_resolution/resolved_entities_{rid}.csv"
    review    = f"outputs/entity_resolution/review_queue_{rid}.csv"
    relations = f"outputs/relations/candidate_relations_{rid}.csv"

    _run([sys.executable, f"{src}/preprocess.py",
          "--input", args.input,
          "--cleaned-output", cleaned,
          "--invalid-output", invalid])

    _run([sys.executable, f"{src}/ner.py",
          "--input", cleaned,
          "--output", ner_out,
          "--spacy-model", args.spacy_model])

    _run([sys.executable, f"{src}/resolve_alias.py",
          "--input", ner_out,
          "--resolved-output", resolved,
          "--review-output", review])

    _run([sys.executable, f"{src}/extract_candidate_relations.py",
          "--input", resolved,
          "--output", relations])

    print(f"\n✓ Pipeline complete. Final relations: {relations}")


if __name__ == "__main__":
    main()
