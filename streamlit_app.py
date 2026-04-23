"""
CSL Behring Knowledge Graph Pipeline — FIXED VERSION
Run with: streamlit run streamlit_app.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import sys
import textwrap
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SRC = ROOT / "src"
LLM_SRC = ROOT / "LLM_Validation" / "src"

for p in [LLM_SRC, SRC, ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ─────────────────────────────────────────────────────────────
# SCHEMA GUARDS (prevents silent pipeline corruption)
# ─────────────────────────────────────────────────────────────

REQUIRED_STAGE1 = {"source_id", "company_seed", "source_type", "date", "raw_text"}

def validate_columns(df: pd.DataFrame, required: set, stage: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{stage}] Missing required columns: {missing}")

def safe_get(d: dict, k: str, default=None):
    return d.get(k, default)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KG Pipeline Demo (Fixed)",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for k in [
    "stage1_df", "stage2_df", "stage3_df",
    "stage3_review_df", "stage4_df", "stage4_raw_df",
    "stage5_df"
]:
    if k not in st.session_state:
        st.session_state[k] = None

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.title("Knowledge Graph Pipeline (Fixed + Hardened)")

# ─────────────────────────────────────────────────────────────
# STAGE 1 — DATA COLLECTION
# ─────────────────────────────────────────────────────────────
st.header("Stage 1 — Data Collection")

if st.button("Run Stage 1"):

    from app import collect_for_company
    from config import Settings

    settings = Settings()
    settings.max_records_per_source = 10

    seeds = ["Pfizer", "Moderna", "Merck", "Novartis"]

    all_records = []
    next_id = 1

    for seed in seeds:
        batch, next_id = collect_for_company(seed, next_id, settings)
        all_records.extend([r.to_dict() for r in batch])

    df = pd.DataFrame(all_records)

    validate_columns(df, REQUIRED_STAGE1, "STAGE 1")

    st.session_state.stage1_df = df
    st.session_state.stage2_df = None
    st.session_state.stage3_df = None
    st.session_state.stage4_df = None
    st.session_state.stage5_df = None

if st.session_state.stage1_df is not None:
    st.dataframe(st.session_state.stage1_df.head(20))

# ─────────────────────────────────────────────────────────────
# STAGE 2 — NER
# ─────────────────────────────────────────────────────────────
st.header("Stage 2 — NER")

if st.session_state.stage1_df is not None and st.button("Run Stage 2"):

    import spacy
    from preprocess import preprocess_dataframe
    from ner import _add_entity_ruler, extract_mentions, dedupe_mentions

    df_raw = st.session_state.stage1_df.copy()

    valid_df, invalid_df = preprocess_dataframe(df_raw)

    nlp = spacy.load("en_core_web_sm")
    nlp = _add_entity_ruler(nlp, valid_df)

    records = extract_mentions(valid_df, nlp, {"ORG", "PRODUCT", "EVENT"})
    ner_df = dedupe_mentions(records)

    st.session_state.stage2_df = ner_df

if st.session_state.stage2_df is not None:
    st.dataframe(st.session_state.stage2_df.head(20))

# ─────────────────────────────────────────────────────────────
# STAGE 3 — ENTITY RESOLUTION
# ─────────────────────────────────────────────────────────────
st.header("Stage 3 — Entity Resolution")

if st.session_state.stage2_df is not None and st.button("Run Stage 3"):

    from resolve_alias import resolve

    resolved_df, review_df = resolve(st.session_state.stage2_df)

    # normalize canonical names (FIXED)
    resolved_df["canonical_name"] = (
        resolved_df["canonical_name"].astype(str).str.strip().str.lower()
    )

    st.session_state.stage3_df = resolved_df
    st.session_state.stage3_review_df = review_df

if st.session_state.stage3_df is not None:
    st.dataframe(st.session_state.stage3_df.head(20))

# ─────────────────────────────────────────────────────────────
# STAGE 4 — RELATION EXTRACTION (FIXED: no data loss)
# ─────────────────────────────────────────────────────────────
st.header("Stage 4 — Relation Extraction")

if st.session_state.stage3_df is not None and st.button("Run Stage 4"):

    from extract_candidate_relations import build_candidate_rows, _assign_sentence

    df_res = st.session_state.stage3_df.copy()

    df_res["sentence_text"] = df_res.apply(
        lambda r: _assign_sentence(str(r.get("raw_text", "")), r.get("start_char", -1)),
        axis=1
    )

    rows = build_candidate_rows(df_res)

    raw_df = pd.DataFrame(rows)

    # SAVE BOTH (FIX)
    st.session_state.stage4_raw_df = raw_df.copy()

    def clean(df):
        if df.empty:
            return df

        df = df[
            df["entity_a"].astype(str).str.len() > 2
            & df["entity_b"].astype(str).str.len() > 2
            & (df["entity_a"] != df["entity_b"])
        ]
        return df

    st.session_state.stage4_df = clean(raw_df)

if st.session_state.stage4_df is not None:
    st.dataframe(st.session_state.stage4_df.head(20))

# ─────────────────────────────────────────────────────────────
# STAGE 5 — LLM VALIDATION (FIXED sampling + safety)
# ─────────────────────────────────────────────────────────────
st.header("Stage 5 — LLM Validation")

if st.session_state.stage4_df is not None and st.button("Run Stage 5"):

    import anthropic
    from datetime import datetime
    from llm_validator import validate_with_claude

    df = st.session_state.stage4_df

    # FIX: sample only for compute, not destruction
    sample = df.sample(min(5, len(df)), random_state=42).copy()

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    outputs = []

    for _, row in sample.iterrows():

        try:
            claude = validate_with_claude(row, client)
        except Exception:
            claude = {"valid": None, "reasoning": ""}

        outputs.append({
            **row.to_dict(),
            "claude_valid": claude.get("valid"),
            "claude_reasoning": claude.get("reasoning"),
            "final_confidence": 0.75 if claude.get("valid") else 0.25
        })

    result_df = pd.DataFrame(outputs)

    st.session_state.stage5_df = result_df

if st.session_state.stage5_df is not None:
    st.dataframe(st.session_state.stage5_df)

# ─────────────────────────────────────────────────────────────
# STAGE 6 — GRAPH (FIXED TEMP FILE + NORMALIZATION)
# ─────────────────────────────────────────────────────────────
st.header("Stage 6 — Graph")

if st.session_state.stage5_df is not None and st.button("Render Graph"):

    from pyvis.network import Network
    import streamlit.components.v1 as components

    df = st.session_state.stage5_df.copy()

    df["entity_a"] = df["entity_a"].astype(str).str.lower().str.strip()
    df["entity_b"] = df["entity_b"].astype(str).str.lower().str.strip()

    net = Network(height="600px", width="100%", bgcolor="#0a0f1e")

    nodes = set(df["entity_a"]).union(set(df["entity_b"]))

    for n in nodes:
        net.add_node(n, label=n)

    for _, r in df.iterrows():
        net.add_edge(r["entity_a"], r["entity_b"])

    path = f"/tmp/kg_{uuid.uuid4().hex}.html"   # FIX collision

    net.save_graph(path)

    with open(path) as f:
        components.html(f.read(), height=600)