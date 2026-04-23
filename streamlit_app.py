"""
CSL Behring Knowledge Graph Pipeline — Demo UI
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
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KG Pipeline Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1f3c 60%, #0a2640 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,120,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #e8f4ff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #7aa8cc;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
.hero .tag {
    display: inline-block;
    background: rgba(0,120,255,0.15);
    border: 1px solid rgba(0,120,255,0.3);
    color: #4a9eff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 1rem;
    letter-spacing: 1px;
}

/* Stage cards */
.stage-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1rem;
}
.stage-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4a9eff;
    background: rgba(74,158,255,0.1);
    border: 1px solid rgba(74,158,255,0.25);
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 1px;
}
.stage-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e8f4ff;
    margin: 0;
}

/* Metrics row */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 10px 16px;
    min-width: 110px;
}
.metric-pill .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #4a9eff;
    line-height: 1;
}
.metric-pill .lbl {
    font-size: 0.72rem;
    color: #7aa8cc;
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Decision badges */
.badge-accept  { background:#0d3320; color:#3ddc84; border:1px solid #1a5a38; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-family:'IBM Plex Mono',monospace; }
.badge-reject  { background:#3d0d0d; color:#ff6b6b; border:1px solid #5a1a1a; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-family:'IBM Plex Mono',monospace; }
.badge-review  { background:#332b00; color:#ffc107; border:1px solid #5a4800; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-family:'IBM Plex Mono',monospace; }
.badge-audit   { background:#1a1a3d; color:#a78bfa; border:1px solid #3a2f7a; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-family:'IBM Plex Mono',monospace; }

/* Relation card */
.rel-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid #4a9eff;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}
.rel-card .entities {
    font-family: 'IBM Plex Mono', monospace;
    color: #e8f4ff;
    font-size: 0.82rem;
}
.rel-card .rel-type {
    color: #4a9eff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
    margin: 2px 0;
}
.rel-card .evidence {
    color: #7aa8cc;
    font-size: 0.78rem;
    margin-top: 4px;
    font-style: italic;
}

/* Locked stage */
.locked {
    opacity: 0.4;
    pointer-events: none;
}
.lock-msg {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #4a6680;
    border: 1px dashed #1e3a5f;
    border-radius: 6px;
    padding: 8px 14px;
    margin-top: 8px;
}

/* Divider */
.stage-divider {
    border: none;
    border-top: 1px solid #1e3a5f;
    margin: 1.5rem 0;
}

/* Confidence bar */
.conf-bar-wrap { background:#0d1f3c; border-radius:4px; height:8px; width:100%; margin-top:4px; }
.conf-bar { height:8px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_seed_list(path: str) -> list[str]:
    seeds = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                seeds.append(line)
    return seeds


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metric_html(val, label):
    return f'<div class="metric-pill"><div class="val">{val}</div><div class="lbl">{label}</div></div>'


def badge(decision: str) -> str:
    cls = {"auto_accept": "badge-accept", "auto_reject": "badge-reject"}.get(decision, "badge-review")
    return f'<span class="{cls}">{decision}</span>'


def conf_bar(score: float) -> str:
    color = "#3ddc84" if score >= 0.7 else "#ffc107" if score >= 0.5 else "#ff6b6b"
    pct = int(score * 100)
    return f'<div class="conf-bar-wrap"><div class="conf-bar" style="width:{pct}%;background:{color};"></div></div>'


# ── Session state init ────────────────────────────────────────────────────────
for k in ["stage1_df", "stage2_df", "stage3_df", "stage3_review_df",
          "stage4_df", "stage5_df"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="tag">CORNELL TECH · CSL BEHRING · STUDIO PROJECT</div>
  <h1>Knowledge Graph Pipeline</h1>
  <p>Multi-source ingestion → NER → entity resolution → relation extraction → LLM cross-validation → Neo4j</p>
</div>
""", unsafe_allow_html=True)

# ── Load seeds ────────────────────────────────────────────────────────────────
def _find_seed_file(root: Path) -> Path | None:
    """Find the organizations seed file regardless of exact filename variant."""
    for candidate in sorted(root.glob("organizations*.txt")):
        return candidate
    return None

_seed_path = _find_seed_file(ROOT)
try:
    ALL_SEEDS = load_seed_list(str(_seed_path)) if _seed_path else []
    if not ALL_SEEDS:
        raise FileNotFoundError
except (FileNotFoundError, TypeError):
    ALL_SEEDS = [
        "Pfizer", "CSL Behring", "Moderna", "Johnson & Johnson", "Merck",
        "AbbVie", "Amgen", "Gilead Sciences", "Novartis", "10X Genomics",
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Data Collection
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("**STAGE 01 — Data Collection**", expanded=True):
    st.markdown('<div class="stage-num">01 / DATA COLLECTION</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        n_companies = st.slider(
            "Number of companies to sample",
            min_value=1, max_value=min(20, len(ALL_SEEDS)),
            value=5, step=1,
        )
        seed_sample = random.sample(ALL_SEEDS, k=min(n_companies, len(ALL_SEEDS)))
        st.caption(f"**Sampled seeds:** {', '.join(seed_sample)}")

    with col_right:
        st.markdown("**Active sources**")
        use_sec    = st.checkbox("SEC / EDGAR", value=True)
        use_pubmed = st.checkbox("PubMed / NCBI", value=True)
        use_uspto  = st.checkbox("USPTO", value=False)

    run_s1 = st.button("▶ Run Data Collection", key="run_s1", type="primary")

    if run_s1:
        try:
            from config import Settings
            from src.preprocess import normalize_company_seed, parse_date_to_mon_yy
            from schema import Stage1Record

            settings = Settings()
            settings.sec_enabled    = use_sec
            settings.ncbi_enabled   = use_pubmed
            settings.uspto_enabled  = False  # requires API key; skip silently
            settings.max_records_per_source = 10

            from app import collect_for_company

            all_records = []
            next_id = 1
            progress = st.progress(0, text="Collecting…")
            for i, seed in enumerate(seed_sample):
                progress.progress((i + 1) / len(seed_sample), text=f"Querying: {seed}")
                batch, next_id = collect_for_company(seed, next_id, settings)
                all_records.extend(r.to_dict() for r in batch)

            progress.empty()
            df = pd.DataFrame(all_records)
            st.session_state["stage1_df"] = df
            st.session_state["stage1_seeds"] = seed_sample
            # Invalidate downstream
            for k in ["stage2_df", "stage3_df", "stage3_review_df", "stage4_df", "stage5_df"]:
                st.session_state[k] = None

        except Exception as exc:
            st.error(f"Stage 1 failed: {exc}")

    if st.session_state["stage1_df"] is not None:
        df1 = st.session_state["stage1_df"]
        counts = df1.groupby("source_type").size().to_dict() if "source_type" in df1.columns else {}
        pills = "".join(metric_html(v, k) for k, v in counts.items())
        pills += metric_html(len(df1), "total rows")
        st.markdown(f'<div class="metric-row">{pills}</div>', unsafe_allow_html=True)
        st.dataframe(df1[["source_id", "company_seed", "source_type", "date", "raw_text"]].head(50),
                     use_container_width=True, height=220)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Preprocessing + NER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
with st.expander("**STAGE 02 — Preprocessing + NER**",
                 expanded=st.session_state["stage1_df"] is not None):

    st.markdown('<div class="stage-num">02 / PREPROCESS + NER</div>', unsafe_allow_html=True)

    if st.session_state["stage1_df"] is None:
        st.markdown('<div class="lock-msg">⟳ Complete Stage 01 first.</div>', unsafe_allow_html=True)
    else:
        spacy_model = st.selectbox("spaCy model", ["en_core_web_sm", "en_core_web_md"], index=0)
        run_s2 = st.button("▶ Run Preprocessing + NER", key="run_s2", type="primary")

        if run_s2:
            try:
                from preprocess import preprocess_dataframe
                import spacy
                from ner import _add_entity_ruler, extract_mentions, dedupe_mentions

                df_raw = st.session_state["stage1_df"].copy()
                with st.spinner("Preprocessing…"):
                    valid_df, invalid_df = preprocess_dataframe(df_raw)

                with st.spinner(f"Loading {spacy_model} + running NER…"):
                    nlp = spacy.load(spacy_model)
                    nlp = _add_entity_ruler(nlp, valid_df)
                    records = extract_mentions(valid_df, nlp, {"ORG"}, batch_size=16)
                    ner_df = dedupe_mentions(records)

                st.session_state["stage2_df"] = ner_df
                st.session_state["stage2_invalid_count"] = len(invalid_df)
                for k in ["stage3_df", "stage3_review_df", "stage4_df", "stage5_df"]:
                    st.session_state[k] = None

            except Exception as exc:
                st.error(f"Stage 2 failed: {exc}")

        if st.session_state["stage2_df"] is not None:
            ner_df = st.session_state["stage2_df"]
            inv = st.session_state.get("stage2_invalid_count", 0)
            pills = (metric_html(len(ner_df), "mentions") +
                     metric_html(ner_df["raw_mention"].nunique() if "raw_mention" in ner_df.columns else 0, "unique entities") +
                     metric_html(inv, "invalid rows"))
            st.markdown(f'<div class="metric-row">{pills}</div>', unsafe_allow_html=True)
            show_cols = [c for c in ["source_id", "company_seed", "raw_mention",
                                      "entity_label", "start_char", "source_type"] if c in ner_df.columns]
            st.dataframe(ner_df[show_cols].head(50), use_container_width=True, height=220)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Entity Resolution
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
with st.expander("**STAGE 03 — Entity Resolution**",
                 expanded=st.session_state["stage2_df"] is not None):

    st.markdown('<div class="stage-num">03 / ENTITY RESOLUTION</div>', unsafe_allow_html=True)

    if st.session_state["stage2_df"] is None:
        st.markdown('<div class="lock-msg">⟳ Complete Stage 02 first.</div>', unsafe_allow_html=True)
    else:
        run_s3 = st.button("▶ Run Entity Resolution", key="run_s3", type="primary")

        if run_s3:
            try:
                from resolve_alias import resolve
                with st.spinner("Resolving aliases…"):
                    resolved_df, review_df = resolve(st.session_state["stage2_df"])
                st.session_state["stage3_df"] = resolved_df
                st.session_state["stage3_review_df"] = review_df
                for k in ["stage4_df", "stage5_df"]:
                    st.session_state[k] = None
            except Exception as exc:
                st.error(f"Stage 3 failed: {exc}")

        if st.session_state["stage3_df"] is not None:
            res_df = st.session_state["stage3_df"]
            rev_df = st.session_state["stage3_review_df"]
            auto_merged = (res_df["merge_decision"] == "auto_merge").sum() if "merge_decision" in res_df.columns else 0
            singletons  = (res_df["merge_decision"] == "singleton").sum() if "merge_decision" in res_df.columns else 0
            pills = (metric_html(len(res_df), "resolved rows") +
                     metric_html(auto_merged, "auto-merged") +
                     metric_html(len(rev_df), "for review"))
            st.markdown(f'<div class="metric-row">{pills}</div>', unsafe_allow_html=True)

            tab_res, tab_rev = st.tabs(["Resolved entities", "Review queue"])
            with tab_res:
                show_cols = [c for c in ["raw_mention", "canonical_name", "merge_decision",
                                          "merge_confidence", "evidence_note"] if c in res_df.columns]
                st.dataframe(res_df[show_cols].head(50), use_container_width=True, height=220)
            with tab_rev:
                if len(rev_df) == 0:
                    st.info("No pairs flagged for review.")
                else:
                    st.dataframe(rev_df.head(30), use_container_width=True, height=200)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Relation Extraction
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
with st.expander("**STAGE 04 — Relation Extraction**",
                 expanded=st.session_state["stage3_df"] is not None):

    st.markdown('<div class="stage-num">04 / RELATION EXTRACTION</div>', unsafe_allow_html=True)

    if st.session_state["stage3_df"] is None:
        st.markdown('<div class="lock-msg">⟳ Complete Stage 03 first.</div>', unsafe_allow_html=True)
    else:
        run_s4 = st.button("▶ Extract Candidate Relations", key="run_s4", type="primary")

        if run_s4:
            try:
                from extract_candidate_relations import build_candidate_rows
                df_res = st.session_state["stage3_df"].copy()
                with st.spinner("Extracting relations…"):
                    df_res["sentence_text"] = df_res.apply(
                        lambda r: str(r.get("raw_text", ""))[:500], axis=1
                    )
                    rows = build_candidate_rows(df_res)
                    cand_df = pd.DataFrame(rows) if rows else pd.DataFrame()

                st.session_state["stage4_df"] = cand_df
                st.session_state["stage5_df"] = None
            except Exception as exc:
                st.error(f"Stage 4 failed: {exc}")

        if st.session_state["stage4_df"] is not None:
            cand_df = st.session_state["stage4_df"]
            if len(cand_df) == 0:
                st.warning("No candidate relations found. Try enabling more sources or increasing N.")
            else:
                rel_counts = cand_df["candidate_relation"].value_counts().to_dict() if "candidate_relation" in cand_df.columns else {}
                pills = (metric_html(len(cand_df), "candidates") +
                         "".join(metric_html(v, k) for k, v in rel_counts.items()))
                st.markdown(f'<div class="metric-row">{pills}</div>', unsafe_allow_html=True)

                # Show as cards (first 15)
                display_df = cand_df.head(15)
                for _, row in display_df.iterrows():
                    ev = textwrap.shorten(str(row.get("evidence_text", "")), width=140, placeholder="…")
                    rel_color_map = {
                        "ACQUIRED": "#ff6b6b", "LICENSED_FROM": "#ffc107", "LICENSED_TO": "#ffc107",
                        "PARTNERED_WITH": "#4a9eff", "COLLABORATED_WITH": "#a78bfa",
                        "FILED_PATENT_WITH": "#3ddc84", "FUNDED_WITH": "#fb923c",
                    }
                    rel = str(row.get("candidate_relation", ""))
                    rel_color = rel_color_map.get(rel, "#7aa8cc")
                    st.markdown(f"""
                    <div class="rel-card">
                      <div class="entities">
                        {row.get('entity_a','?')} <span style="color:{rel_color}"> ──{rel}──▶ </span> {row.get('entity_b','?')}
                      </div>
                      <div class="evidence">"{ev}"</div>
                      <div style="font-size:0.7rem;color:#4a6680;margin-top:4px;">
                        {row.get('source_type','')} · {row.get('date','')}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — LLM Validation
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
with st.expander("**STAGE 05 — LLM Cross-Validation**",
                 expanded=st.session_state["stage4_df"] is not None and len(st.session_state["stage4_df"] or []) > 0):

    st.markdown('<div class="stage-num">05 / LLM CROSS-VALIDATION</div>', unsafe_allow_html=True)

    if st.session_state["stage4_df"] is None or len(st.session_state["stage4_df"]) == 0:
        st.markdown('<div class="lock-msg">⟳ Complete Stage 04 first.</div>', unsafe_allow_html=True)
    else:
        cand_df = st.session_state["stage4_df"]

        col_a, col_b = st.columns([2, 1])
        with col_a:
            n_validate = st.slider(
                "Relations to validate",
                min_value=1,
                max_value=min(30, len(cand_df)),
                value=min(5, len(cand_df)),
            )
        with col_b:
            # Prefer st.secrets (Streamlit Cloud), fall back to env vars, then manual entry
            _ant_default = (st.secrets.get("ANTHROPIC_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", ""))
            _oai_default = (st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""))
            anthropic_key = st.text_input("Anthropic API key", type="password",
                                           value=_ant_default,
                                           placeholder="sk-ant-…")
            openai_key    = st.text_input("OpenAI API key (optional)", type="password",
                                           value=_oai_default,
                                           placeholder="sk-… (skip to use Claude only)")

        run_s5 = st.button("▶ Run LLM Validation", key="run_s5", type="primary")

        if run_s5:
            if not anthropic_key:
                st.error("Anthropic API key required.")
            else:
                try:
                    import anthropic as _anthropic
                    from llm_validator import (
                        validate_with_claude, validate_with_gpt4o,
                        _add_corroboration_counts, _recency_score,
                        _model_valid_agreement, _model_strength_score,
                        _weighted_confidence, _rejection_confidence,
                        _assign_bucket, _mark_audit_sample,
                    )
                    from datetime import datetime

                    sample_df = cand_df.sample(n=n_validate, random_state=42).reset_index(drop=True)
                    sample_df = _add_corroboration_counts(sample_df)

                    claude_client = _anthropic.Anthropic(api_key=anthropic_key)
                    gpt_client    = None
                    if openai_key:
                        import openai as _openai
                        gpt_client = _openai.OpenAI(api_key=openai_key)

                    ref_date = datetime.now()
                    rows_out = []
                    progress = st.progress(0, text="Validating…")

                    for i, (_, row) in enumerate(sample_df.iterrows()):
                        progress.progress((i + 1) / len(sample_df),
                                          text=f"[{i+1}/{len(sample_df)}] {row.get('entity_a','')} → {row.get('entity_b','')}")

                        claude_res = validate_with_claude(row, claude_client)
                        gpt_res    = validate_with_gpt4o(row, gpt_client) if gpt_client else \
                                     {"valid": None, "evidence_strength": "none", "reasoning": "no key"}

                        from llm_validator import SOURCE_AUTHORITY, STRENGTH_MAP
                        src_auth = SOURCE_AUTHORITY.get(str(row.get("source_type","")).upper(), 0.5)
                        rec      = _recency_score(row.get("date",""), ref_date)
                        valid_ag, had_err = _model_valid_agreement(gpt_res["valid"], claude_res["valid"])
                        strength = _model_strength_score(gpt_res.get("evidence_strength"),
                                                         claude_res.get("evidence_strength"))
                        corr     = int(row.get("corroboration_count", 1))

                        final_conf  = _weighted_confidence(src_auth, corr, valid_ag, strength, rec)
                        reject_conf = _rejection_confidence(src_auth, corr, valid_ag, rec)

                        rows_out.append({
                            **row.to_dict(),
                            "gpt4o_valid": gpt_res["valid"],
                            "gpt4o_evidence_strength": gpt_res.get("evidence_strength"),
                            "gpt4o_reasoning": gpt_res.get("reasoning"),
                            "claude_valid": claude_res["valid"],
                            "claude_evidence_strength": claude_res.get("evidence_strength"),
                            "claude_reasoning": claude_res.get("reasoning"),
                            "source_authority_score": src_auth,
                            "recency_score": round(rec, 4),
                            "model_valid_agreement": round(valid_ag, 4),
                            "model_strength_score": round(strength, 4),
                            "model_had_error": had_err,
                            "final_confidence": final_conf,
                            "rejection_confidence": reject_conf,
                        })

                    progress.empty()
                    result_df = pd.DataFrame(rows_out)
                    result_df["decision_bucket"] = result_df.apply(_assign_bucket, axis=1)
                    result_df = _mark_audit_sample(result_df)
                    st.session_state["stage5_df"] = result_df

                except Exception as exc:
                    st.error(f"Stage 5 failed: {exc}")

        if st.session_state["stage5_df"] is not None:
            val_df = st.session_state["stage5_df"]

            accept = (val_df["decision_bucket"] == "auto_accept").sum()
            reject = (val_df["decision_bucket"] == "auto_reject").sum()
            review = (val_df["decision_bucket"] == "review").sum()
            audit  = val_df.get("audit_flag", pd.Series(dtype=bool)).sum()

            st.markdown(f"""
            <div class="metric-row">
              {metric_html(accept, "auto-accept")}
              {metric_html(reject, "auto-reject")}
              {metric_html(review, "review")}
              {metric_html(audit, "audit-flagged")}
            </div>
            """, unsafe_allow_html=True)

            # Per-row validation detail
            for _, row in val_df.iterrows():
                decision = row.get("decision_bucket", "review")
                conf     = row.get("final_confidence", 0.0)
                is_audit = bool(row.get("audit_flag", False))

                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        rel = str(row.get("candidate_relation",""))
                        st.markdown(
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;color:#e8f4ff;">'
                            f'{row.get("entity_a","?")} <span style="color:#4a9eff">──{rel}──▶</span> {row.get("entity_b","?")}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        c_r = textwrap.shorten(str(row.get("claude_reasoning","")), 120, placeholder="…")
                        g_r = textwrap.shorten(str(row.get("gpt4o_reasoning","")), 120, placeholder="…")
                        st.markdown(
                            f'<div style="font-size:0.75rem;color:#7aa8cc;margin-top:2px;">'
                            f'<b>Claude:</b> {c_r}</div>'
                            f'<div style="font-size:0.75rem;color:#7aa8cc;">'
                            f'<b>GPT-4o:</b> {g_r}</div>',
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            badge(decision) + (" &nbsp;" + '<span class="badge-audit">AUDIT</span>' if is_audit else "") +
                            f'<div style="font-size:0.72rem;color:#7aa8cc;margin-top:4px;">conf {conf:.3f}</div>' +
                            conf_bar(conf),
                            unsafe_allow_html=True,
                        )
                    st.markdown("<hr style='border-color:#1e3a5f;margin:8px 0'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Graph Visualization
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
with st.expander("**STAGE 06 — Knowledge Graph**",
                 expanded=st.session_state["stage5_df"] is not None):

    st.markdown('<div class="stage-num">06 / KNOWLEDGE GRAPH</div>', unsafe_allow_html=True)

    if st.session_state["stage5_df"] is None:
        st.markdown('<div class="lock-msg">⟳ Complete Stage 05 first.</div>', unsafe_allow_html=True)
    else:
        val_df = st.session_state["stage5_df"]
        accepted = val_df[val_df["decision_bucket"] == "auto_accept"]
        review   = val_df[val_df["decision_bucket"] == "review"]
        graph_df = pd.concat([accepted, review], ignore_index=True)

        min_conf_graph = st.slider("Min confidence to display", 0.0, 1.0, 0.5, 0.05)
        graph_df = graph_df[graph_df["final_confidence"] >= min_conf_graph]

        if len(graph_df) == 0:
            st.warning("No relations meet the confidence threshold. Lower the slider.")
        else:
            try:
                from pyvis.network import Network
                import streamlit.components.v1 as components

                EDGE_COLORS = {
                    "ACQUIRED":          "#ff6b6b",
                    "LICENSED_FROM":     "#ffc107",
                    "LICENSED_TO":       "#ffc107",
                    "PARTNERED_WITH":    "#4a9eff",
                    "COLLABORATED_WITH": "#a78bfa",
                    "FILED_PATENT_WITH": "#3ddc84",
                    "FUNDED_WITH":       "#fb923c",
                }

                net = Network(height="560px", width="100%", bgcolor="#0a0f1e",
                              font_color="#e8f4ff", directed=True)
                net.set_options(json.dumps({
                    "physics": {"solver": "forceAtlas2Based",
                                "forceAtlas2Based": {"gravitationalConstant": -80,
                                                      "centralGravity": 0.01,
                                                      "springLength": 160,
                                                      "springConstant": 0.08},
                                "stabilization": {"iterations": 150}},
                    "edges": {"smooth": {"type": "dynamic"}, "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}}},
                    "nodes": {"borderWidth": 2, "shadow": True},
                    "interaction": {"hover": True, "tooltipDelay": 100},
                }))

                node_freq: dict[str, int] = {}
                for _, row in graph_df.iterrows():
                    a, b = str(row.get("entity_a","")), str(row.get("entity_b",""))
                    node_freq[a] = node_freq.get(a, 0) + 1
                    node_freq[b] = node_freq.get(b, 0) + 1

                for name, freq in node_freq.items():
                    size = min(14 + freq * 4, 36)
                    net.add_node(name, label=name, title=name, size=size,
                                 color={"background": "#1e3a5f", "border": "#4a9eff",
                                        "highlight": {"background": "#2a5080", "border": "#7ac2ff"}})

                for _, row in graph_df.iterrows():
                    a   = str(row.get("entity_a",""))
                    b   = str(row.get("entity_b",""))
                    rel = str(row.get("candidate_relation",""))
                    conf = float(row.get("final_confidence", 0.5))
                    color = EDGE_COLORS.get(rel, "#7aa8cc")
                    ev   = textwrap.shorten(str(row.get("evidence_text","")), 100, placeholder="…")
                    tip  = f"{a} → {rel} → {b}\nconf: {conf:.3f}\n{ev}"
                    net.add_edge(a, b, label=rel, title=tip, color=color,
                                 width=1.5 + conf * 2.5, font={"size": 9, "color": color})

                html_path = "/tmp/kg_graph.html"
                net.save_graph(html_path)
                with open(html_path) as f:
                    html_content = f.read()
                components.html(html_content, height=570, scrolling=False)

                # Legend
                legend_items = "".join(
                    f'<span style="display:inline-flex;align-items:center;gap:5px;margin:4px 8px 4px 0;">'
                    f'<span style="width:18px;height:3px;background:{c};display:inline-block;border-radius:2px;"></span>'
                    f'<span style="font-size:0.72rem;color:#7aa8cc;">{r}</span></span>'
                    for r, c in EDGE_COLORS.items()
                )
                st.markdown(f'<div style="margin-top:8px;">{legend_items}</div>', unsafe_allow_html=True)

            except ImportError:
                st.warning("pyvis not installed. Run: `pip install pyvis`")
                # Fallback: table view
                show_cols = [c for c in ["entity_a", "candidate_relation", "entity_b",
                                          "final_confidence", "decision_bucket"] if c in graph_df.columns]
                st.dataframe(graph_df[show_cols], use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem 0;border-top:1px solid #1e3a5f;
            text-align:center;font-size:0.75rem;color:#4a6680;
            font-family:'IBM Plex Mono',monospace;">
  CSL Behring × Cornell Tech · Knowledge Graph Pipeline Demo
</div>
""", unsafe_allow_html=True)
