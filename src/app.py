"""Streamlit Web Application: Compare Research Papers.

Modes:
- Compare Papers: Side-by-side comparison across 6 criteria (2-4 papers)
- Ask Questions: Question-answering using content from uploaded papers

State Machine:
- idle: No analysis running, controls enabled
- analyzing: Analysis in progress, all controls locked
- complete: Analysis finished, results displayed
- error: Analysis failed, error shown
"""

import os
import sys
import re
import json
import streamlit as st

from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.ingest import (
    validate_pdf_uploads,
    process_pdf_document,
    cleanup_session_files,
)
from src.rag.embed import VectorIndex
from src.rag.retrieve import COMPARISON_TOPICS, format_chunks_for_prompt
from src.rag.extract import (
    get_llm_client,
    evaluate_topic,
    generate_executive_summary,
)

load_dotenv()

st.set_page_config(
    page_title="PaperDiff | Compare Research Papers",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

STYLING_CSS = """
<style>
    #MainMenu {visibility: hidden; display: none !important;}
    footer {visibility: hidden; display: none !important;}
    header {visibility: hidden; display: none !important;}
    .stAppDeployButton {display:none !important;}

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    .section-gap { margin-top: 1.8rem; }

    .verdict-hero {
        text-align: center;
        padding: 1.0rem 1.2rem;
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(16,185,129,0.08));
        border: 1px solid rgba(128,128,128,0.2);
        margin-bottom: 0.6rem;
    }
    .verdict-hero .verdict-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.2rem;
    }
    .verdict-hero .verdict-text {
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .verdict-hero .verdict-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    .meta-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        padding: 0.5rem 0;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .meta-strip span { white-space: nowrap; }

    .elicit-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.6rem;
        margin-bottom: 1.0rem;
        font-size: 0.88rem;
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }
    .elicit-table th {
        background-color: rgba(128, 128, 128, 0.10);
        padding: 8px 12px;
        text-align: left;
        border: 1px solid rgba(128, 128, 128, 0.25);
        font-weight: 700;
        font-size: 0.85rem;
    }
    .elicit-table td {
        padding: 10px 12px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        vertical-align: top;
        line-height: 1.45;
    }
    .elicit-table tr:hover {
        background-color: rgba(128, 128, 128, 0.04);
    }
    .elicit-table tr:nth-child(even) {
        background-color: rgba(128, 128, 128, 0.02);
    }

    .badge-pill-a, .badge-pill-b, .badge-pill-c, .badge-pill-d {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-pill-a {
        background-color: rgba(59, 130, 246, 0.18);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-pill-b {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-pill-c {
        background-color: rgba(234, 179, 8, 0.18);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.4);
    }
    .badge-pill-d {
        background-color: rgba(168, 85, 247, 0.18);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.4);
    }

    .adv-pill {
        display: inline-block;
        font-size: 0.80rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 4px;
        background-color: rgba(234, 179, 8, 0.18);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.4);
    }

    .cite-pill {
        display: inline-block;
        font-size: 0.72rem;
        padding: 1px 5px;
        border-radius: 3px;
        background-color: rgba(128, 128, 128, 0.12);
        color: #64748b;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-top: 4px;
    }

    .not-assessable {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 700;
        background-color: rgba(128, 128, 128, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(128, 128, 128, 0.35);
    }

    .dim-card {
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border-radius: 6px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.02);
    }
</style>
"""
st.markdown(STYLING_CSS, unsafe_allow_html=True)

AVAILABLE_MODELS = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m3:free",
    "google/gemini-2.5-flash"
]

PAPER_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#a855f7"]
PAPER_IDS = ["Paper A", "Paper B", "Paper C", "Paper D"]
BADGE_CLASSES = ["badge-pill-a", "badge-pill-b", "badge-pill-c", "badge-pill-d"]


def init_state():
    """Initialize session state with proper state machine variables."""
    defaults = {
        "analysis_state": "idle",
        "analysis_mode": None,
        "used_model": None,
        "locked_model": None,
        "error_message": None,
        "results": {},
        "executive_summary": None,
        "academic_strength": None,
        "topic_strength": None,
        "papers": [],
        "selected_model": "openai/gpt-4o-mini",
        "show_uploader": True,
        "rag_query": "",
        "rag_answer": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


def is_controls_locked() -> bool:
    """Check if controls should be locked (analysis in progress)."""
    return st.session_state.analysis_state == "analyzing"


def render_elicit_grid_table(results: dict, papers: list) -> str:
    """Generate HTML for structured comparison grid table with N papers."""
    paper_ids = [p["id"] for p in papers]

    # Build header
    header_cells = "<th style='width: 14%;'>Criteria</th>"
    col_width = max(18, int(86 / len(paper_ids)))
    for p in papers:
        header_cells += f"<th style='width: {col_width}%;'>{'🔵🟢🟡🟣'[papers.index(p) % 4]} {p['id']} ({p['meta'].title[:20]}...)</th>"
    header_cells += "<th style='width: 18%;'>Advantage</th>"

    rows = []
    for key, res in results.items():
        # Dimension cell
        row = (
            f"<td style='width: 14%; font-weight: 600;'>"
            f"<div style='font-size: 1.3rem; margin-bottom: 4px;'>{res.topic_icon}</div>"
            f"<div><strong>{res.topic_name}</strong></div>"
            f"</td>"
        )

        # Paper cells
        for pid in paper_ids:
            assess = res.assessments.get(pid)
            if assess and assess.status == "not_assessable":
                score_html = "<span class='not-assessable'>Not Assessable</span>"
            elif assess:
                color = PAPER_COLORS[paper_ids.index(pid) % len(PAPER_COLORS)]
                score_html = f"<span style='color: {color}; font-weight: 700;'>⭐ {assess.rating:.1f}/5.0</span>"
            else:
                score_html = "<span class='not-assessable'>N/A</span>"

            take = assess.key_takeaway if assess else ""
            expl = assess.explanation if assess else ""
            cite = f"<div class='cite-pill'>📍 {assess.page_reference}</div>" if assess and assess.page_reference else ""

            row += (
                f"<td style='width: {col_width}%;'>"
                f"<div>{score_html}</div>"
                f"<div style='margin-top: 6px; font-weight: 600;'>{take}</div>"
                f"<div style='margin-top: 4px; font-size: 0.88rem; opacity: 0.9;'>{expl}</div>"
                f"{cite}"
                f"</td>"
            )

        # Advantage cell
        row += (
            f"<td style='width: 18%;'>"
            f"<div><span class='adv-pill'>🏷️ {res.better_paper}</span></div>"
            f"<div style='margin-top: 6px; font-weight: 600; color: #facc15;'>{res.one_sentence_verdict}</div>"
            f"<div style='margin-top: 4px; font-size: 0.88rem; opacity: 0.9;'>{res.signification}</div>"
            f"</td>"
        )
        rows.append(f"<tr>{row}</tr>")

    table_html = (
        f"<table class='elicit-table'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
    )
    return table_html


def ensure_papers_indexed():
    """Parse and index papers if not already done. Returns list of paper dicts."""
    if st.session_state.papers and all(p.get("index") is not None for p in st.session_state.papers):
        return st.session_state.papers

    papers = []
    for paper_spec in st.session_state.papers:
        pid = paper_spec["id"]
        raw_bytes = paper_spec["raw_bytes"]
        filename = paper_spec["filename"]

        meta, chunks = process_pdf_document(raw_bytes, filename, paper_id=pid)
        index = VectorIndex(pid, meta.title)
        index.add_chunks(chunks)

        papers.append({
            "id": pid,
            "filename": filename,
            "raw_bytes": raw_bytes,
            "meta": meta,
            "index": index,
        })

    st.session_state.papers = papers
    return papers


def _compute_strength_scores(results: dict, papers: list) -> dict:
    """Compute Research Quality and Topic Fit scores from results across N papers.

    Research Quality: overall research quality from assessable criteria.
    Topic Fit: how directly each paper covers the shared topic.
    """
    paper_ids = [p["id"] for p in papers]
    n = len(paper_ids)

    # Collect scores per paper
    assessable_counts = {pid: 0 for pid in paper_ids}
    assessable_sums = {pid: 0.0 for pid in paper_ids}
    topic_sums = {pid: 0.0 for pid in paper_ids}

    topic_method_keys = ["methodology_rigor", "evidence_findings", "contribution_novelty"]

    for key, res in results.items():
        for pid in paper_ids:
            assess = res.assessments.get(pid)
            if assess and assess.status == "assessed":
                assessable_counts[pid] += 1
                assessable_sums[pid] += assess.rating
            raw_score = assess.rating if (assess and assess.status == "assessed") else 0
            topic_sums[pid] += raw_score

    # Academic averages
    academic_avgs = {}
    for pid in paper_ids:
        academic_avgs[pid] = round(assessable_sums[pid] / assessable_counts[pid], 2) if assessable_counts[pid] > 0 else 0

    # Topic averages (methodology + evidence + contribution)
    topic_counts = sum(1 for k in topic_method_keys if k in results)
    topic_avgs = {}
    for pid in paper_ids:
        topic_avgs[pid] = round(topic_sums[pid] / max(topic_counts, 1), 2)

    # Determine winners
    sorted_academic = sorted(paper_ids, key=lambda p: academic_avgs[p], reverse=True)
    if n >= 2 and academic_avgs[sorted_academic[0]] > academic_avgs[sorted_academic[1]] + 0.3:
        academic_winner = sorted_academic[0]
    else:
        academic_winner = "No clear winner"

    sorted_topic = sorted(paper_ids, key=lambda p: topic_avgs[p], reverse=True)
    if n >= 2 and topic_avgs[sorted_topic[0]] > topic_avgs[sorted_topic[1]] + 0.3:
        topic_winner = sorted_topic[0]
    else:
        topic_winner = "No clear winner"

    return {
        "academic": {
            "averages": academic_avgs,
            "winner": academic_winner,
            "assessable_counts": assessable_counts,
        },
        "topic": {
            "averages": topic_avgs,
            "winner": topic_winner,
        },
    }


def start_analysis(model_name: str, mode: str = "comparative"):
    """Phase 1: Lock state and rerun to prevent UI interaction during analysis."""
    st.session_state.analysis_state = "analyzing"
    st.session_state.analysis_mode = mode
    st.session_state.locked_model = model_name
    st.session_state.error_message = None
    st.rerun()


def execute_comparative_analysis():
    """Phase 2: Run the actual analysis (called only when state is 'analyzing')."""
    model_name = st.session_state.locked_model
    papers = st.session_state.papers
    n = len(papers)
    total_stages = 7

    stage_labels = [
        "Checking research methods",
        "Checking evidence quality",
        "Checking originality",
        "Checking readability",
        "Checking references",
        "Checking ethics",
    ]

    try:
        with st.status(f"Comparing {n} papers", expanded=True) as status:
            for p in papers:
                st.caption(f"  {p['id']}: {p['filename']}")
            st.caption(f"  Model: {model_name}")

            st.progress(0, text=f"0 of {total_stages} stages complete")

            papers_indexed = ensure_papers_indexed()
            client = get_llm_client()
            results = {}
            topics = list(COMPARISON_TOPICS.items())

            for idx, (topic_key, topic_def) in enumerate(topics):
                status.update(label=f"Comparing papers — {stage_labels[idx]} ({idx + 1}/{total_stages})")
                res = evaluate_topic(
                    topic=topic_def,
                    papers=papers_indexed,
                    client=client,
                    model_name=model_name,
                )
                results[topic_key] = res
                st.progress((idx + 1) / total_stages, text=f"{idx + 1} of {total_stages} stages complete")

            status.update(label="Comparing papers — Preparing results")
            st.progress(6 / 7, text=f"6 of {total_stages} stages complete")

            papers_meta = {p["id"]: p["meta"] for p in papers_indexed}
            exec_summary = generate_executive_summary(
                papers_meta, results, client=client, model_name=model_name
            )
            strength = _compute_strength_scores(results, papers_indexed)

            st.progress(1.0, text=f"{total_stages} of {total_stages} stages complete")
            status.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state.results = results
        st.session_state.executive_summary = exec_summary
        st.session_state.academic_strength = strength["academic"]
        st.session_state.topic_strength = strength["topic"]
        st.session_state.used_model = model_name
        st.session_state.analysis_state = "complete"
        st.session_state.show_uploader = False

    except Exception as e:
        st.session_state.analysis_state = "error"
        st.session_state.error_message = str(e)

    st.rerun()


def execute_rag_query():
    """Execute RAG query against uploaded documents with evidence traceability."""
    model_name = st.session_state.locked_model
    query = st.session_state.get("rag_query", "")
    papers = st.session_state.papers
    n = len(papers)
    total_stages = 3

    try:
        papers_indexed = ensure_papers_indexed()
        client = get_llm_client()

        with st.status(f"Searching {n} papers", expanded=True) as status:
            for p in papers_indexed:
                st.caption(f"  {p['id']}: {p['filename']}")
            st.caption(f"  Model: {model_name}")

            st.progress(0, text="0 of 3 stages complete")

            # Stage 1: Retrieve relevant chunks using the actual query
            status.update(label="Finding relevant passages (1/3)")
            context_parts = []
            for p in papers_indexed:
                meta = p["meta"]
                index = p["index"]
                # Use the user's query directly for retrieval
                matches = index.search(query, top_k=6)
                context = format_chunks_for_prompt(matches, meta.title)
                context_parts.append(f"=== {p['id']}: {meta.title} ===\n{context}")
            combined_context = "\n\n".join(context_parts)
            st.progress(1 / 3, text="1 of 3 stages complete")

            # Stage 2: Generate answer
            status.update(label="Generating answer (2/3)")

            if client is None:
                answer = (
                    "To use this feature, please add an API key to your `.env` file. "
                    "Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY`."
                )
                sources = []
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a research assistant. Answer questions using ONLY the provided document excerpts.\n\n"
                                "Rules:\n"
                                "- Rewrite evidence into clear, structured sentences. Do NOT copy raw chunks verbatim.\n"
                                "- Every substantive claim must include its source (Paper, Page, Section).\n"
                                "- If the documents do not contain enough evidence, respond with: 'Insufficient evidence in the uploaded documents.'\n"
                                "- Do NOT invent findings or infer missing information as fact.\n"
                                "- Do NOT add facts not present in the retrieved evidence.\n\n"
                                "Return JSON:\n"
                                '{"answer": "Your rewritten answer...", "sources": [{"paper": "Paper A", "page": "7", "section": "Methods", "quote": "relevant quote..."}]}'
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Document Excerpts:\n{combined_context}\n\nQuestion: {query}",
                        },
                    ],
                    temperature=0.2,
                    max_tokens=800,
                )
                raw_response = getattr(response.choices[0].message, "content", "") or "{}"
                try:
                    clean = raw_response.strip()
                    clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
                    clean = re.sub(r"\n?```$", "", clean).strip()
                    result_data = json.loads(clean)
                    answer = result_data.get("answer", "No response generated.")
                    sources = result_data.get("sources", [])
                except (json.JSONDecodeError, KeyError):
                    answer = raw_response
                    sources = []

            st.progress(2 / 3, text="2 of 3 stages complete")
            st.progress(1.0, text="3 of 3 stages complete")
            status.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state.rag_answer = {
            "question": query,
            "answer": answer,
            "sources": sources,
            "model": model_name,
        }
        st.session_state.used_model = model_name
        st.session_state.analysis_state = "complete"

    except Exception as e:
        st.session_state.analysis_state = "error"
        st.session_state.error_message = str(e)

    st.rerun()


# ==========================================
# Header
# ==========================================
col_hdr, col_actions = st.columns([3, 1])

with col_hdr:
    st.title("PaperDiff")
    st.caption("Compare research papers side-by-side. Supports 2-4 papers.")

with col_actions:
    if st.session_state.analysis_state == "complete" and st.session_state.analysis_mode == "comparative":
        if st.button(
            "Upload Different Papers",
            use_container_width=True,
            disabled=is_controls_locked(),
        ):
            st.session_state.show_uploader = not st.session_state.show_uploader
            st.rerun()

st.markdown("---")


# ==========================================
# ERROR STATE DISPLAY
# ==========================================
if st.session_state.analysis_state == "error":
    st.error("Analysis could not be completed.")
    if st.session_state.error_message:
        st.caption(st.session_state.error_message)
    col_retry, col_dismiss = st.columns([1, 1])
    with col_retry:
        if st.button("Try Again", type="primary", use_container_width=True):
            st.session_state.analysis_state = "idle"
            st.session_state.error_message = None
            st.rerun()
    with col_dismiss:
        if st.button("Dismiss", use_container_width=True):
            st.session_state.analysis_state = "idle"
            st.session_state.error_message = None
            st.rerun()
    st.stop()


# ==========================================
# ANALYZING STATE
# ==========================================
if st.session_state.analysis_state == "analyzing":
    if st.session_state.analysis_mode == "comparative":
        execute_comparative_analysis()
    elif st.session_state.analysis_mode == "rag":
        execute_rag_query()
    else:
        st.info("Analysis in progress. Controls are locked.")
        st.stop()


# ==========================================
# UPLOAD SECTION (idle state only, or when viewing results and toggling)
# ==========================================
show_upload = (
    st.session_state.analysis_state == "idle"
    or (st.session_state.show_uploader and st.session_state.analysis_state == "complete")
)

if show_upload:
    paper_count = len(st.session_state.papers)
    locked = is_controls_locked()

    # ---- Section: Research Papers ----
    st.markdown("### Research Papers")
    st.caption(f"{paper_count} of 4 papers uploaded")

    # Show uploaded papers as compact rows
    for p in st.session_state.papers:
        with st.container():
            c_info, c_action = st.columns([5, 1])
            with c_info:
                st.markdown(f"**{p['id']}** — `{p['filename']}`")
            with c_action:
                if not locked:
                    if st.button("Remove", key=f"remove_{p['id']}", use_container_width=True):
                        st.session_state.papers = [x for x in st.session_state.papers if x["id"] != p["id"]]
                        st.rerun()

    # Upload new paper slot
    if paper_count < 4 and not locked:
        used_ids = {p["id"] for p in st.session_state.papers}
        next_pid = next(pid for pid in PAPER_IDS if pid not in used_ids)
        new_file = st.file_uploader(
            f"Add {next_pid} (PDF)",
            type=["pdf"],
            key=f"file_uploader_{paper_count}",
            label_visibility="collapsed",
        )
        if new_file:
            already = any(x["id"] == next_pid for x in st.session_state.papers)
            if not already:
                st.session_state.papers.append({
                    "id": next_pid,
                    "filename": new_file.name,
                    "raw_bytes": new_file.getvalue(),
                    "meta": None,
                    "index": None,
                })
                st.rerun()

    # Quick demo
    if paper_count == 0 and not locked:
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _papers_dir = os.path.join(_project_root, "data", "papers")
        has_a = os.path.exists(os.path.join(_papers_dir, "Lewis_et_al_2020_RAG.pdf"))
        has_b = os.path.exists(os.path.join(_papers_dir, "Karpukhin_et_al_2020_DPR.pdf"))
        if has_a and has_b:
            if st.button("Try Demo (RAG vs DPR)"):
                with open(os.path.join(_papers_dir, "Lewis_et_al_2020_RAG.pdf"), "rb") as f:
                    bytes_a = f.read()
                with open(os.path.join(_papers_dir, "Karpukhin_et_al_2020_DPR.pdf"), "rb") as f:
                    bytes_b = f.read()
                st.session_state.papers = [
                    {"id": "Paper A", "filename": "Lewis_et_al_2020_RAG.pdf", "raw_bytes": bytes_a, "meta": None, "index": None},
                    {"id": "Paper B", "filename": "Karpukhin_et_al_2020_DPR.pdf", "raw_bytes": bytes_b, "meta": None, "index": None},
                ]
                st.session_state.analysis_state = "idle"
                st.rerun()

    st.markdown("---")

    # ---- Section: Analysis Mode ----
    st.markdown("### Analysis Mode")

    mode_options = ["Compare Papers", "Ask Questions"]
    mode_labels = [
        "Compare your papers across 6 criteria.",
        "Ask questions about your papers.",
    ]

    current_mode_index = 0
    if st.session_state.analysis_mode == "rag":
        current_mode_index = 1

    selected_mode_index = st.radio(
        "Choose analysis mode",
        options=range(len(mode_options)),
        format_func=lambda i: f"**{mode_options[i]}** — {mode_labels[i]}",
        index=current_mode_index,
        disabled=locked,
        horizontal=False,
        label_visibility="collapsed",
    )
    selected_mode = "comparative" if selected_mode_index == 0 else "rag"
    if not locked:
        st.session_state.analysis_mode = selected_mode

    st.markdown("---")

    # ---- Section: Analysis Model ----
    st.markdown("### Analysis Model")
    st.caption("This model reads your papers and generates the comparison.")

    chosen_model = st.selectbox(
        "Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.selected_model)
        if st.session_state.selected_model in AVAILABLE_MODELS
        else 0,
        disabled=locked,
        label_visibility="collapsed",
    )
    st.session_state.selected_model = chosen_model

    st.markdown("---")

    # ---- Section: Summary + Run ----
    # Validate uploads
    if paper_count >= 2:
        upload_specs = [(p["filename"], len(p["raw_bytes"])) for p in st.session_state.papers]
        is_valid, validation_msg = validate_pdf_uploads(upload_specs)
    else:
        is_valid = False
        validation_msg = ""

    if not is_valid and paper_count >= 2:
        st.error(f"⚠️ {validation_msg}")

    # Pre-analysis summary
    can_run = False
    if paper_count >= 2 and is_valid:
        can_run = True
    if selected_mode == "rag" and paper_count >= 1:
        can_run = True
    if selected_mode == "comparative" and paper_count < 2:
        can_run = False

    if paper_count >= 1:
        mode_label = "Compare Papers" if selected_mode == "comparative" else "Ask Questions"
        st.markdown(
            f"**Papers:** {paper_count}  ·  "
            f"**Mode:** {mode_label}  ·  "
            f"**Model:** {chosen_model}"
        )

    # Run button
    if selected_mode == "comparative":
        btn_label = "Run Comparison"
        btn_type = "primary"

        if st.button(
            btn_label,
            type=btn_type,
            use_container_width=True,
            disabled=locked or not can_run,
        ):
            start_analysis(chosen_model, selected_mode)

        if not can_run and paper_count < 2:
            st.caption("Need at least 2 papers to compare.")

    if st.session_state.analysis_state == "idle" and paper_count < 1:
        st.stop()


# ==========================================
# RAG ANALYSIS MODE
# ==========================================
if (
    st.session_state.analysis_state in ("idle", "complete")
    and st.session_state.analysis_mode == "rag"
    and st.session_state.papers
):
    # Show papers
    paper_parts = []
    for p in st.session_state.papers:
        meta = p.get("meta")
        title = meta.title[:35] + "..." if meta else p["filename"]
        paper_parts.append(f"{p['id']}: {title}")
    st.caption(" &middot; ".join(paper_parts))

    st.markdown("### Ask Questions")
    st.caption("Get answers based only on content from your papers.")

    col_query, col_btn = st.columns([3, 1])
    with col_query:
        rag_query = st.text_input(
            "Ask a question about the papers",
            placeholder="e.g., How do the methodologies compare in terms of sample size?",
            disabled=is_controls_locked(),
            key="rag_query_input",
        )
    with col_btn:
        st.write("")
        st.write("")

        if not is_controls_locked():
            st.selectbox(
                "Model",
                options=AVAILABLE_MODELS,
                index=AVAILABLE_MODELS.index(st.session_state.selected_model)
                if st.session_state.selected_model in AVAILABLE_MODELS
                else 0,
                disabled=is_controls_locked(),
                key="rag_model_select",
            )

        if st.button(
            "Ask",
            type="primary",
            use_container_width=True,
            disabled=is_controls_locked() or not rag_query.strip(),
        ):
            st.session_state.rag_query = rag_query.strip()
            start_analysis(
                st.session_state.get("rag_model_select", st.session_state.selected_model),
                "rag",
            )

    if st.session_state.rag_answer and st.session_state.analysis_state == "complete":
        answer_data = st.session_state.rag_answer
        with st.container():
            question_display = (
                answer_data.get('question', '')
                or st.session_state.get('rag_query', '')
            )
            if question_display:
                st.markdown(f"**Q:** {question_display}")
            st.markdown(answer_data["answer"])

            sources = answer_data.get("sources", [])
            if sources:
                for i, src in enumerate(sources, 1):
                    paper = src.get("paper", "?")
                    page = src.get("page", "?")
                    section = src.get("section", "")
                    quote = src.get("quote", "")
                    location = f"{paper}, Page {page}"
                    if section:
                        location += f" | {section}"
                    st.caption(f"{i}. {location}")
                    if quote:
                        with st.expander(f"View excerpt ({paper}, p.{page})"):
                            st.caption(f'"{quote}"')

            st.caption(f"Model: `{answer_data['model']}`")

    st.markdown("---")
    col_foot_l, col_foot_r = st.columns([3, 1])
    with col_foot_r:
        if st.button("Start Over", use_container_width=True, disabled=is_controls_locked(), key="rag_reset"):
            _project_root_reset = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cleanup_session_files(os.path.join(_project_root_reset, "data", "papers"))
            st.session_state.clear()
            init_state()
            st.rerun()

    st.stop()


# ==========================================
# COMPARATIVE ANALYSIS - RESULTS DASHBOARD
# ==========================================
if (
    st.session_state.analysis_state == "complete"
    and st.session_state.analysis_mode == "comparative"
    and st.session_state.results
):
    papers = st.session_state.papers
    paper_ids = [p["id"] for p in papers]
    n = len(papers)
    results = st.session_state.results
    exec_sum = st.session_state.executive_summary
    used_model = st.session_state.used_model
    academic = st.session_state.academic_strength
    topic = st.session_state.topic_strength

    # ---- Compute overall verdict and confidence ----
    assessed_count = sum(
        1 for r in results.values()
        if any(r.assessments.get(pid, type("", (), {"status": ""})()).status == "assessed" for pid in paper_ids)
    )
    all_na_count = sum(
        1 for r in results.values()
        if all(
            r.assessments.get(pid, type("", (), {"status": "not_assessable"})()).status == "not_assessable"
            for pid in paper_ids
        )
    )

    if academic and topic:
        acad_w = academic.get("winner", "No clear winner")
        topic_w = topic.get("winner", "No clear winner")

        if acad_w == topic_w and acad_w != "No clear winner":
            overall = acad_w
        elif acad_w == "No clear winner" and topic_w == "No clear winner":
            overall = "No clear winner"
        elif acad_w != "No clear winner" and topic_w != "No clear winner" and acad_w != topic_w:
            overall = "Different strengths"
        else:
            winner = acad_w if acad_w != "No clear winner" else topic_w
            overall = winner
    else:
        overall = "No clear winner"

    if assessed_count >= 5:
        confidence = "High"
    elif assessed_count >= 3:
        confidence = "Medium"
    elif assessed_count >= 1:
        confidence = "Low"
    else:
        confidence = "Insufficient Evidence"

    # ---- Paper info strip (compact) ----
    paper_parts = []
    for p in papers:
        paper_parts.append(f"{p['id']}: {p['meta'].title[:35]}...")
    model_part = f"Model: {used_model}"
    st.markdown(
        f"<div class='meta-strip'>"
        + " &middot; ".join(f"<span>{part}</span>" for part in paper_parts + [model_part])
        + "</div>",
        unsafe_allow_html=True,
    )

    # ---- Re-run bar ----
    reeval_cols = st.columns([3, 1])
    new_model = used_model
    with reeval_cols[0]:
        if is_controls_locked():
            st.caption(f"Using model: `{st.session_state.locked_model or used_model}`")
        else:
            cur_idx = AVAILABLE_MODELS.index(used_model) if used_model in AVAILABLE_MODELS else 0
            new_model = st.selectbox(
                "Try a different model",
                options=AVAILABLE_MODELS,
                index=cur_idx,
                label_visibility="collapsed",
                disabled=is_controls_locked(),
                key="reeval_model_select",
            )
    with reeval_cols[1]:
        reeval_disabled = is_controls_locked()
        if st.button(
            "Re-run",
            type="secondary",
            use_container_width=True,
            disabled=reeval_disabled,
        ):
            if not reeval_disabled:
                start_analysis(new_model, "comparative")

    # ==========================================================
    # 1. OVERALL RESULT (Hero — visible immediately)
    # ==========================================================
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # Hero verdict
    if overall == "No clear winner":
        verdict_color = "#94a3b8"
    elif "Different" in overall:
        verdict_color = "#facc15"
    else:
        # Match winner paper to its color
        verdict_color = "#94a3b8"
        for i, pid in enumerate(paper_ids):
            if pid in overall:
                verdict_color = PAPER_COLORS[i % len(PAPER_COLORS)]
                break

    st.markdown(
        f"<div class='verdict-hero'>"
        f"<div class='verdict-label'>Overall Winner</div>"
        f"<div class='verdict-text' style='color:{verdict_color}'>{overall}</div>"
        f"<div class='verdict-sub'>Confidence: {confidence} &middot; {assessed_count} of 6 criteria assessed</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Sub-metrics row
    metric_cols = st.columns(min(n + 2, 6))
    for i, pid in enumerate(paper_ids):
        with metric_cols[i]:
            if academic:
                a_avg = academic["averages"].get(pid, 0)
                st.metric(f"{pid}", f"{a_avg:.1f}", label_visibility="visible")
    with metric_cols[min(n, len(metric_cols) - 2)]:
        acad_label = academic.get("winner", "Tie") if academic else "Tie"
        st.metric("Research Quality", acad_label)
    with metric_cols[min(n + 1, len(metric_cols) - 1)]:
        topic_label = topic.get("winner", "Tie") if topic else "Tie"
        st.metric("Topic Fit", topic_label)

    # ==========================================================
    # 2. WHY? (Evidence-based reasons)
    # ==========================================================
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### Why")

    reasons = []

    if academic:
        acad_w = academic["winner"]
        acad_avgs = academic["averages"]
        if acad_w != "No clear winner":
            sorted_pids = sorted(paper_ids, key=lambda p: acad_avgs.get(p, 0), reverse=True)
            reasons.append(
                f"**{sorted_pids[0]}** has the highest research quality score ({acad_avgs[sorted_pids[0]]:.1f}/5.0)."
            )
        else:
            avg_str = ", ".join(f"{pid}: {acad_avgs.get(pid, 0):.1f}" for pid in paper_ids)
            reasons.append(f"Research quality is similar across papers ({avg_str}).")

    if topic:
        topic_w = topic["winner"]
        if topic_w != "No clear winner":
            reasons.append(f"**{topic_w}** covers the topic more directly.")
        else:
            reasons.append("All papers cover the topic with similar depth.")

    if all_na_count > 0:
        reasons.append(f"{all_na_count} topic(s) could not be checked — not enough evidence in the papers.")

    if overall == "Different strengths":
        reasons.append("The papers have different strengths in research quality and topic coverage.")

    if not reasons:
        reasons.append("Limited evidence. See the detailed analysis below.")

    for reason in reasons[:4]:
        st.markdown(f"{reason}")

    # ==========================================================
    # 3. SCORE SUMMARY (Compact)
    # ==========================================================
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### Scores")

    score_header = "<th>Criteria</th>" + "".join(f"<th>{pid}</th>" for pid in paper_ids) + "<th>Stronger</th>"
    score_rows = []
    for key, res in results.items():
        cells = [f"<td style='font-weight: 600;'>{res.topic_icon} {res.topic_name}</td>"]
        for pid in paper_ids:
            assess = res.assessments.get(pid)
            if assess and assess.status == "not_assessable":
                cells.append("<td style='color:#64748b;font-size:0.82rem;'>N/A</td>")
            elif assess:
                color = PAPER_COLORS[paper_ids.index(pid) % len(PAPER_COLORS)]
                cells.append(f"<td style='color: {color}; font-weight: 700;'>{assess.rating:.1f}</td>")
            else:
                cells.append("<td style='color:#64748b;'>-</td>")

        bp = res.better_paper
        stronger_cell = "<td style='color:#64748b;'>-</td>"
        for pid in paper_ids:
            if pid in bp and "Advantage" in bp:
                color = PAPER_COLORS[paper_ids.index(pid) % len(PAPER_COLORS)]
                stronger_cell = f"<td style='color: {color}; font-weight: 700;'>{pid}</td>"
                break
        cells.append(stronger_cell)
        score_rows.append(f"<tr>{''.join(cells)}</tr>")

    score_table_html = f"<table class='elicit-table'><thead><tr>{score_header}</tr></thead><tbody>{''.join(score_rows)}</tbody></table>"
    st.markdown(score_table_html, unsafe_allow_html=True)

    # ==========================================================
    # 4. KEY DIFFERENCES
    # ==========================================================
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### Main Differences")

    key_diffs = []
    for key, res in results.items():
        if res.tradeoff_summary:
            key_diffs.append(f"**{res.topic_name}:** {res.tradeoff_summary}")
        elif res.one_sentence_verdict:
            key_diffs.append(f"**{res.topic_name}:** {res.one_sentence_verdict}")

    if not key_diffs and exec_sum and exec_sum.top_differentiators:
        key_diffs = [f"{d}" for d in exec_sum.top_differentiators[:5]]

    for diff in key_diffs[:5]:
        st.markdown(f"{diff}")

    if not key_diffs:
        st.caption("No major differences identified beyond the detailed analysis below.")

    st.markdown("---")

    # ==========================================================
    # 5. DETAILED ANALYSIS
    # ==========================================================
    tab_detailed, tab_matrix, tab_agreements, tab_sources = st.tabs([
        "Detailed Analysis",
        "Comparison Matrix",
        "Agreed Points & Differences",
        "Sources",
    ])

    with tab_detailed:
        assessed_results = []
        not_assessed_results = []
        for key, res in results.items():
            all_na = all(
                res.assessments.get(pid, type("", (), {"status": "not_assessable"})()).status == "not_assessable"
                for pid in paper_ids
            )
            if all_na:
                not_assessed_results.append(res)
            else:
                assessed_results.append(res)

        if assessed_results:
            st.markdown("**What Was Found**")
            for res in assessed_results:
                bp = res.better_paper
                if "Advantage" in bp:
                    winner_pid = bp.replace(" Advantage", "")
                    idx = paper_ids.index(winner_pid) if winner_pid in paper_ids else 0
                    status_badge = f"{winner_pid} stronger"
                    badge_cls = BADGE_CLASSES[idx % len(BADGE_CLASSES)]
                else:
                    status_badge = "Comparable"
                    badge_cls = "badge-pill-a"

                st.markdown(
                    f"<div class='dim-card'>"
                    f"<div style='display:flex;align-items:center;gap:0.5rem;'>"
                    f"<span style='font-size:1.2rem;'>{res.topic_icon}</span>"
                    f"<strong style='font-size:0.95rem;'>{res.topic_name}</strong>"
                    f"&nbsp;<span class='{badge_cls}'>{status_badge}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                score_parts = []
                for pid in paper_ids:
                    assess = res.assessments.get(pid)
                    if assess and assess.status == "assessed":
                        color = PAPER_COLORS[paper_ids.index(pid) % len(PAPER_COLORS)]
                        score_parts.append(f"<span style='color:{color};font-weight:600;'>{pid}: {assess.rating:.1f}</span>")
                    elif assess:
                        score_parts.append(f"<span style='color:#64748b;'>{pid}: N/A</span>")
                if score_parts:
                    st.markdown(
                        f"<div style='font-size:0.82rem;margin:0.3rem 0;'>{' &middot; '.join(score_parts)}</div>",
                        unsafe_allow_html=True,
                    )

                reasoning = res.comparison_reasoning or res.one_sentence_verdict
                if reasoning:
                    st.markdown(f"<div style='font-size:0.88rem;color:#cbd5e1;margin:0.3rem 0;'>{reasoning}</div>", unsafe_allow_html=True)

                explain_cols = st.columns(min(n, 4))
                for i, pid in enumerate(paper_ids):
                    with explain_cols[i % len(explain_cols)]:
                        assess = res.assessments.get(pid)
                        st.markdown(f"<span style='font-size:0.82rem;font-weight:600;'>{pid}</span>", unsafe_allow_html=True)
                        if not assess or assess.status == "not_assessable":
                            st.caption("Insufficient evidence.")
                        else:
                            st.markdown(f"<div style='font-size:0.82rem;color:#94a3b8;'>{assess.explanation}</div>", unsafe_allow_html=True)

                has_evidence = any(
                    res.assessments.get(pid) and (res.assessments[pid].key_evidence or res.assessments[pid].page_reference)
                    for pid in paper_ids
                )
                if has_evidence:
                    with st.expander("Evidence"):
                        evid_cols = st.columns(min(n, 4))
                        for i, pid in enumerate(paper_ids):
                            with evid_cols[i % len(evid_cols)]:
                                assess = res.assessments.get(pid)
                                st.markdown(f"**{pid}**")
                                if assess and assess.key_evidence:
                                    for ev in assess.key_evidence:
                                        st.caption(f'"{ev}"')
                                if assess and assess.page_reference:
                                    st.caption(f"Source: {assess.page_reference}")

                if res.signification and res.comparison_reasoning and res.signification != res.comparison_reasoning:
                    st.caption(f"{res.signification}")

                st.markdown('</div>', unsafe_allow_html=True)

        if not_assessed_results:
            st.markdown("---")
            st.markdown("**What Was Not Found**")
            st.caption("These topics did not have enough evidence in the uploaded documents.")
            for res in not_assessed_results:
                st.markdown(
                    f"<div class='dim-card' style='opacity:0.7;'>"
                    f"<div style='display:flex;align-items:center;gap:0.5rem;'>"
                    f"<span style='font-size:1.2rem;'>{res.topic_icon}</span>"
                    f"<strong style='font-size:0.95rem;'>{res.topic_name}</strong>"
                    f"&nbsp;<span class='not-assessable'>Not enough evidence</span>"
                    f"</div>"
                    f"<div style='font-size:0.82rem;color:#94a3b8;margin-top:0.3rem;'>The documents did not contain sufficient information to evaluate this topic.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with tab_matrix:
        st.markdown("### Comparison Matrix")
        grid_table_html = render_elicit_grid_table(results, papers)
        st.markdown(grid_table_html, unsafe_allow_html=True)

    with tab_agreements:
        st.markdown("### Agreed Points & Differences")
        if exec_sum and exec_sum.consensus_report:
            c_rep = exec_sum.consensus_report
            st.info(f"**Main difference:** {c_rep.core_philosophical_split}")

            st.markdown("**What both papers agree on**")
            for ag in c_rep.shared_agreements:
                st.markdown(f"- {ag}")

            st.markdown("**Where they differ**")
            for div in c_rep.key_divergences:
                st.markdown(f"- {div}")
        else:
            st.caption("Comparison generated during analysis.")

    with tab_sources:
        st.markdown("### Source Evidence")
        for key, res in results.items():
            with st.expander(f"{res.topic_icon} {res.topic_name}"):
                src_cols = st.columns(min(n, 4))
                for i, pid in enumerate(paper_ids):
                    with src_cols[i % len(src_cols)]:
                        st.markdown(f"**{pid}**")
                        assess = res.assessments.get(pid)
                        if assess and assess.key_evidence:
                            for ev in assess.key_evidence:
                                st.caption(f'"{ev}"')
                        else:
                            st.caption("No excerpts.")

    st.markdown("---")
    col_foot_l, col_foot_r = st.columns([3, 1])
    with col_foot_r:
        if st.button("Start Over", use_container_width=True, disabled=is_controls_locked()):
            _project_root_reset = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cleanup_session_files(os.path.join(_project_root_reset, "data", "papers"))
            st.session_state.clear()
            init_state()
            st.rerun()
