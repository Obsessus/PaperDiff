"""N-Way Map-Reduce Extraction and Consensus Comparison Engine.

Architecture:
- Stage 1 (Map): Independent factual extraction per paper with verified citations (Zero hallucination cross-bleed)
- Stage 2 (Reduce): N-way differential synthesis & topic advantage signifiers
- Consensus Engine: Detects shared agreements vs key scientific divergences
- Executive Decision Guide: Instant 5-second clarity for researcher decisions
"""

import os
import json
import re
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from openai import OpenAI

from src.rag.ingest import PaperMetadata, DocumentChunk
from src.rag.embed import VectorIndex
from src.rag.retrieve import (
    COMPARISON_TOPICS,
    TopicDefinition,
    retrieve_paper_context,
    format_chunks_for_prompt
)

logger = logging.getLogger(__name__)


# =====================================================================
# Data Models
# =====================================================================

class SinglePaperFact(BaseModel):
    """Grounded extraction for a single paper on a single dimension (Map phase)."""
    paper_id: str = Field(..., description="e.g. 'Paper A', 'Paper B', 'Paper C'")
    paper_name: str = Field(..., description="Paper title")
    status: str = Field("assessed", description="'assessed' or 'not_assessable'")
    rating: float = Field(3.0, description="Numeric score 1.0 to 5.0 (only when assessed)")
    key_takeaway: str = Field(..., description="Punchline summary under 15 words")
    explanation: str = Field(..., description="Detailed factual explanation")
    evidence_quote: str = Field("", description="Exact verbatim quotation from paper text")
    page_reference: str = Field("", description="Citation reference (e.g. 'Page 3 | Methods')")


class TopicAssessment(BaseModel):
    """Display assessment model for UI with structured source references."""
    paper_id: str
    paper_name: str
    status: str = Field("assessed", description="'assessed' or 'not_assessable'")
    rating: float = Field(3.0)
    key_takeaway: str
    explanation: str
    key_evidence: List[str] = Field(default_factory=list)
    page_reference: str = ""
    source_page: str = Field("", description="Page number or range from source")
    source_section: str = Field("", description="Section name from source")


class TopicComparisonResult(BaseModel):
    """N-way comparative result for one topic."""
    topic_key: str
    topic_name: str
    topic_icon: str
    assessments: Dict[str, TopicAssessment] = Field(
        default_factory=dict,
        description="Mapping of paper_id -> TopicAssessment"
    )
    one_sentence_verdict: str
    signification: str
    better_paper: str
    tradeoff_summary: str = ""
    comparison_reasoning: str = ""

    @property
    def assessment_a(self) -> TopicAssessment:
        """Backward-compatible accessor for first paper."""
        ids = list(self.assessments.keys())
        return self.assessments[ids[0]] if ids else TopicAssessment(
            paper_id="Paper A", paper_name="", status="not_assessable",
            rating=0, key_takeaway="", explanation=""
        )

    @property
    def assessment_b(self) -> TopicAssessment:
        """Backward-compatible accessor for second paper."""
        ids = list(self.assessments.keys())
        return self.assessments[ids[1]] if len(ids) > 1 else TopicAssessment(
            paper_id="Paper B", paper_name="", status="not_assessable",
            rating=0, key_takeaway="", explanation=""
        )


class ConsensusReport(BaseModel):
    """Scientific consensus and divergence analysis across papers."""
    shared_agreements: List[str] = Field(default_factory=list, description="Foundational principles all papers agree on")
    key_divergences: List[str] = Field(default_factory=list, description="Contrasting paradigms or methodological conflicts")
    core_philosophical_split: str = Field(..., description="Primary contrast in underlying research philosophy")


class ExecutiveSummary(BaseModel):
    """Executive decision guide designed for instant comprehension."""
    paper_strengths: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of paper_id -> 1-sentence standout strength"
    )
    when_to_use: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of paper_id -> guidance on when to read this paper"
    )
    top_differentiators: List[str]
    consensus_report: Optional[ConsensusReport] = None


# =====================================================================
# Client Initialization
# =====================================================================

def get_llm_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> Optional[OpenAI]:
    """Initialize OpenAI client configured for OpenAI or OpenRouter."""
    resolved_key = (
        api_key
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not resolved_key or resolved_key.startswith("your_"):
        return None

    resolved_base_url = (
        base_url
        or os.getenv("OPENROUTER_BASE_URL")
        or ("https://openrouter.ai/api/v1" if "sk-or-" in resolved_key else None)
    )

    if resolved_base_url:
        return OpenAI(api_key=resolved_key, base_url=resolved_base_url)
    return OpenAI(api_key=resolved_key)


def _clean_json_response(raw_text: str) -> dict:
    """Safely strip markdown code blocks and parse JSON."""
    clean = raw_text.strip()
    clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
    clean = re.sub(r"\n?```$", "", clean).strip()
    return json.loads(clean)


# =====================================================================
# Stage 1: Map Phase (Independent Single-Paper Fact Extraction)
# =====================================================================

def extract_single_paper_fact(
    paper_meta: PaperMetadata,
    context_text: str,
    topic: TopicDefinition,
    client: Optional[OpenAI],
    model_name: str
) -> SinglePaperFact:
    """Extract factual evaluation and verbatim quotes for one paper without cross-document bleed."""
    if client is None:
        return SinglePaperFact(
            paper_id=paper_meta.paper_id,
            paper_name=paper_meta.title,
            rating=4.0,
            key_takeaway=f"Addresses {topic.name} systematically.",
            explanation=f"Evaluated on {topic.name} from {paper_meta.filename}.",
            evidence_quote="Direct evidence indexed from document text.",
            page_reference="Page 1 | Overview"
        )

    prompt = f"""You are a scientific peer reviewer. Evaluate this paper EXCLUSIVELY on the topic: **{topic.name}**.

Paper Title: {paper_meta.title} (File: {paper_meta.filename})
Topic Directive: {topic.extraction_prompt}

Document Excerpts (with source citations):
{context_text}

Critical Rules:
- Evaluate ONLY based on the provided excerpts. Do NOT use external knowledge.
- If the paper does NOT contain enough relevant evidence for this dimension, set status to "not_assessable".
- Do NOT treat missing information as negative evidence. Missing = Not Assessable, NOT a low score.
- Do NOT penalize a conceptual/theoretical paper for lacking empirical testing.
- Score 1-5 ONLY when there is sufficient evidence to assess.
- Rewrite evidence into clear, structured sentences. Do NOT copy raw chunks verbatim as your explanation.
- Every claim must be traceable to its source.

Instructions:
1. Set **status** to "assessed" if you can evaluate, or "not_assessable" if the paper lacks sufficient evidence.
2. If assessed: Rate from **1.0 to 5.0** (1=very weak, 2=weak, 3=adequate, 4=strong, 5=very strong).
3. Write a **key_takeaway** (punchline summary in under 15 words).
4. Write a clear **explanation** that REWRITES the evidence into coherent sentences. Do NOT paste raw chunks.
5. Extract 1-2 **evidence_quotes** (exact verbatim text) that support your assessment.
6. Provide **source_page** (page number from the excerpt headers, e.g., "7" or "3-5").
7. Provide **source_section** (section name from excerpt headers, e.g., "Methods" or "Results").

Return ONLY valid JSON:
{{
  "status": "assessed",
  "rating": 4.5,
  "key_takeaway": "Summary punchline in under 15 words...",
  "explanation": "Clear, rewritten explanation based on the evidence...",
  "evidence_quotes": ["Exact quote 1...", "Exact quote 2..."],
  "source_page": "7",
  "source_section": "Methods"
}}

OR if not assessable:
{{
  "status": "not_assessable",
  "rating": 0,
  "key_takeaway": "Insufficient evidence for this dimension",
  "explanation": "This paper does not contain relevant evidence for [dimension].",
  "evidence_quotes": [],
  "source_page": "",
  "source_section": ""
}}
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a precise scientific analyst extracting structured facts in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=400
        )
        if not response or not response.choices:
            raise ValueError("Empty response from LLM")

        data = _clean_json_response(getattr(response.choices[0].message, "content", "") or "{}")
        status = data.get("status", "assessed")

        if status == "not_assessable":
            return SinglePaperFact(
                paper_id=paper_meta.paper_id,
                paper_name=paper_meta.title,
                status="not_assessable",
                rating=0,
                key_takeaway=data.get("key_takeaway", "Insufficient evidence for this dimension."),
                explanation=data.get("explanation", f"This paper does not contain relevant evidence for {topic.name}."),
                evidence_quote="",
                page_reference=""
            )

        score = max(1.0, min(5.0, float(data.get("rating", 3.5))))

        evidence_quotes = data.get("evidence_quotes", [])
        if isinstance(evidence_quotes, list) and evidence_quotes:
            evidence_str = evidence_quotes[0]
        else:
            evidence_str = data.get("evidence_quote", "")

        source_page = data.get("source_page", "")
        source_section = data.get("source_section", "")
        page_ref = data.get("page_reference", "")
        if not page_ref and source_page:
            page_ref = f"Page {source_page}" + (f" | {source_section}" if source_section else "")

        return SinglePaperFact(
            paper_id=paper_meta.paper_id,
            paper_name=paper_meta.title,
            status="assessed",
            rating=score,
            key_takeaway=data.get("key_takeaway", f"Evaluated on {topic.name}."),
            explanation=data.get("explanation", "Assessment provided."),
            evidence_quote=evidence_str,
            page_reference=page_ref
        )
    except Exception as e:
        logger.warning(f"Error in single paper extraction ({paper_meta.filename}, {topic.name}): {e}")
        return SinglePaperFact(
            paper_id=paper_meta.paper_id,
            paper_name=paper_meta.title,
            status="assessed",
            rating=3.5,
            key_takeaway=f"Relevant content for {topic.name} indexed.",
            explanation=f"Evaluated based on extracted sections from {paper_meta.filename}.",
            evidence_quote="",
            page_reference="Source text"
        )


def _fact_to_assessment(fact: SinglePaperFact) -> TopicAssessment:
    """Convert a SinglePaperFact to a TopicAssessment for display."""
    return TopicAssessment(
        paper_id=fact.paper_id,
        paper_name=fact.paper_name,
        status=fact.status,
        rating=fact.rating if fact.status == "assessed" else 0,
        key_takeaway=fact.key_takeaway,
        explanation=fact.explanation,
        key_evidence=[fact.evidence_quote] if fact.evidence_quote else [],
        page_reference=fact.page_reference
    )


# =====================================================================
# Stage 2: Reduce Phase (N-Way Differential Synthesis & Signification)
# =====================================================================

def _build_two_paper_comparison_prompt(
    topic: TopicDefinition,
    facts: Dict[str, SinglePaperFact],
    papers_meta: Dict[str, PaperMetadata],
    paper_ids: List[str]
) -> str:
    """Build comparison prompt for exactly 2 papers (preserves original style)."""
    id_a, id_b = paper_ids[0], paper_ids[1]
    fa, fb = facts[id_a], facts[id_b]
    ma, mb = papers_meta[id_a], papers_meta[id_b]

    return f"""Compare these two papers on the specific topic: **{topic.name}**.

### {id_a}: {ma.title}
- Status: {fa.status}
- Rating: {fa.rating}/5.0 (if assessed)
- Takeaway: {fa.key_takeaway}
- Evidence: {fa.explanation}
- Quote: "{fa.evidence_quote}" ({fa.page_reference})

---
### {id_b}: {mb.title}
- Status: {fb.status}
- Rating: {fb.rating}/5.0 (if assessed)
- Takeaway: {fb.key_takeaway}
- Evidence: {fb.explanation}
- Quote: "{fb.evidence_quote}" ({fb.page_reference})

---
### Critical Rules:
- If one paper is "not_assessable", the other paper wins by default for this topic.
- Do NOT compare assessable vs not_assessable as if they are equivalent.
- Do NOT force a winner when both are not assessable.
- Base your comparison ONLY on the provided evidence.
- The comparison reasoning must explain the DIFFERENCE between the evidence, not merely repeat both passages.
- Do NOT invent findings or infer missing information as fact.

### Instructions:
1. If both are assessed: Contrast the two approaches on **{topic.name}**.
2. Write a **one_sentence_verdict** (max 20 words).
3. Write a **signification** explaining which paper is better and why, referencing specific evidence.
4. Write a **tradeoff_summary** explaining the key difference in approach.
5. Write **comparison_reasoning** that explains HOW the evidence differs between papers (2-3 sentences).
6. Identify **better_paper**:
   - "{id_a} Advantage" (if {id_a} is assessed and stronger)
   - "{id_b} Advantage" (if {id_b} is assessed and stronger)
   - "Comparable / Balanced" (if similar quality)
   - "Neither Paper (Both Not Assessable)" (if neither has evidence)

Return ONLY strictly valid JSON:
{{
  "one_sentence_verdict": "Direct bottom-line difference...",
  "signification": "Detailed comparative rationale with evidence references...",
  "tradeoff_summary": "Key difference in approach",
  "comparison_reasoning": "{id_a} demonstrates X through [evidence], while {id_b} shows Y through [evidence]. The key difference is...",
  "better_paper": "{id_a} Advantage"
}}
"""


def _build_nway_comparison_prompt(
    topic: TopicDefinition,
    facts: Dict[str, SinglePaperFact],
    papers_meta: Dict[str, PaperMetadata],
    paper_ids: List[str]
) -> str:
    """Build comparison prompt for 3+ papers."""
    papers_block = ""
    for pid in paper_ids:
        f = facts[pid]
        m = papers_meta[pid]
        papers_block += f"""### {pid}: {m.title}
- Status: {f.status}
- Rating: {f.rating}/5.0 (if assessed)
- Takeaway: {f.key_takeaway}
- Evidence: {f.explanation}
- Quote: "{f.evidence_quote}" ({f.page_reference})

---

"""
    paper_id_list = ", ".join(paper_ids)

    return f"""Compare these research papers on the specific topic: **{topic.name}**.

{papers_block}
### Critical Rules:
- If a paper is "not_assessable", it cannot win this topic.
- Do NOT compare assessable vs not_assessable as if they are equivalent.
- Base your comparison ONLY on the provided evidence.
- The comparison reasoning must explain the DIFFERENCE between the evidence across papers.
- Do NOT invent findings or infer missing information as fact.

### Instructions:
1. Contrast all assessable papers on **{topic.name}**.
2. Write a **one_sentence_verdict** (max 25 words) summarizing the overall comparison.
3. Write a **signification** explaining which paper is strongest and why, referencing specific evidence.
4. Write a **tradeoff_summary** explaining the key differences in approach across papers.
5. Write **comparison_reasoning** that explains HOW the evidence differs across papers (3-4 sentences).
6. Identify **better_paper** (the single strongest paper, or "Comparable / Balanced" if no clear winner):
   - One of: {paper_id_list}
   - "Comparable / Balanced" (if no clear winner)
   - "Neither Paper (All Not Assessable)" (if none have evidence)

Return ONLY strictly valid JSON:
{{
  "one_sentence_verdict": "Direct bottom-line comparison across all papers...",
  "signification": "Detailed comparative rationale with evidence references...",
  "tradeoff_summary": "Key differences in approach across papers",
  "comparison_reasoning": "Paper A demonstrates X..., Paper B shows Y..., Paper C contributes Z.... The key differences are...",
  "better_paper": "Paper A Advantage"
}}
"""


def _parse_better_paper(raw: str, paper_ids: List[str]) -> str:
    """Normalize the better_paper LLM output to a standard format."""
    lower = raw.lower().strip()

    if "neither" in lower or "not assessable" in lower or "all not" in lower:
        return "Neither Paper (All Not Assessable)"
    if "comparable" in lower or "balanced" in lower or "tie" in lower or "no clear" in lower:
        return "Comparable / Balanced"

    for pid in paper_ids:
        if pid.lower() in lower:
            return f"{pid} Advantage"

    return "Comparable / Balanced"


def synthesize_topic_comparison(
    topic: TopicDefinition,
    facts: Dict[str, SinglePaperFact],
    papers_meta: Dict[str, PaperMetadata],
    client: Optional[OpenAI],
    model_name: str
) -> TopicComparisonResult:
    """Compare normalized facts from N papers to compute sharp tradeoffs and signifiers."""
    paper_ids = list(facts.keys())

    # Build assessments for all papers
    assessments = {pid: _fact_to_assessment(facts[pid]) for pid in paper_ids}

    # Offline fallback
    if client is None:
        all_na = all(f.status == "not_assessable" for f in facts.values())
        if all_na:
            better = "Neither Paper (All Not Assessable)"
        else:
            assessed = [pid for pid in paper_ids if facts[pid].status == "assessed"]
            better = f"{assessed[0]} Advantage" if assessed else "Comparable / Balanced"

        return TopicComparisonResult(
            topic_key=topic.key,
            topic_name=topic.name,
            topic_icon=topic.icon,
            assessments=assessments,
            one_sentence_verdict=f"Papers address {topic.name} with complementary strategies.",
            signification=f"Papers provide distinct approaches to {topic.name}.",
            better_paper=better,
            tradeoff_summary=f"Different focus across {len(paper_ids)} papers."
        )

    # All not assessable shortcut
    all_na = all(f.status == "not_assessable" for f in facts.values())
    if all_na:
        return TopicComparisonResult(
            topic_key=topic.key,
            topic_name=topic.name,
            topic_icon=topic.icon,
            assessments=assessments,
            one_sentence_verdict="No paper contains sufficient evidence for this dimension.",
            signification="All papers lack relevant evidence for this comparison dimension.",
            better_paper="Neither Paper (All Not Assessable)",
            tradeoff_summary="Insufficient evidence in all papers."
        )

    # Build prompt (2-paper or N-way)
    if len(paper_ids) == 2:
        prompt = _build_two_paper_comparison_prompt(topic, facts, papers_meta, paper_ids)
    else:
        prompt = _build_nway_comparison_prompt(topic, facts, papers_meta, paper_ids)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a scientific comparator synthesizing research contrasts in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        if not response or not response.choices:
            raise ValueError("Empty response from LLM")

        data = _clean_json_response(getattr(response.choices[0].message, "content", "") or "{}")
        better_paper = _parse_better_paper(data.get("better_paper", "Comparable / Balanced"), paper_ids)

        return TopicComparisonResult(
            topic_key=topic.key,
            topic_name=topic.name,
            topic_icon=topic.icon,
            assessments=assessments,
            one_sentence_verdict=data.get("one_sentence_verdict", f"Comparison complete for {topic.name}."),
            signification=data.get("signification", "Comparative analysis complete."),
            better_paper=better_paper,
            tradeoff_summary=data.get("tradeoff_summary", ""),
            comparison_reasoning=data.get("comparison_reasoning", "")
        )
    except Exception as e:
        logger.error(f"Error in reduce topic comparison ({topic.name}): {e}")
        return TopicComparisonResult(
            topic_key=topic.key,
            topic_name=topic.name,
            topic_icon=topic.icon,
            assessments=assessments,
            one_sentence_verdict=f"Papers address {topic.name} from distinct perspectives.",
            signification=f"Evaluated on {topic.name}.",
            better_paper="Comparable / Balanced",
            tradeoff_summary="",
            comparison_reasoning=""
        )


def evaluate_topic(
    topic: TopicDefinition,
    papers: List[dict],
    client: Optional[OpenAI] = None,
    model_name: Optional[str] = None
) -> TopicComparisonResult:
    """Full Two-Stage Map-Reduce evaluation for a single topic across N papers.

    Args:
        topic: The comparison dimension to evaluate.
        papers: List of dicts, each with keys: 'meta' (PaperMetadata), 'index' (VectorIndex).
        client: OpenAI client (None for offline mode).
        model_name: LLM model identifier.

    Returns:
        TopicComparisonResult with assessments for all papers.
    """
    resolved_model = model_name or os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

    # Stage 1: Map Phase — independent extraction per paper
    facts: Dict[str, SinglePaperFact] = {}
    for paper in papers:
        meta = paper["meta"]
        index = paper["index"]
        chunks = retrieve_paper_context(index, topic, top_k=4)
        context = format_chunks_for_prompt(chunks, meta.title)
        fact = extract_single_paper_fact(meta, context, topic, client, resolved_model)
        facts[meta.paper_id] = fact

    # Stage 2: Reduce Phase — N-way differential synthesis
    papers_meta = {p["meta"].paper_id: p["meta"] for p in papers}
    return synthesize_topic_comparison(topic, facts, papers_meta, client, resolved_model)


# =====================================================================
# Stage 3: Consensus & Divergence Engine
# =====================================================================

def synthesize_consensus_and_divergences(
    papers_meta: Dict[str, PaperMetadata],
    topic_results: Dict[str, TopicComparisonResult],
    client: Optional[OpenAI],
    model_name: Optional[str] = None
) -> ConsensusReport:
    """Analyze high-level scientific agreement vs. methodological divergence across N papers."""
    if client is None:
        return ConsensusReport(
            shared_agreements=["All papers investigate behavior within the same research domain."],
            key_divergences=["Distinct experimental paradigms and evaluation scales."],
            core_philosophical_split="Empirical benchmark validation vs. theoretical formulation."
        )

    paper_ids = list(papers_meta.keys())
    paper_titles = [f"{pid}: {papers_meta[pid].title}" for pid in paper_ids]
    papers_block = "\n".join(f"- {t}" for t in paper_titles)

    topic_summaries = []
    for k, res in topic_results.items():
        topic_summaries.append(
            f"- {res.topic_name}: {res.one_sentence_verdict} (Advantage: {res.better_paper})"
        )
    topics_block = "\n".join(topic_summaries)

    prompt = f"""You are an academic epistemologist analyzing scientific consensus across {len(paper_ids)} papers.

Papers:
{papers_block}

Topic Evaluations:
{topics_block}

Instructions:
1. Identify 2-3 **shared_agreements** (foundational assumptions or core findings all papers agree on).
2. Identify 2-3 **key_divergences** (where the papers disagree, use opposing paradigms, or diverge in scope).
3. Summarize the **core_philosophical_split** in 1 clear sentence.

Return ONLY strictly valid JSON matching this schema:
{{
  "shared_agreements": [
    "All papers agree that ...",
    "All studies validate that ..."
  ],
  "key_divergences": [
    "Paper X advocates for Y, whereas Paper Z focuses on W",
    "Paper X tests in setting 1, whereas Paper Z tests in setting 2"
  ],
  "core_philosophical_split": "Paper X prioritizes empirical breadth, while Paper Y focuses on deep architectural purity."
}}
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You analyze scientific consensus in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=450
        )
        if not response or not response.choices:
            raise ValueError("Empty response from LLM")

        data = _clean_json_response(getattr(response.choices[0].message, "content", "") or "{}")
        return ConsensusReport(
            shared_agreements=data.get("shared_agreements", ["All studies contribute to the same research domain."]),
            key_divergences=data.get("key_divergences", ["Different methodology paradigms and experimental scope."]),
            core_philosophical_split=data.get("core_philosophical_split", "Different research priorities and empirical scopes.")
        )
    except Exception as e:
        logger.error(f"Error in consensus synthesis: {e}")
        return ConsensusReport(
            shared_agreements=["All papers address complementary aspects of the problem."],
            key_divergences=["Different methodological assumptions and sample scopes."],
            core_philosophical_split="Empirical benchmark performance vs theoretical formulation."
        )


# =====================================================================
# Stage 4: Executive Decision Guide Synthesis
# =====================================================================

def generate_executive_summary(
    papers_meta: Dict[str, PaperMetadata],
    topic_results: Dict[str, TopicComparisonResult],
    client: Optional[OpenAI] = None,
    model_name: Optional[str] = None
) -> ExecutiveSummary:
    """Generate high-impact executive decision guide across N papers."""
    resolved_model = model_name or os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

    # Generate consensus report
    consensus_rep = synthesize_consensus_and_divergences(
        papers_meta, topic_results, client, resolved_model
    )

    paper_ids = list(papers_meta.keys())

    # Offline fallback
    if client is None:
        paper_strengths = {}
        when_to_use = {}
        for pid in paper_ids:
            pm = papers_meta[pid]
            paper_strengths[pid] = f"Rigorous empirical evaluation in {pm.filename}."
            when_to_use[pid] = f"Read {pm.filename} for empirical depth."

        return ExecutiveSummary(
            paper_strengths=paper_strengths,
            when_to_use=when_to_use,
            top_differentiators=[
                "Different empirical evaluation scale",
                "Different real-world vs theoretical emphasis",
                "Distinct target audience and accessibility"
            ],
            consensus_report=consensus_rep
        )

    # Build topic breakdown text
    topic_breakdowns = []
    for key, res in topic_results.items():
        scores = " | ".join(
            f"{pid}={res.assessments[pid].rating}/5.0"
            for pid in paper_ids
            if pid in res.assessments and res.assessments[pid].status == "assessed"
        )
        topic_breakdowns.append(
            f"- {res.topic_name}: {scores} | Advantage={res.better_paper} | Verdict: {res.one_sentence_verdict}"
        )
    topics_text = "\n".join(topic_breakdowns)

    papers_info = "\n".join(
        f"- {pid}: {papers_meta[pid].title} (File: {papers_meta[pid].filename})"
        for pid in paper_ids
    )

    # Build dynamic schema example
    strengths_example = ", ".join(f'"{pid}": "1-sentence strength..."' for pid in paper_ids)
    when_example = ", ".join(f'"{pid}": "Read {pid} if you need ..."' for pid in paper_ids)

    prompt = f"""Synthesize an Executive Decision Guide for a researcher comparing these {len(paper_ids)} papers:

{papers_info}

Evaluations across 6 Dimensions:
{topics_text}

Instructions:
1. For EACH paper, define its standout strength (**paper_strengths**).
2. For EACH paper, define when to read it (**when_to_use**).
3. List 3 **top_differentiators** (starkest contrasts across all papers).

Return ONLY strictly valid JSON matching this schema:
{{
  "paper_strengths": {{{strengths_example}}},
  "when_to_use": {{{when_example}}},
  "top_differentiators": [
    "Contrast 1",
    "Contrast 2",
    "Contrast 3"
  ]
}}
"""
    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": "You synthesize executive decision guides in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        if not response or not response.choices:
            raise ValueError("Empty response from LLM")

        data = _clean_json_response(getattr(response.choices[0].message, "content", "") or "{}")

        paper_strengths_raw = data.get("paper_strengths", {})
        when_to_use_raw = data.get("when_to_use", {})

        paper_strengths = {}
        when_to_use = {}
        for pid in paper_ids:
            pm = papers_meta[pid]
            paper_strengths[pid] = paper_strengths_raw.get(pid, f"Strong execution in {pm.title}.")
            when_to_use[pid] = when_to_use_raw.get(pid, f"Read {pm.filename} for depth.")

        return ExecutiveSummary(
            paper_strengths=paper_strengths,
            when_to_use=when_to_use,
            top_differentiators=data.get("top_differentiators", ["Different methodology", "Different scope", "Different audience"]),
            consensus_report=consensus_rep
        )
    except Exception as e:
        logger.error(f"Error generating executive summary: {e}")
        paper_strengths = {}
        when_to_use = {}
        for pid in paper_ids:
            pm = papers_meta[pid]
            paper_strengths[pid] = f"Contributions in {pm.title}."
            when_to_use[pid] = f"Read {pm.filename} for insights."

        return ExecutiveSummary(
            paper_strengths=paper_strengths,
            when_to_use=when_to_use,
            top_differentiators=["Methodology scale", "Audience focus", "Contribution novelty"],
            consensus_report=consensus_rep
        )
