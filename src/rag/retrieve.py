"""Context Retrieval, Targeted Section Routing, and Topic Query Mapping Module.

Implements:
- Section-targeted routing (Elicit style)
- Hybrid semantic vector + keyword scoring
- Exact page & section citation breadcrumbs for verifiable LLM grounding
"""

import re
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from src.rag.embed import VectorIndex
from src.rag.ingest import DocumentChunk


class TopicDefinition(BaseModel):
    """Specification for one of the 6 comparison topics with targeted section routing."""
    key: str
    name: str
    icon: str
    target_sections: List[str] = Field(default_factory=list, description="Preferred academic sections to search")
    keyword_boosts: List[str] = Field(default_factory=list, description="Keywords that give relevance bonuses")
    extraction_prompt: str
    search_queries: List[str]
    description: str


# The 6 standard comparison topics with section routing and hybrid boosts
COMPARISON_TOPICS: Dict[str, TopicDefinition] = {
    "methodology_rigor": TopicDefinition(
        key="methodology_rigor",
        name="Methodology Rigor",
        icon="🔬",
        target_sections=["Methods", "Results", "General"],
        keyword_boosts=["sample", "control", "dataset", "ablation", "procedure", "participants", "baseline", "protocol", "statistical"],
        extraction_prompt="Extract experimental design, sample size, control groups, ablations, and procedural rigor. Rate 1-5 and explain why. If the paper is conceptual with no empirical testing, mark as not_assessable.",
        search_queries=[
            "experimental design methodology procedure sample size controls variables measurement accuracy rigor statistical tests apparatus dataset baseline"
        ],
        description="Quality, thoroughness, sample scale, and scientific validity of the research design and execution.",
    ),
    "evidence_findings": TopicDefinition(
        key="evidence_findings",
        name="Evidence / Findings Support",
        icon="📊",
        target_sections=["Results", "Discussion", "Introduction", "General"],
        keyword_boosts=["accuracy", "improvement", "significant", "real-world", "outperformed", "state-of-the-art", "implications", "findings", "evidence", "data"],
        extraction_prompt="Extract the main empirical findings, supporting evidence, and data-driven conclusions. Rate 1-5. If the paper is purely conceptual with no empirical findings, mark as not_assessable.",
        search_queries=[
            "main findings results outcomes evidence data empirical validation benchmark metrics conclusions supporting evidence"
        ],
        description="Strength of empirical evidence, data-driven findings, and documented results.",
    ),
    "contribution_novelty": TopicDefinition(
        key="contribution_novelty",
        name="Contribution Novelty",
        icon="💡",
        target_sections=["Introduction", "Discussion", "Methods", "General"],
        keyword_boosts=["novel", "first", "unique", "proposed", "paradigm", "distinction", "breakthrough"],
        extraction_prompt="What unique conceptual, theoretical, or architectural contribution does this paper make? Rate 1-5.",
        search_queries=[
            "unique contribution novel approach innovative paradigm originality paradigm shift departure from prior literature new theoretical framework"
        ],
        description="Distinctiveness, originality, and conceptual breakthrough compared to prior literature.",
    ),
    "clarity_accessibility": TopicDefinition(
        key="clarity_accessibility",
        name="Clarity & Accessibility",
        icon="📖",
        target_sections=["Introduction", "Discussion", "General"],
        keyword_boosts=["overview", "introduction", "narrative", "audience", "clarity", "readability", "pedagogical"],
        extraction_prompt="How easy to understand is this paper? Who is the target audience? Rate 1-5 based on writing clarity, structure, and accessibility.",
        search_queries=[
            "introduction narrative clarity readability terminology pedagogical style target audience structure explanation"
        ],
        description="Clarity of exposition, narrative style, and readability for broader or cross-disciplinary audiences.",
    ),
    "scholarly_grounding": TopicDefinition(
        key="scholarly_grounding",
        name="Scholarly Grounding",
        icon="📚",
        target_sections=["Introduction", "Discussion", "References", "General"],
        keyword_boosts=["foundational", "seminal", "widely", "influence", "citations", "literature", "adopted", "references", "prior work"],
        extraction_prompt="Evaluate the presence and use of references, connection between claims and sources, and engagement with prior literature. Rate 1-5. If no references are present, mark as not_assessable. Do NOT claim external citation impact.",
        search_queries=[
            "references bibliography citations prior work literature review related work foundational work"
        ],
        description="Use of references, engagement with prior literature, and connection between claims and sources.",
    ),
    "ethical_considerations": TopicDefinition(
        key="ethical_considerations",
        name="Ethical Considerations",
        icon="⚖️",
        target_sections=["Ethics & Limitations", "Methods", "Discussion", "General"],
        keyword_boosts=["consent", "ethics", "debriefing", "harm", "irb", "safety", "limitations", "risk", "bias", "privacy"],
        extraction_prompt="Are ethical considerations, consent, safety, or limitations discussed? How were they addressed? Rate 1-5. If no ethical procedures are discussed, mark as not_assessable.",
        search_queries=[
            "ethical issues participant welfare informed consent psychological harm deception debriefing institutional review board ethics protocols limitations"
        ],
        description="Ethical considerations, participant treatment, consent, limitations, and harm mitigation.",
    ),
}


def retrieve_paper_context(
    index: VectorIndex,
    topic: TopicDefinition,
    top_k: int = 4
) -> List[tuple[DocumentChunk, float]]:
    """Retrieve top chunks with section-aware routing bonuses and keyword boosting."""
    all_results: Dict[str, tuple[DocumentChunk, float]] = {}
    queries = [topic.extraction_prompt] + topic.search_queries
    
    for query in queries:
        matches = index.search(query, top_k=top_k * 2)
        for chunk, base_score in matches:
            score = base_score
            
            # 1. Section Routing Bonus (Elicit style)
            if chunk.section in topic.target_sections:
                score += 0.15
            
            # 2. Keyword Match Bonus
            chunk_lower = chunk.text.lower()
            keyword_hits = sum(1 for kw in topic.keyword_boosts if kw in chunk_lower)
            score += min(0.12, keyword_hits * 0.03)

            if chunk.chunk_id not in all_results or score > all_results[chunk.chunk_id][1]:
                all_results[chunk.chunk_id] = (chunk, score)

    sorted_chunks = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
    return sorted_chunks[:top_k]


def format_chunks_for_prompt(chunks_with_scores: List[tuple[DocumentChunk, float]], paper_title: str) -> str:
    """Format retrieved passages into a structured prompt block with verifiable page citations."""
    if not chunks_with_scores:
        return f"[No specific excerpts found for {paper_title}]"

    formatted_passages = []
    for idx, (chunk, score) in enumerate(chunks_with_scores, 1):
        formatted_passages.append(
            f"--- [Citation: Page {chunk.page_number} | Section: {chunk.section} | Grounding Score: {score:.2f}] ---\n\"{chunk.text}\""
        )
    return "\n\n".join(formatted_passages)
