"""Unit and Integration Tests for RAG Paper Comparison & Topic Analysis."""

import pytest
import numpy as np
from src.rag.ingest import (
    validate_pdf_uploads,
    clean_extracted_text,
    chunk_text,
    PaperMetadata,
    DocumentChunk
)
from src.rag.embed import VectorIndex
from src.rag.retrieve import COMPARISON_TOPICS, TopicDefinition
from src.rag.extract import (
    TopicAssessment,
    TopicComparisonResult,
    evaluate_topic
)


def test_validate_pdf_uploads_valid():
    """Test validation with two standard compliant PDFs."""
    files = [("paper_a.pdf", 10 * 1024 * 1024), ("paper_b.pdf", 15 * 1024 * 1024)]
    is_valid, msg = validate_pdf_uploads(files)
    assert is_valid is True
    assert "successfully validated" in msg.lower()


def test_validate_pdf_uploads_valid_three():
    """Test validation with three PDFs."""
    files = [
        ("paper_a.pdf", 10 * 1024 * 1024),
        ("paper_b.pdf", 15 * 1024 * 1024),
        ("paper_c.pdf", 12 * 1024 * 1024),
    ]
    is_valid, msg = validate_pdf_uploads(files)
    assert is_valid is True
    assert "successfully validated" in msg.lower()


def test_validate_pdf_uploads_valid_four():
    """Test validation with four PDFs."""
    files = [
        ("paper_a.pdf", 5 * 1024 * 1024),
        ("paper_b.pdf", 5 * 1024 * 1024),
        ("paper_c.pdf", 5 * 1024 * 1024),
        ("paper_d.pdf", 5 * 1024 * 1024),
    ]
    is_valid, msg = validate_pdf_uploads(files)
    assert is_valid is True


def test_validate_pdf_uploads_wrong_count():
    """Test validation fails if not 2-4 PDFs."""
    files_one = [("paper_a.pdf", 5 * 1024 * 1024)]
    is_valid, msg = validate_pdf_uploads(files_one)
    assert is_valid is False
    assert "2-4" in msg


def test_validate_pdf_uploads_too_many():
    """Test validation fails if more than 4 PDFs."""
    files = [
        ("p1.pdf", 5 * 1024 * 1024),
        ("p2.pdf", 5 * 1024 * 1024),
        ("p3.pdf", 5 * 1024 * 1024),
        ("p4.pdf", 5 * 1024 * 1024),
        ("p5.pdf", 5 * 1024 * 1024),
    ]
    is_valid, msg = validate_pdf_uploads(files)
    assert is_valid is False
    assert "2-4" in msg


def test_validate_pdf_uploads_invalid_extension():
    """Test rejection of non-PDF files."""
    files = [("paper_a.docx", 5 * 1024 * 1024), ("paper_b.pdf", 5 * 1024 * 1024)]
    is_valid, msg = validate_pdf_uploads(files)
    assert is_valid is False
    assert "Only PDF files are supported" in msg


def test_validate_pdf_uploads_file_size_limit():
    """Test rejection when single file exceeds 50 MB limit."""
    files = [("large_paper.pdf", 55 * 1024 * 1024), ("small_paper.pdf", 5 * 1024 * 1024)]
    is_valid, msg = validate_pdf_uploads(files)
    assert is_valid is False
    assert "exceeds the limit of 50" in msg


def test_validate_pdf_uploads_total_size_limit():
    """Test rejection when total size exceeds 100 MB limit."""
    files = [("paper_a.pdf", 48 * 1024 * 1024), ("paper_b.pdf", 54 * 1024 * 1024)]
    is_valid, msg = validate_pdf_uploads(files, max_file_mb=60, max_total_mb=100)
    assert is_valid is False
    assert "exceeds the allowed total limit of 100" in msg


def test_clean_extracted_text():
    """Test whitespace cleaning and hyphenation fix."""
    raw = "This is a psycho-\nlogical study of   human     behavior.\n\n\nNext section."
    cleaned = clean_extracted_text(raw)
    assert "psychological" in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunk_text_generation():
    """Test document chunking with metadata."""
    sample_text = "Paragraph 1 describing experimental method.\n\nParagraph 2 describing results."
    chunks = chunk_text(sample_text, paper_id="Paper A", paper_name="Milgram1963.pdf", page_number=1, chunk_size=100)
    assert len(chunks) >= 1
    assert chunks[0].paper_id == "Paper A"
    assert chunks[0].page_number == 1
    assert "Paragraph 1" in chunks[0].text


def test_comparison_topics_defined():
    """Verify that all 6 required comparison topics exist."""
    required_keys = [
        "methodology_rigor",
        "evidence_findings",
        "contribution_novelty",
        "clarity_accessibility",
        "scholarly_grounding",
        "ethical_considerations"
    ]
    for key in required_keys:
        assert key in COMPARISON_TOPICS
        topic = COMPARISON_TOPICS[key]
        assert isinstance(topic, TopicDefinition)
        assert topic.name
        assert topic.extraction_prompt
        assert len(topic.search_queries) > 0


def test_offline_topic_evaluation():
    """Verify offline fallback evaluation produces valid ratings (1-5) and signification."""
    topic = COMPARISON_TOPICS["methodology_rigor"]
    meta_a = PaperMetadata(paper_id="Paper A", filename="doc_a.pdf", title="Stanford Experiment", num_pages=10, total_words=5000, file_size_mb=2.1)
    meta_b = PaperMetadata(paper_id="Paper B", filename="doc_b.pdf", title="Milgram Study", num_pages=12, total_words=6000, file_size_mb=2.5)

    idx_a = VectorIndex("Paper A", meta_a.title)
    idx_b = VectorIndex("Paper B", meta_b.title)

    papers = [
        {"meta": meta_a, "index": idx_a},
        {"meta": meta_b, "index": idx_b},
    ]

    result = evaluate_topic(
        topic=topic,
        papers=papers,
        client=None  # offline mode
    )

    assert isinstance(result, TopicComparisonResult)
    assert "Paper A" in result.assessments
    assert "Paper B" in result.assessments
    assert 1.0 <= result.assessment_a.rating <= 5.0
    assert 1.0 <= result.assessment_b.rating <= 5.0
    assert result.signification
    assert result.better_paper


def test_offline_topic_evaluation_three_papers():
    """Verify offline fallback works for 3 papers."""
    topic = COMPARISON_TOPICS["methodology_rigor"]
    papers_meta = [
        PaperMetadata(paper_id="Paper A", filename="a.pdf", title="Paper Alpha", num_pages=10, total_words=5000, file_size_mb=2.0),
        PaperMetadata(paper_id="Paper B", filename="b.pdf", title="Paper Beta", num_pages=12, total_words=6000, file_size_mb=2.5),
        PaperMetadata(paper_id="Paper C", filename="c.pdf", title="Paper Gamma", num_pages=8, total_words=4000, file_size_mb=1.8),
    ]

    papers = [
        {"meta": pm, "index": VectorIndex(pm.paper_id, pm.title)}
        for pm in papers_meta
    ]

    result = evaluate_topic(topic=topic, papers=papers, client=None)

    assert isinstance(result, TopicComparisonResult)
    assert len(result.assessments) == 3
    assert "Paper A" in result.assessments
    assert "Paper B" in result.assessments
    assert "Paper C" in result.assessments
    assert result.better_paper
