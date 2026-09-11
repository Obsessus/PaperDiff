"""PDF Ingestion, Validation, Section Parsing, and Text Chunking Module.

Handles:
- Upload size and format validation (50 MB / 100 MB rules)
- Structure-aware academic section tagging (Methods, Results, Ethics, Intro, Discussion)
- Text and metadata extraction with PyMuPDF
- Semantic chunking with page-level and section-level attribution
- Session file cleanup
"""

import os
import re
import unicodedata
import fitz  # PyMuPDF
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field


MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_TOTAL_SIZE_MB = float(os.getenv("MAX_TOTAL_SIZE_MB", "100"))

# Academic Section Detection Patterns
SECTION_PATTERNS = {
    "Methods": re.compile(
        r'^\s*(?:\d+[\.\s]+)?(?:methods?|methodology|experimental\s+(?:setup|design|procedure|framework)|materials?\s+and\s+methods?|implementation|study\s+design|procedures?|empirical\s+setup)\b',
        re.IGNORECASE
    ),
    "Results": re.compile(
        r'^\s*(?:\d+[\.\s]+)?(?:results?|findings?|experiments?|evaluations?|empirical\s+analysis|performance(?:\s+analysis)?|observations?)\b',
        re.IGNORECASE
    ),
    "Ethics & Limitations": re.compile(
        r'^\s*(?:\d+[\.\s]+)?(?:ethical\s+(?:considerations?|issues?|aspects?)|ethics?|participants?|informed\s+consent|debriefing|limitations?|broader\s+impacts?|potential\s+risks?)\b',
        re.IGNORECASE
    ),
    "Discussion": re.compile(
        r'^\s*(?:\d+[\.\s]+)?(?:discussions?|implications?|conclusions?|summary|future\s+work)\b',
        re.IGNORECASE
    ),
    "Introduction": re.compile(
        r'^\s*(?:\d+[\.\s]+)?(?:abstract|introduction|background|motivation|related\s+work|overview)\b',
        re.IGNORECASE
    ),
    "References": re.compile(
        r'^\s*(?:\d+[\.\s]+)?(?:references?|bibliography|citations?)\b',
        re.IGNORECASE
    )
}


class PaperMetadata(BaseModel):
    """Metadata representing an uploaded research paper."""
    paper_id: str = Field(..., description="'Paper A' or 'Paper B'")
    filename: str = Field(..., description="Original filename of the PDF")
    title: str = Field(..., description="Extracted paper title or filename")
    num_pages: int = Field(0, description="Total number of pages in the PDF")
    total_words: int = Field(0, description="Total word count in extracted text")
    file_size_mb: float = Field(0.0, description="File size in Megabytes")
    detected_sections: List[str] = Field(default_factory=list, description="Sections identified in document")


class DocumentChunk(BaseModel):
    """A granular chunk of text extracted from a paper with section attribution."""
    chunk_id: str = Field(..., description="Unique ID for this chunk (e.g. PaperA_p1_c0)")
    paper_id: str = Field(..., description="'Paper A' or 'Paper B'")
    paper_name: str = Field(..., description="Name or title of the paper")
    page_number: int = Field(..., description="1-indexed page number where chunk originated")
    section: str = Field("General", description="Academic section tag (Methods, Results, Ethics, etc.)")
    text: str = Field(..., description="Cleaned chunk text content")
    char_count: int = Field(..., description="Number of characters in the chunk")


def validate_pdf_uploads(
    files: List[Tuple[str, int]],
    max_file_mb: float = MAX_FILE_SIZE_MB,
    max_total_mb: float = MAX_TOTAL_SIZE_MB
) -> Tuple[bool, str]:
    """Validate uploaded PDF files against extension and size constraints."""
    if not files:
        return False, "No files uploaded. Please upload 2-4 PDF research papers."

    if len(files) < 2 or len(files) > 4:
        return False, f"Expected 2-4 research papers for comparison, received {len(files)}."

    total_bytes = 0
    for filename, size_bytes in files:
        if not filename.lower().endswith(".pdf"):
            return False, f"Invalid file format for '{filename}'. Only PDF files are supported."

        size_mb = size_bytes / (1024 * 1024)
        if size_mb > max_file_mb:
            return (
                False,
                f"File '{filename}' is {size_mb:.2f} MB, which exceeds the limit of {max_file_mb} MB per file."
            )
        total_bytes += size_bytes

    total_mb = total_bytes / (1024 * 1024)
    if total_mb > max_total_mb:
        return (
            False,
            f"Total upload size is {total_mb:.2f} MB, which exceeds the allowed total limit of {max_total_mb} MB."
        )

    return True, "Files successfully validated."


def clean_extracted_text(text: str) -> str:
    """Normalize whitespace, fix broken hyphenations, and sanitize unicode artifacts."""
    if not text:
        return ""
    # Normalize unicode accents and symbols to nearest ASCII equivalent
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    # Fix hyphenated words broken across line breaks (e.g., 'experi-\nment' -> 'experiment')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace multiple newlines or tabs with a single space or clean newline
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def detect_section_header(line: str) -> Optional[str]:
    """Determine if a text line represents a standard academic section heading."""
    clean_line = line.strip()
    if not clean_line or len(clean_line) > 80:
        return None

    for section_name, pattern in SECTION_PATTERNS.items():
        if pattern.match(clean_line):
            return section_name
    return None


def chunk_text(
    text: str,
    paper_id: str,
    paper_name: str,
    page_number: int,
    section: str = "General",
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[DocumentChunk]:
    """Split text into overlapping chunks with section-level attribution."""
    cleaned = clean_extracted_text(text)
    if not cleaned:
        return []

    chunks: List[DocumentChunk] = []
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    
    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}".strip()
        else:
            if current_chunk:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{paper_id}_p{page_number}_c{chunk_idx}",
                        paper_id=paper_id,
                        paper_name=paper_name,
                        page_number=page_number,
                        section=section,
                        text=current_chunk,
                        char_count=len(current_chunk),
                    )
                )
                chunk_idx += 1
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = f"{overlap_text}\n\n{para}".strip()
            else:
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    sub_text = para[i:i + chunk_size].strip()
                    if sub_text:
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{paper_id}_p{page_number}_c{chunk_idx}",
                                paper_id=paper_id,
                                paper_name=paper_name,
                                page_number=page_number,
                                section=section,
                                text=sub_text,
                                char_count=len(sub_text),
                            )
                        )
                        chunk_idx += 1
                current_chunk = ""

    if current_chunk:
        chunks.append(
            DocumentChunk(
                chunk_id=f"{paper_id}_p{page_number}_c{chunk_idx}",
                paper_id=paper_id,
                paper_name=paper_name,
                page_number=page_number,
                section=section,
                text=current_chunk,
                char_count=len(current_chunk),
            )
        )

    return chunks


def process_pdf_document(
    file_bytes_or_path: Any,
    filename: str,
    paper_id: str = "Paper A",
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> Tuple[PaperMetadata, List[DocumentChunk]]:
    """Extract text, academic section tags, metadata, and chunked passages from a PDF file."""
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        doc = fitz.open(stream=file_bytes_or_path, filetype="pdf")
        file_size_mb = len(file_bytes_or_path) / (1024 * 1024)
    else:
        doc = fitz.open(str(file_bytes_or_path))
        file_size_mb = os.path.getsize(str(file_bytes_or_path)) / (1024 * 1024)

    num_pages = len(doc)
    all_chunks: List[DocumentChunk] = []
    total_words = 0
    extracted_title = ""
    detected_sections_set = set()
    current_section = "Introduction"

    for page_idx in range(num_pages):
        page = doc[page_idx]
        page_text = page.get_text("text")
        
        words = page_text.split()
        total_words += len(words)

        lines = [l.strip() for l in page_text.splitlines() if l.strip()]

        # Extract title from first page
        if page_idx == 0 and not extracted_title:
            if lines:
                candidate = clean_extracted_text(" ".join(lines[:2]))
                extracted_title = candidate[:120]

        # Scan for section header updates
        page_section_chunks = []
        current_block_lines = []
        
        for line in lines:
            header_tag = detect_section_header(line)
            if header_tag:
                # Save previous block with previous section tag
                if current_block_lines:
                    block_text = "\n".join(current_block_lines)
                    page_section_chunks.extend(
                        chunk_text(
                            text=block_text,
                            paper_id=paper_id,
                            paper_name=filename,
                            page_number=page_idx + 1,
                            section=current_section,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    )
                    current_block_lines = []
                current_section = header_tag
                detected_sections_set.add(header_tag)
            else:
                current_block_lines.append(line)

        if current_block_lines:
            block_text = "\n".join(current_block_lines)
            page_section_chunks.extend(
                chunk_text(
                    text=block_text,
                    paper_id=paper_id,
                    paper_name=filename,
                    page_number=page_idx + 1,
                    section=current_section,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            )

        all_chunks.extend(page_section_chunks)

    doc.close()

    metadata = PaperMetadata(
        paper_id=paper_id,
        filename=filename,
        title=extracted_title if extracted_title else os.path.splitext(filename)[0],
        num_pages=num_pages,
        total_words=total_words,
        file_size_mb=round(file_size_mb, 2),
        detected_sections=list(detected_sections_set) if detected_sections_set else ["General"]
    )

    return metadata, all_chunks


def cleanup_session_files(directory: str = "data/papers") -> int:
    """Remove uploaded PDF files from local disk to guarantee zero machine storage accumulation."""
    if not os.path.exists(directory):
        return 0

    deleted_count = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and not item.startswith(".gitkeep"):
            try:
                os.remove(item_path)
                deleted_count += 1
            except Exception:
                pass
    return deleted_count
