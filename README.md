# PaperDiff

Compare research papers side by side.

Upload 2–4 PDFs and PaperDiff will read them, break them into evidence-grounded chunks, and score each paper across 6 criteria — methodology, evidence, novelty, clarity, scholarly grounding, and ethics. You get a verdict with citations back to the exact pages and sections.

## What it does

- **Compare Papers** — side-by-side scoring across 6 criteria with an overall winner and confidence rating
- **Ask Questions** — ask anything about your papers and get answers grounded in the actual text
- **2–4 papers** — supports pairwise and multi-paper comparison
- **Evidence-backed** — every claim links back to page numbers and section headers
- **Works offline** — runs without an API key using heuristic scoring (add an API key for LLM-powered analysis)

## Quick start

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Open `http://localhost:8502`, upload your papers, and hit **Run Comparison**.

Or click **Try Demo** to see it in action with two sample papers.

## Configuration

Set your API key in `.env` (for LLM-powered analysis):

```
OPENROUTER_API_KEY=your-key-here
```

Without a key, the app still works — it uses heuristic scoring instead of calling an LLM.

## Project structure

```
src/
  app.py              # Streamlit UI
  rag/
    ingest.py          # PDF parsing and chunking
    embed.py           # Vector embeddings with FAISS
    retrieve.py        # Section-aware retrieval
    extract.py         # LLM scoring and synthesis
tests/
  test_rag.py          # Unit tests
  test_final.py        # E2E browser test (requires running server)
```

## Running tests

```bash
# Unit tests
python -m pytest tests/test_rag.py

# E2E test (requires server running on port 8502)
python tests/test_final.py
```

## Tech stack

Streamlit · PyMuPDF · sentence-transformers · FAISS · OpenRouter API

## License

MIT
