"""Generate stub PDFs for the Try Demo feature.

Run once after cloning the repo:
    python scripts/create_demo_papers.py
"""
import os

PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "papers")

PAPERS = {
    "Lewis_et_al_2020_RAG.pdf": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "Karpukhin_et_al_2020_DPR.pdf": "Dense Passage Retrieval for Open-Domain Question Answering",
}


def make_stub_pdf(title: str) -> bytes:
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj\n<</Length 44>>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (" + title.encode() + b") Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype=Type1/BaseFont=Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n429\n%%EOF"
    )


if __name__ == "__main__":
    os.makedirs(PAPERS_DIR, exist_ok=True)
    for filename, title in PAPERS.items():
        path = os.path.join(PAPERS_DIR, filename)
        with open(path, "wb") as f:
            f.write(make_stub_pdf(title))
        print(f"Created: {filename}")
    print("Done. You can now run the app and click Try Demo.")
