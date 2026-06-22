"""
Parses docs/*.pdf, chunks by page, embeds with OpenAI, saves data/rag_index.json.
Run once after generate_pdfs.py:  python ingest.py
Requires OPENAI_API_KEY in environment or .env file.
"""

import json
import os
import re

import numpy as np
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DOCS_DIR = "docs"
OUT_FILE = os.path.join("data", "rag_index.json")
EMBED_MODEL = "text-embedding-3-small"

# Map doc prefix to asset type and language
DOC_META = {
    "MAN-PM": {"asset_type": "Point Machine",  "language": "EN"},
    "MAN-TC": {"asset_type": "Track Circuit",   "language": "EN"},
    "MAN-AC": {"asset_type": "Axle Counter",    "language": "EN"},
}
LANG_OVERRIDE = {
    "MAN-PM-004": "DE",
}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def extract_pages(pdf_path: str) -> list[dict]:
    """Return list of {page, text} dicts, skipping the title page (page 1)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i == 0:
                continue  # skip title page
            text = page.extract_text() or ""
            # strip header/footer noise lines (short lines at very top/bottom)
            lines = text.splitlines()
            # drop lines that are pure header or footer boilerplate
            cleaned = [
                ln for ln in lines
                if not re.match(
                    r"^(Siemens Mobility|Page \d+|CONFIDENTIAL|Copyright)", ln.strip()
                )
            ]
            text = "\n".join(cleaned).strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages


def embed(texts: list[str]) -> list[list[float]]:
    """Batch embed texts using OpenAI."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    pdf_files = sorted(
        f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")
    )
    if not pdf_files:
        print("No PDFs found in docs/. Run generate_pdfs.py first.")
        return

    chunks = []
    for filename in pdf_files:
        doc_id = filename.replace(".pdf", "")
        pdf_path = os.path.join(DOCS_DIR, filename)

        # Determine metadata
        prefix = "-".join(doc_id.split("-")[:2])
        meta = DOC_META.get(prefix, {"asset_type": "Unknown", "language": "EN"})
        language = LANG_OVERRIDE.get(doc_id, meta["language"])

        pages = extract_pages(pdf_path)
        print(f"  {doc_id}: {len(pages)} content pages")

        for page_info in pages:
            chunks.append({
                "doc_id": doc_id,
                "asset_type": meta["asset_type"],
                "language": language,
                "page": page_info["page"],
                "text": page_info["text"],
                "embedding": None,  # filled below
            })

    # Embed all chunks in one batch call (< 2048 items, fine for this scale)
    print(f"\nEmbedding {len(chunks)} chunks with {EMBED_MODEL}...")
    texts = [c["text"] for c in chunks]
    embeddings = embed(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    with open(OUT_FILE, "w") as f:
        json.dump(chunks, f)

    print(f"Saved {len(chunks)} chunks to {OUT_FILE}")
    print("RAG index ready. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
