"""
rag/pdf_processor.py
---------------------
Turns an uploaded PDF into clean, page-aware text chunks ready for embedding.

Pipeline: extract text page-by-page -> clean it -> split into overlapping
chunks that keep track of which page (and which document) they came from.
"""

import re

import pymupdf as fitz  # PyMuPDF (the "import fitz" alias is deprecated; this is the current form)


class PDFProcessingError(Exception):
    """Raised when a PDF can't be read or has no usable text."""
    pass


def extract_pages(pdf_bytes: bytes) -> list:
    """
    Extract text from a PDF, page by page.
    Returns a list of {"page": <1-indexed page number>, "text": "..."}.
    Pages with no extractable text are still included (with empty text)
    so page numbers are never silently skipped.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise PDFProcessingError(f"Couldn't open this file as a PDF: {e}")

    pages = []
    for i, page in enumerate(doc):
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        pages.append({"page": i + 1, "text": text or ""})
    doc.close()

    total_chars = sum(len(p["text"].strip()) for p in pages)
    if total_chars < 20:
        raise PDFProcessingError(
            "This PDF doesn't seem to contain extractable text (it may be a "
            "scanned image). Try a different PDF with selectable text."
        )

    return pages


def clean_text(text: str) -> str:
    """Light cleanup: collapse whitespace, drop stray control characters."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list, document_name: str, chunk_size: int = 1000, overlap: int = 150) -> list:
    """
    Split cleaned page text into overlapping character-based chunks.
    Each chunk keeps its source page number, document name, and a chunk_id
    so retrieved results can always be traced back to "Page N" of the doc.

    A chunk never spans multiple pages, which keeps page references accurate
    (a chunk that started on page 3 will never secretly contain page 4 text).
    """
    chunks = []
    chunk_id = 0

    for page_info in pages:
        text = clean_text(page_info["text"])
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "text": piece,
                    "page": page_info["page"],
                    "document": document_name,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1
            if end == len(text):
                break
            start = end - overlap  # step forward, keeping some overlap

    if not chunks:
        raise PDFProcessingError("No usable text chunks could be created from this PDF.")

    return chunks
