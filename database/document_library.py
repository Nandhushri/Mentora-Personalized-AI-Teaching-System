"""
database/document_library.py
------------------------------
The DOCUMENT LIBRARY (Phase 4.5) - persists uploaded PDFs so a student can
revisit them later without re-uploading, and so Revision Mode can reuse the
existing extracted text and FAISS index instead of reprocessing the file.

Kept deliberately separate from database/student_db.py's tables (STUDENT
MEMORY): this module owns "what material has this student uploaded and
what does it contain", not "what does this student know/struggle with".
The only link between the two is a plain document_id foreign-key-style
value stored on the student's concept_progress rows (see student_db.py) -
the two are joined by that id when needed, never merged into one table.

What's stored per document:
- the raw PDF bytes, on disk (so it can be viewed/downloaded again)
- its extracted, chunked text, in the document_chunks table
- its FAISS index, serialized to a BLOB in the documents table
  (see rag.vector_store.VectorStore.serialize_index /  from_serialized)

Reconstructing a document's VectorStore from this data never calls the
embedding API again - that's the whole point of this module.
"""

import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from database.student_db import DB_PATH, StudentDBError
from rag.vector_store import VectorStore

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_filename(name: str) -> str:
    """Strip anything that isn't safe for a filesystem path."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:150]


@contextmanager
def _connect():
    """Same pattern as student_db._connect() - same database FILE, separate TABLES."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        _create_tables(conn)
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        raise StudentDBError(f"Database error: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            page_count INTEGER,
            chunk_count INTEGER,
            processed_status TEXT NOT NULL DEFAULT 'processed',
            file_path TEXT,
            vector_index_blob BLOB,
            UNIQUE (student_id, filename)
        );

        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            page INTEGER NOT NULL
        );
    """)


# ---------------------------------------------------------------------------
# Saving a newly-processed document
# ---------------------------------------------------------------------------

def save_document(student_id: str, filename: str, page_count: int,
                   pdf_bytes: bytes, chunks: list, vector_store: VectorStore) -> int:
    """
    Persist a freshly-processed document: its raw bytes (to disk, for later
    viewing/downloading), its chunks, and its FAISS index (as a BLOB) - so
    it never needs to be extracted or embedded again. Returns document_id.
    """
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents
                (student_id, filename, upload_date, page_count, chunk_count, processed_status)
            VALUES (?, ?, ?, ?, ?, 'processed')
            ON CONFLICT (student_id, filename) DO UPDATE SET
                upload_date = excluded.upload_date,
                page_count = excluded.page_count,
                chunk_count = excluded.chunk_count,
                processed_status = 'processed'
            """,
            (student_id, filename, _now(), page_count, len(chunks)),
        )
        row = conn.execute(
            "SELECT document_id FROM documents WHERE student_id = ? AND filename = ?",
            (student_id, filename),
        ).fetchone()
        document_id = row["document_id"]

        # Clear out any previously-saved chunks for this document_id (relevant on re-save/replace).
        conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        conn.executemany(
            "INSERT INTO document_chunks (document_id, chunk_index, text, page) VALUES (?, ?, ?, ?)",
            [(document_id, i, c["text"], c["page"]) for i, c in enumerate(chunks)],
        )

        student_dir = os.path.join(DOCUMENTS_DIR, _safe_filename(student_id))
        os.makedirs(student_dir, exist_ok=True)
        file_path = os.path.join(student_dir, f"{document_id}_{_safe_filename(filename)}")
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)

        conn.execute(
            "UPDATE documents SET file_path = ?, vector_index_blob = ? WHERE document_id = ?",
            (file_path, vector_store.serialize_index(), document_id),
        )

        return document_id


# ---------------------------------------------------------------------------
# Retrieval - always scoped by student_id, so one student never sees
# another's documents (Phase 4.5 spec section 3).
# ---------------------------------------------------------------------------

def get_document_by_filename(student_id: str, filename: str) -> dict:
    """Look up an already-processed document by name for this student, or {} if none exists."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE student_id = ? AND filename = ? AND processed_status = 'processed'",
            (student_id, filename),
        ).fetchone()
        return dict(row) if row else {}


def get_documents_for_student(student_id: str) -> list:
    """This student's document library, most recently uploaded first. Never another student's."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT document_id, student_id, filename, upload_date, page_count, chunk_count, processed_status "
            "FROM documents WHERE student_id = ? ORDER BY upload_date DESC",
            (student_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_document(document_id: int, student_id: str) -> dict:
    """
    Fetch one document's full row (including its file_path and index
    blob), but ONLY if it belongs to student_id - an ownership check, not
    just a lookup, so a guessed/wrong document_id can never leak another
    student's material.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ? AND student_id = ?",
            (document_id, student_id),
        ).fetchone()
        return dict(row) if row else {}


def get_document_chunks(document_id: int) -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT text, page FROM document_chunks WHERE document_id = ? ORDER BY chunk_index ASC",
            (document_id,),
        ).fetchall()
        return [{"text": r["text"], "page": r["page"], "document": None, "chunk_id": i}
                for i, r in enumerate(rows)]


def load_vector_store(document_id: int, student_id: str) -> VectorStore:
    """
    Reconstruct this document's VectorStore from its saved chunks + FAISS
    index - WITHOUT calling the embedding API again. Returns None if the
    document doesn't exist, doesn't belong to this student, or has no
    saved index yet.
    """
    document = get_document(document_id, student_id)
    if not document or not document.get("vector_index_blob"):
        return None
    chunks = get_document_chunks(document_id)
    if not chunks:
        return None
    for c in chunks:
        c["document"] = document["filename"]
    return VectorStore.from_serialized(chunks, document["vector_index_blob"])


def get_pdf_bytes(document_id: int, student_id: str) -> bytes:
    """Read back the original uploaded PDF bytes, for viewing/downloading. None if unavailable."""
    document = get_document(document_id, student_id)
    file_path = document.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return f.read()
