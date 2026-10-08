"""
rag/vector_store.py
--------------------
A small wrapper around a FAISS index that keeps embeddings and their chunk
metadata (text, page, document, chunk_id) together, and exposes one simple
reusable function: retrieve_relevant_chunks(query, top_k).

Phase 4.5 adds serialize_index() / from_serialized() so a previously-built
index can be saved (see database/document_library.py) and reconstructed
later WITHOUT calling the embedding API again - the whole point of "don't
reprocess a PDF that's already been processed".
"""

import faiss
import numpy as np

from rag.embeddings import embed_chunks, embed_query


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product search behaves like cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms


class VectorStore:
    """Holds one document's chunks + FAISS index for the current session."""

    def __init__(self, chunks: list):
        """
        chunks: list of {"text", "page", "document", "chunk_id"} dicts,
        as produced by rag.pdf_processor.chunk_pages().

        This always embeds and builds a fresh index. To reconstruct a
        previously-built store without re-embedding, use
        VectorStore.from_serialized() instead.
        """
        self.chunks = chunks
        vectors = embed_chunks([c["text"] for c in chunks])
        vectors = _normalize(vectors)

        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)

    @classmethod
    def from_serialized(cls, chunks: list, index_bytes: bytes) -> "VectorStore":
        """
        Reconstruct a VectorStore from previously-saved chunks and a
        serialized FAISS index (see serialize_index() below), without
        calling the embedding API again. Bypasses __init__ deliberately -
        that's the whole point, since __init__ always re-embeds.
        """
        store = cls.__new__(cls)
        store.chunks = chunks
        index_array = np.frombuffer(index_bytes, dtype="uint8")
        store.index = faiss.deserialize_index(index_array)
        store.dimension = store.index.d
        return store

    def serialize_index(self) -> bytes:
        """Serialize the FAISS index to raw bytes, suitable for storing in SQLite as a BLOB."""
        return faiss.serialize_index(self.index).tobytes()

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> list:
        """
        Embed `query`, search the FAISS index, and return the top_k most
        relevant chunks (each with its text, page, document, and a
        similarity score), best match first.
        """
        query_vector = embed_query(query)
        query_vector = _normalize(query_vector.reshape(1, -1))

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results
