"""
rag/embeddings.py
------------------
Thin wrapper around the Gemini embedding API.

Reuses the same Gemini client/setup as teacher.py (no separate API key
handling, no duplicated client code).
"""

import numpy as np
from google.genai import types

from teacher import get_client, GeminiResponseError

EMBEDDING_MODEL = "gemini-embedding-001"

# Gemini's embedding API accepts a batch of texts per call, but very large
# batches are slower to retry on failure - chunking requests keeps things
# simple and robust for a hackathon-sized document.
BATCH_SIZE = 20


def _embed_batch(texts: list, task_type: str) -> list:
    client = get_client()
    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
    except Exception as e:
        raise GeminiResponseError(f"Embedding request failed: {e}")

    if not result.embeddings:
        raise GeminiResponseError("Embedding API returned no vectors.")

    return [np.array(e.values, dtype="float32") for e in result.embeddings]


def embed_chunks(chunk_texts: list) -> np.ndarray:
    """Embed a list of document chunk texts (for indexing). Returns an (N, D) array."""
    vectors = []
    for i in range(0, len(chunk_texts), BATCH_SIZE):
        batch = chunk_texts[i:i + BATCH_SIZE]
        vectors.extend(_embed_batch(batch, task_type="RETRIEVAL_DOCUMENT"))
    return np.vstack(vectors)


def embed_query(query: str) -> np.ndarray:
    """Embed a single search query (for retrieval). Returns a (D,) array."""
    vectors = _embed_batch([query], task_type="RETRIEVAL_QUERY")
    return vectors[0]
