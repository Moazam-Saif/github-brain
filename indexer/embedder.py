"""
embedder.py
-----------
Converts chunk text into embedding vectors using Gemini text-embedding-004.

Responsibilities:
  - Embed a list of chunk texts via Gemini's embedding API
  - Respect the free tier rate limit (100 requests/min)
  - Batch requests to stay within limits with automatic throttling
  - Return vectors aligned 1:1 with the input chunks list

Gemini text-embedding-004:
  - Output dimension: 768 floats per vector
  - Free tier: 100 requests/minute
  - Each request embeds one text (no batch endpoint on free tier)
  - task_type "RETRIEVAL_DOCUMENT" is used at index time
  - task_type "RETRIEVAL_QUERY"    is used at query time (in retriever.py)

Why task_type matters:
  Gemini's embedding model is asymmetric — it produces slightly different
  vector spaces depending on whether the text is a document being stored
  or a query being searched. Using the correct task_type for each case
  improves retrieval accuracy.
"""

import os
import time
from google import genai
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL   = "models/text-embedding-004"
EMBEDDING_DIM     = 768        # output dimension for text-embedding-004

# Free tier: 100 requests/min → 1 request per 0.6s to stay safely under limit.
# We use 0.65s to add a small safety buffer.
REQUEST_INTERVAL_SECONDS = 0.65

# Batch size: how many chunks to embed before printing a progress update.
PROGRESS_BATCH_SIZE = 20


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _configure_genai():
    """Return a configured Gemini client."""
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# Single embedding call
# ---------------------------------------------------------------------------

def _embed_single(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
    max_retries: int = 3,
) -> Optional[list[float]]:
    """
    Embed a single text string and return its vector.

    Returns a list of 768 floats, or None if all retries fail.

    task_type options:
      "RETRIEVAL_DOCUMENT" → used when indexing (storing chunks)
      "RETRIEVAL_QUERY"    → used when searching (embedding user questions)
    """
    client = _configure_genai()

    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config={"task_type": task_type},
            )
            return result.embeddings[0].values

        except Exception as e:
            error_str = str(e)

            # Rate limit — back off and retry.
            if "429" in error_str or "quota" in error_str.lower():
                wait = 60 * (attempt + 1)
                print(f"  [embedder] Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
                continue

            # Resource exhausted or server error — short wait then retry.
            if "500" in error_str or "503" in error_str:
                wait = 10 * (attempt + 1)
                print(f"  [embedder] Server error (attempt {attempt + 1}). Waiting {wait}s...")
                time.sleep(wait)
                continue

            # Unrecoverable error.
            print(f"  [embedder] Failed to embed text (attempt {attempt + 1}): {e}")
            break

    return None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed the text of each chunk and attach the vector as "embedding".

    Parameters:
        chunks  List of chunk dicts from chunker.py. Each must have a "text" key.

    Returns the same list with "embedding" added to each chunk dict.
    Chunks where embedding fails (None returned) are filtered out of the result
    so the caller never receives a chunk without a valid vector.

    Progress is logged every PROGRESS_BATCH_SIZE chunks.
    """
    total       = len(chunks)
    embedded    = []
    failed      = 0

    print(f"  [embedder] Embedding {total} chunks...")

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()

        if not text:
            failed += 1
            continue

        vector = _embed_single(text, task_type="RETRIEVAL_DOCUMENT")

        if vector is None:
            failed += 1
            continue

        chunk["embedding"] = vector
        embedded.append(chunk)

        # Progress update.
        if (i + 1) % PROGRESS_BATCH_SIZE == 0 or (i + 1) == total:
            print(f"  [embedder] {i + 1}/{total} embedded, {failed} failed")

        # Rate limit throttle — stay under 100 req/min on free tier.
        time.sleep(REQUEST_INTERVAL_SECONDS)

    print(f"  [embedder] Done. {len(embedded)} embedded, {failed} failed/skipped.")
    return embedded


def embed_query(question: str) -> Optional[list[float]]:
    """
    Embed a user query for similarity search at query time.
    Uses task_type="RETRIEVAL_QUERY" — asymmetric counterpart to RETRIEVAL_DOCUMENT.
    """
    return _embed_single(question, task_type="RETRIEVAL_QUERY")
