"""
chunker.py
----------
Splits file content into overlapping chunks for embedding and storage.

Responsibilities:
  - Split file text into chunks of ~512 tokens
  - Overlap consecutive chunks by ~50 tokens to avoid context loss at boundaries
  - Attach full repo metadata + file-level fields to each chunk
  - Skip files that are empty or whitespace-only after content is fetched

Why 512 tokens / 50 token overlap:
  512 tokens is large enough to contain a complete function or logical block
  with surrounding context, but small enough that retrieved chunks stay focused
  on one concept. 50-token overlap prevents meaning loss at chunk boundaries.
  See blueprint Section 2 for full rationale.

Token estimation:
  We use a simple character-based approximation (1 token ≈ 4 chars) rather than
  loading a full tokenizer. This keeps dependencies minimal and is accurate
  enough for chunking purposes — the goal is approximately 512 tokens, not exactly.
"""

from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE_TOKENS    = 512
OVERLAP_TOKENS       = 50
CHARS_PER_TOKEN      = 4   # approximation: 1 token ≈ 4 characters

CHUNK_SIZE_CHARS     = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN    # 2048 chars
OVERLAP_CHARS        = OVERLAP_TOKENS   * CHARS_PER_TOKEN     # 200 chars


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str) -> list[str]:
    """
    Split a text string into overlapping character-window chunks.

    Uses a sliding window of CHUNK_SIZE_CHARS with a step of
    (CHUNK_SIZE_CHARS - OVERLAP_CHARS) so consecutive chunks share
    OVERLAP_CHARS characters of content.

    Empty or whitespace-only input returns an empty list.
    """
    text = text.strip()
    if not text:
        return []

    step = CHUNK_SIZE_CHARS - OVERLAP_CHARS
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE_CHARS
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def chunk_file(
    file: dict,
    repo_metadata: dict,
) -> list[dict]:
    """
    Chunk a single file and attach metadata to each chunk.

    Parameters:
        file            Dict with keys: path, content, ext
                        (as returned by GitHubClient.get_indexable_files)
        repo_metadata   The repo-level metadata dict from metadata_generator.py
                        (repo_name, repo_description, repo_technologies, etc.)

    Returns a list of chunk dicts. Each chunk dict contains everything needed
    for embedding and Deep Lake storage:

    {
        "text":        <the chunk text>,
        "repo_name":   "claimsense",
        "repo_description": "...",
        "repo_technologies": [...],
        "repo_purpose": "web app",
        "repo_language": "python",
        "deployment_url": "https://...",
        "repo_topics": [...],
        "metadata_source": "readme",
        "metadata_confidence": "high",
        "file_path":   "src/parser.py",
        "file_type":   ".py",
        "chunk_index": 0,
        "indexed_at":  "2026-06-10T12:00:00+00:00"
    }

    Returns an empty list if the file content produces no chunks
    (empty file, whitespace only).
    """
    content  = file.get("content", "")
    path     = file.get("path", "")
    ext      = file.get("ext", "")

    raw_chunks = _split_into_chunks(content)
    if not raw_chunks:
        return []

    timestamp = datetime.now(timezone.utc).isoformat()

    result = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk = {
            # The actual content that gets embedded and stored.
            "text": chunk_text,

            # Repo-level metadata — copied from repo_metadata.
            # Every field is present on every chunk so retrieval
            # never needs a second lookup.
            "repo_name":           repo_metadata.get("repo_name"),
            "repo_description":    repo_metadata.get("repo_description"),
            "repo_technologies":   repo_metadata.get("repo_technologies", []),
            "repo_purpose":        repo_metadata.get("repo_purpose"),
            "repo_language":       repo_metadata.get("repo_language"),
            "deployment_url":      repo_metadata.get("deployment_url"),
            "repo_topics":         repo_metadata.get("repo_topics", []),
            "metadata_source":     repo_metadata.get("metadata_source"),
            "metadata_confidence": repo_metadata.get("metadata_confidence"),

            # File-level fields.
            "file_path":    path,
            "file_type":    ext,

            # Chunk position within this file.
            "chunk_index":  idx,

            # Timestamp of when this chunk was indexed.
            "indexed_at":   timestamp,
        }
        result.append(chunk)

    return result


def chunk_repo_files(
    files: list[dict],
    repo_metadata: dict,
) -> list[dict]:
    """
    Chunk all files for a repo and return a flat list of all chunks.

    Parameters:
        files           List of file dicts from GitHubClient.get_indexable_files()
        repo_metadata   Repo-level metadata from metadata_generator.py

    Returns a flat list of all chunk dicts across all files.
    Files that produce no chunks (empty, binary, whitespace) are silently skipped.
    """
    all_chunks = []

    for file in files:
        chunks = chunk_file(file, repo_metadata)
        if chunks:
            all_chunks.extend(chunks)

    repo_name  = repo_metadata.get("repo_name", "unknown")
    file_count = len(files)
    chunk_count = len(all_chunks)
    print(f"  [chunker] {repo_name}: {file_count} files → {chunk_count} chunks")

    return all_chunks
