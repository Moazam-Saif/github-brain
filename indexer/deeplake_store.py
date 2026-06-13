"""
deeplake_store.py
-----------------
All Deep Lake read/write operations for the GitHub Brain project.

PURPOSE:
  This is the database layer. It owns all communication with Deep Lake —
  creating the dataset, writing chunks into it, deleting chunks from it,
  and reading chunks back out during search.

HOW IT WORKS:
  Deep Lake stores three tensors per row (one row = one chunk):
    embedding → 768 float32 numbers representing the chunk's meaning
    text      → the raw code/markdown content of the chunk
    metadata  → JSON string containing repo info, file path, chunk index, etc.

  At query time we load ALL embeddings into memory as a numpy matrix in one
  network call, then do a single matrix multiplication to score every chunk
  against the query vector simultaneously. This is O(1) network calls
  regardless of dataset size, making it fast enough for free tier use.

  The old approach loaded one embedding per network call in a loop —
  O(N) network calls, unusably slow for anything beyond a handful of chunks.

SEARCH MODES:
  similarity_search(query_vector, repo_name=None)  → semantic search across all repos
  similarity_search(query_vector, repo_name="X")   → semantic search within one repo
  list_all_repos()                                  → return metadata summaries, no vectors

WHY ONE DATASET FOR ALL REPOS:
  Keeping everything in one dataset (hub://ORG/github_brain) lets us do
  cross-repo search trivially — no joining, no federation. Filtering to a
  specific repo is done after loading metadata, which is cheap.
"""

import os
import json
import numpy as np
import deeplake
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset path
# ---------------------------------------------------------------------------

def _get_dataset_path() -> str:
    org = os.getenv("ACTIVELOOP_ORG")
    if not org:
        raise ValueError("ACTIVELOOP_ORG is required but was not provided.")
    return f"hub://{org}/github_brain_v5"


def _get_token() -> str:
    token = os.getenv("ACTIVELOOP_TOKEN")
    if not token:
        raise ValueError("ACTIVELOOP_TOKEN is required but was not provided.")
    return token


# ---------------------------------------------------------------------------
# Dataset initialization
# ---------------------------------------------------------------------------

def get_or_create_dataset() -> deeplake.Dataset:
    path  = _get_dataset_path()
    token = _get_token()

    ds = deeplake.dataset(path, token=token)

    if "embedding" not in ds.tensors:
        print(f"[deeplake] Creating tensors in dataset: {path}")
        ds.create_tensor("embedding", htype="embedding",
                         dtype="float32", sample_compression=None)
        ds.create_tensor("text",      htype="text")
        ds.create_tensor("metadata",  htype="json")
        print(f"[deeplake] Dataset ready.")
    else:
        print(f"[deeplake] Opened existing dataset: {path}")

    return ds


# ---------------------------------------------------------------------------
# Write: store chunks
# ---------------------------------------------------------------------------

def store_chunks(chunks: list[dict]) -> int:
    """
    Append a list of embedded chunks to the dataset.

    Each chunk must have:
      "embedding" → list[float] of length 768
      "text"      → str

    All other keys in the chunk dict become the metadata JSON.
    Chunks missing an embedding are silently skipped.

    Returns the number of chunks successfully stored.
    Does NOT overwrite existing data — always appends.
    """
    if not chunks:
        print("  [deeplake] No chunks to store.")
        return 0

    ds    = get_or_create_dataset()
    count = 0

    with ds:
        for chunk in chunks:
            embedding = chunk.get("embedding")
            text      = chunk.get("text", "")
            if embedding is None:
                continue
            metadata = {k: v for k, v in chunk.items()
                        if k not in ("embedding", "text")}
            ds.append({
                "embedding": np.array(embedding, dtype=np.float32),
                "text":      text,
                "metadata":  json.dumps(metadata),
            })
            count += 1

    print(f"  [deeplake] Stored {count} chunks.")
    return count


# ---------------------------------------------------------------------------
# Write: delete repo chunks
# ---------------------------------------------------------------------------

def delete_repo_chunks(repo_name: str) -> int:
    """
    Delete every chunk belonging to a specific repo.
    Called before re-indexing a repo to avoid duplicates.

    Deep Lake has no filtered delete, so we:
      1. Scan all metadata to find matching indices
      2. Delete those indices in reverse order

    Reverse order is critical: deleting index 5 shifts everything above it
    down by one, so if we deleted forwards we'd skip entries or hit wrong ones.

    Returns the number of chunks deleted.
    """
    ds    = get_or_create_dataset()
    total = len(ds)

    if total == 0:
        print(f"  [deeplake] Dataset is empty, nothing to delete for: {repo_name}")
        return 0

    indices = []
    for i in range(total):
        try:
            raw = ds.metadata[i].numpy()
            # Deep Lake can return bytes, str, or ndarray — normalise to str
            if hasattr(raw, 'item'):
                raw = raw.item()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            meta = json.loads(str(raw))
            if meta.get("repo_name") == repo_name:
                indices.append(i)
        except Exception:
            continue

    if not indices:
        print(f"  [deeplake] No existing chunks found for repo: {repo_name}")
        return 0

    for idx in sorted(indices, reverse=True):
        ds.pop(idx)

    print(f"  [deeplake] Deleted {len(indices)} chunks for repo: {repo_name}")
    return len(indices)


# ---------------------------------------------------------------------------
# Read: load all data (internal)
# ---------------------------------------------------------------------------

def _load_all(repo_name: Optional[str] = None) -> tuple[np.ndarray, list[str], list[dict]]:
    """
    Load all embeddings, texts, and metadata from the dataset in one pass.

    This is the core of the fast search approach:
      - One call to ds.embedding.numpy() loads ALL vectors as a matrix (N, 768)
      - One call to ds.metadata.numpy() loads ALL metadata strings
      - No per-row network calls

    If repo_name is given, filters to only that repo's rows before returning.

    Returns:
        embeddings  np.ndarray shape (N, 768)
        texts       list of N strings
        metas       list of N metadata dicts
    """
    ds    = get_or_create_dataset()
    total = len(ds)

    if total == 0:
        return np.array([]), [], []

    # Load everything in bulk.
    try:
        all_embeddings = ds.embedding.numpy()   # shape: (total, 768)
        all_texts      = [str(ds.text[i].numpy()) for i in range(total)]
        all_metas      = []
        for i in range(total):
            try:
                raw = ds.metadata[i].numpy()
                if hasattr(raw, 'item'):
                    raw = raw.item()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                all_metas.append(json.loads(str(raw)))
            except Exception:
                all_metas.append({})
    except Exception as e:
        print(f"  [deeplake] Error loading data: {e}")
        return np.array([]), [], []

    if repo_name is None:
        return all_embeddings, all_texts, all_metas

    # Filter to the requested repo.
    indices = [i for i, m in enumerate(all_metas)
               if m.get("repo_name") == repo_name]

    if not indices:
        return np.array([]), [], []

    filtered_embeddings = all_embeddings[indices]
    filtered_texts      = [all_texts[i] for i in indices]
    filtered_metas      = [all_metas[i] for i in indices]

    return filtered_embeddings, filtered_texts, filtered_metas


# ---------------------------------------------------------------------------
# Read: similarity search
# ---------------------------------------------------------------------------

def similarity_search(
    query_vector: list[float],
    top_k: int = 4,
    repo_name: Optional[str] = None,
) -> list[dict]:
    """
    Find the top-k chunks most semantically similar to a query vector.

    HOW IT WORKS:
      1. Load all embeddings as a matrix M of shape (N, 768) — one network call
      2. Normalize M and the query vector to unit length
      3. Compute M @ query_vector — one matrix multiply, gives N cosine scores
      4. argsort scores, take top_k indices
      5. Build result dicts from those indices

    This replaces the old per-row loop which made N separate network calls.
    For 1000 chunks: old = ~50 seconds, new = ~1-2 seconds.

    Parameters:
        query_vector  768-dim float list from embedder.embed_query()
        top_k         Max number of results to return
        repo_name     If given, search only within this repo.
                      If None, search across all repos.

    Returns list of dicts sorted by similarity score descending:
    [{"text": "...", "score": 0.91, "repo_name": "...", "file_path": "...", ...}]
    """
    embeddings, texts, metas = _load_all(repo_name=repo_name)

    if len(embeddings) == 0:
        print(f"  [deeplake] No chunks found"
              + (f" for repo: {repo_name}" if repo_name else "") + ".")
        return []

    query_np = np.array(query_vector, dtype=np.float32)

    # Normalize query vector.
    query_norm = np.linalg.norm(query_np)
    if query_norm == 0:
        return []
    query_np = query_np / query_norm

    # Normalize all embedding rows.
    # norms shape: (N,) — one norm per row
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)   # avoid divide by zero
    normalized = embeddings / norms           # shape: (N, 768)

    # One matrix multiply → N cosine similarity scores.
    scores = normalized @ query_np            # shape: (N,)

    # Get top_k indices (argsort ascending, take last top_k reversed).
    actual_k    = min(top_k, len(scores))
    top_indices = np.argsort(scores)[-actual_k:][::-1]

    results = []
    for idx in top_indices:
        result = {
            "text":  texts[idx],
            "score": float(scores[idx]),
            **metas[idx],
        }
        results.append(result)

    return results


def similarity_search_per_repo(
    query_vector: list[float],
    max_repos: int = 10,
) -> list[dict]:
    """
    Return the single best-matching chunk per repo for a given query.

    WHY THIS EXISTS:
      Regular similarity_search(top_k=6) returns the 6 highest-scoring
      chunks globally. For "do any repos implement rate limiting?", those
      6 chunks might all be from 1-2 repos, completely missing other repos
      that also have rate limiting but scored 7th, 8th, 9th overall.

      This function guarantees every repo gets exactly one representative
      chunk — its highest-scoring one — so Gemini can make a complete
      assessment across all repos, not just the ones that happened to
      dominate the global top-k.

    HOW IT WORKS:
      1. Score ALL chunks against the query (same batch matrix multiply)
      2. For each repo, find the index of its highest-scoring chunk
      3. Return those best-per-repo chunks sorted by score descending

    Parameters:
        query_vector  768-dim float list
        max_repos     Cap on how many repos to return (default 10).
                      In practice limited by how many repos are indexed.

    Returns list of chunk dicts, one per repo, sorted by score descending:
    [{"text": "...", "score": 0.91, "repo_name": "skillswap", ...}, ...]
    """
    embeddings, texts, metas = _load_all(repo_name=None)

    if len(embeddings) == 0:
        return []

    query_np   = np.array(query_vector, dtype=np.float32)
    query_norm = np.linalg.norm(query_np)
    if query_norm == 0:
        return []
    query_np = query_np / query_norm

    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms      = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    scores     = normalized @ query_np   # shape (N,)

    # For each repo, find the index of its best-scoring chunk.
    best: dict[str, tuple[int, float]] = {}   # repo_name → (index, score)
    for i, meta in enumerate(metas):
        repo = meta.get("repo_name", "unknown")
        s    = float(scores[i])
        if repo not in best or s > best[repo][1]:
            best[repo] = (i, s)

    # Build result list sorted by score descending.
    results = []
    for repo, (idx, score) in sorted(best.items(), key=lambda x: -x[1][1]):
        results.append({
            "text":  texts[idx],
            "score": round(score, 4),
            **metas[idx],
        })

    results = results[:max_repos]
    repos_str = " | ".join(f"{r['repo_name']} ({r['score']:.3f})"
                           for r in results)
    print(f"  [deeplake] Per-repo best: {repos_str}")
    return results


# ---------------------------------------------------------------------------
# Read: aggregated similarity search (repo-level scoring)
# ---------------------------------------------------------------------------

def similarity_search_aggregated(
    query_vector: list[float],
    top_repos: int = 3,
    chunks_per_repo: int = 3,
    candidate_k: int = 50,
) -> list[dict]:
    """
    Score repos as a whole rather than returning individual top-k chunks.

    WHY THIS EXISTS:
      Regular similarity_search returns the top-k chunks globally. For
      comparative questions ("which repo has the best auth?") this is broken —
      a repo with many auth-related chunks floods the top-k slots and crowds
      out other repos entirely, even if those repos have equally good auth.

      This function fixes that by aggregating chunk scores per repo first,
      then ranking repos, then returning the best chunks from the top repos.

    HOW IT WORKS:
      1. Search broadly — fetch candidate_k chunks (default 50) globally
      2. Group chunks by repo_name
      3. For each repo, compute a repo_score = average of its top-3 chunk scores
         (top-N average is more robust than max — penalises repos that only
         have one fluke high-scoring chunk)
      4. Rank repos by repo_score descending
      5. For each of the top `top_repos` repos, take its best `chunks_per_repo`
         chunks as the context to send to Gemini

    WHY TOP-3 AVERAGE (not max, not all-chunk average):
      - Max score: too easy to game — one flukey high-scoring chunk wins
      - All-chunk average: penalises repos that have many chunks (larger repos
        have more off-topic chunks pulling the average down)
      - Top-3 average: fair across repo sizes, robust against flukes

    Parameters:
        query_vector      768-dim float list from embedder.embed_query()
        top_repos         How many repos to return after ranking (default 3)
        chunks_per_repo   How many chunks to include per top repo (default 3)
        candidate_k       How many raw chunks to fetch before aggregating (default 50)
                          Should be large enough to cover all repos meaningfully.
                          Rule of thumb: candidate_k >= top_repos × avg_chunks_per_repo × 3

    Returns:
        List of repo result dicts, sorted by repo_score descending:
        [
          {
            "repo_name":       "claimsense",
            "repo_score":      0.89,          ← top-3 average chunk score
            "repo_rank":       1,
            "chunks":          [              ← best chunks_per_repo chunks
              {"text": "...", "score": 0.91, "file_path": "...", ...},
              {"text": "...", "score": 0.89, "file_path": "...", ...},
              {"text": "...", "score": 0.87, "file_path": "...", ...},
            ],
            "repo_description": "...",        ← from metadata
            "repo_technologies": [...],
            "deployment_url":   "...",
          },
          ...
        ]
    """
    # Step 1: Broad candidate search — no repo filter, high k.
    candidates = similarity_search(
        query_vector=query_vector,
        top_k=candidate_k,
        repo_name=None,
    )

    if not candidates:
        return []

    # Step 2: Group chunks by repo.
    repo_chunks: dict[str, list[dict]] = {}
    for chunk in candidates:
        name = chunk.get("repo_name", "unknown")
        repo_chunks.setdefault(name, []).append(chunk)

    # Step 3: Score each repo using top-N average.
    TOP_N = 3
    repo_scores = []
    for name, chunks in repo_chunks.items():
        # Chunks are already sorted descending by score from similarity_search.
        top_n_scores = [c["score"] for c in chunks[:TOP_N]]
        repo_score   = sum(top_n_scores) / len(top_n_scores)
        repo_scores.append((name, repo_score, chunks))

    # Step 4: Rank repos by score descending.
    repo_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 5: Build result dicts for top_repos repos.
    results = []
    for rank, (name, score, chunks) in enumerate(repo_scores[:top_repos], start=1):
        # Take best chunks_per_repo chunks for this repo.
        best_chunks = chunks[:chunks_per_repo]

        # Pull repo-level metadata from the first chunk (all chunks from a repo
        # carry identical repo-level metadata fields).
        first = chunks[0]
        result = {
            "repo_name":         name,
            "repo_score":        round(score, 4),
            "repo_rank":         rank,
            "chunks":            best_chunks,
            "repo_description":  first.get("repo_description"),
            "repo_technologies": first.get("repo_technologies", []),
            "repo_purpose":      first.get("repo_purpose"),
            "repo_language":     first.get("repo_language"),
            "deployment_url":    first.get("deployment_url"),
        }
        results.append(result)

    # Log the ranking.
    ranking_str = " | ".join(
        f"{r['repo_name']} ({r['repo_score']:.3f})" for r in results
    )
    print(f"  [deeplake] Repo ranking: {ranking_str}")

    return results


# ---------------------------------------------------------------------------
# Read: list all repos from metadata
# ---------------------------------------------------------------------------

def list_all_repos() -> list[dict]:
    """
    Return one metadata summary dict per unique repo in the dataset.

    Used by:
      - query engine for list_repos questions (no vector search)
      - query engine for cross_repo_metadata questions (filter by field)
      - CLI to show what's indexed

    Does not load embeddings — only reads metadata, which is fast.

    Returns list sorted by repo_name:
    [
      {
        "repo_name":           "claimsense",
        "repo_description":    "...",
        "repo_technologies":   ["python", "flask", ...],
        "repo_purpose":        "web app",
        "repo_language":       "python",
        "deployment_url":      "https://...",
        "repo_topics":         [...],
        "metadata_confidence": "high",
      },
      ...
    ]
    """
    _, _, all_metas = _load_all(repo_name=None)

    seen = {}
    for meta in all_metas:
        repo_name = meta.get("repo_name")
        if repo_name and repo_name not in seen:
            seen[repo_name] = {
                "repo_name":           repo_name,
                "repo_description":    meta.get("repo_description"),
                "repo_technologies":   meta.get("repo_technologies", []),
                "repo_purpose":        meta.get("repo_purpose"),
                "repo_language":       meta.get("repo_language"),
                "deployment_url":      meta.get("deployment_url"),
                "repo_topics":         meta.get("repo_topics", []),
                "metadata_confidence": meta.get("metadata_confidence"),
                # Richer fields for metadata-based routing.
                "has_authentication":   meta.get("has_authentication", False),
                "has_database":         meta.get("has_database", False),
                "database_type":        meta.get("database_type"),
                "has_api":              meta.get("has_api", False),
                "api_style":            meta.get("api_style"),
                "has_frontend":         meta.get("has_frontend", False),
                "frontend_framework":   meta.get("frontend_framework"),
                "architecture_pattern": meta.get("architecture_pattern"),
                "key_features":         meta.get("key_features", []),
                "external_services":    meta.get("external_services", []),
                "has_tests":            meta.get("has_tests", False),
            }

    return sorted(seen.values(), key=lambda r: r["repo_name"])
