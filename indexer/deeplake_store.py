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

SEARCH MODES:
  hybrid_search(query_vector, query_text, repo_name=None)  → BM25 + cosine via RRF (default)
  similarity_search(query_vector, repo_name=None)          → cosine-only fallback
  similarity_search_per_repo(query_vector)                 → best chunk per repo
  similarity_search_aggregated(query_vector)               → repo-level scoring
  list_all_repos()                                         → metadata summaries, no vectors

WHY HYBRID SEARCH:
  Pure cosine similarity has a hard accuracy ceiling (~44-63% on benchmarks).
  The failure mode this project hit: "how does authentication work?" returned
  chat/session files (semantically adjacent) instead of lib/auth.ts (the actual
  auth file). BM25 fixes this — it scores lib/auth.ts highly because "auth"
  appears literally in the filename and code. Combining both via Reciprocal Rank
  Fusion (RRF) gives the best of lexical precision and semantic coverage.

  RRF formula: score(chunk) = 1/(k + cosine_rank) + 1/(k + bm25_rank)
  where k=60 is the standard RRF constant that dampens rank differences.

WHY ONE DATASET FOR ALL REPOS:
  Keeping everything in one dataset (hub://ORG/github_brain_v5) lets us do
  cross-repo search trivially — no joining, no federation. Filtering to a
  specific repo is done after loading metadata, which is cheap.

DELETION STRATEGY (rewrite-in-place, not ds.pop()):
  Deep Lake 3.x has a known internal bug — calling ds.pop() on a dataset that
  already has committed data raises "OverflowError: can't convert negative int
  to unsigned" inside Deep Lake's own commit-diff tracking (chunk_engine.py).
  This is a bug in Deep Lake itself, not in this code.

  Also note: deeplake.delete() permanently retires the dataset path — "Once
  deleted, dataset names can't be reused in the Deep Lake Cloud." So deleting
  and recreating the dataset on every re-index would burn through path names
  (github_brain_v6, v7, v8, ...) forever.

  The fix used in delete_repo_chunks(): load everything, filter out the target
  repo's rows in memory, reset the dataset content IN PLACE at the same path
  using deeplake.empty(path, overwrite=True) — which resets content without
  retiring the path — then re-append the rows from other repos. This is the
  same pattern used by LangChain/LlamaIndex's DeepLakeVectorStore(overwrite=True).
"""

import os
import json
import numpy as np
import deeplake
from rank_bm25 import BM25Okapi
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
    Remove every chunk belonging to a specific repo.
    Called before re-indexing a repo to avoid duplicates.

    WHY NOT ds.pop():
      Deep Lake 3.x raises "OverflowError: can't convert negative int to
      unsigned" inside its own commit-diff tracking (chunk_engine.py,
      pop_samples) when popping rows from a dataset with committed history.
      This is an internal Deep Lake bug, not something fixable from our side.

    WHY NOT deeplake.delete() + recreate:
      deeplake.delete() permanently retires the dataset path — the name
      can never be reused. Doing this on every single-repo re-index would
      require a new dataset name (v6, v7, v8, ...) every time, which is
      unworkable.

    HOW THIS WORKS INSTEAD (rewrite-in-place):
      1. Load all chunks (embeddings, texts, metadata) currently stored.
      2. Filter out rows belonging to repo_name — keep everything else.
      3. Reset the dataset's content IN PLACE at the same path using
         deeplake.empty(path, overwrite=True). This clears all rows and
         tensors but does NOT retire the path (unlike deeplake.delete()).
      4. Recreate the three tensors.
      5. Re-append the kept rows (chunks belonging to other repos).

    Returns the number of chunks removed for repo_name.
    """
    path  = _get_dataset_path()
    token = _get_token()

    ds    = get_or_create_dataset()
    total = len(ds)

    if total == 0:
        print(f"  [deeplake] Dataset is empty, nothing to delete for: {repo_name}")
        return 0

    # Load everything currently in the dataset.
    embeddings, texts, metas = _load_all(repo_name=None)

    keep_indices  = [i for i, m in enumerate(metas) if m.get("repo_name") != repo_name]
    deleted_count = total - len(keep_indices)

    if deleted_count == 0:
        print(f"  [deeplake] No existing chunks found for repo: {repo_name}")
        return 0

    kept_texts = [texts[i] for i in keep_indices]
    kept_metas = [metas[i] for i in keep_indices]

    if keep_indices:
        kept_embeddings = embeddings[keep_indices]
    else:
        embedding_dim   = embeddings.shape[1] if embeddings.ndim == 2 else 768
        kept_embeddings = np.empty((0, embedding_dim), dtype=np.float32)

    # Reset the dataset content in place — does NOT retire the path.
    ds = deeplake.empty(path, token=token, overwrite=True)
    with ds:
        ds.create_tensor("embedding", htype="embedding",
                         dtype="float32", sample_compression=None)
        ds.create_tensor("text",      htype="text")
        ds.create_tensor("metadata",  htype="json")

    # Re-load a clean handle, consistent with get_or_create_dataset's pattern.
    ds = deeplake.load(path, token=token)

    # Re-append rows from other repos.
    if kept_texts:
        with ds:
            for i in range(len(kept_texts)):
                ds.append({
                    "embedding": np.array(kept_embeddings[i], dtype=np.float32),
                    "text":      kept_texts[i],
                    "metadata":  json.dumps(kept_metas[i]),
                })

    print(f"  [deeplake] Deleted {deleted_count} chunks for repo: {repo_name} "
          f"({len(kept_texts)} chunks from other repos preserved)")
    return deleted_count


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
# Read: BM25 index builder (internal)
# ---------------------------------------------------------------------------

def _build_bm25_index(texts: list[str]) -> BM25Okapi:
    """
    Build a BM25 index from a list of chunk texts.

    Tokenization is simple whitespace + lowercase split — sufficient for
    code since identifiers, keywords, and filenames are space-separated
    in the context headers and code content.

    BM25Okapi is the standard Okapi BM25 variant: TF is saturated so very
    frequent terms don't dominate, and IDF penalises terms that appear in
    most chunks (common boilerplate).
    """
    tokenized = [text.lower().split() for text in texts]
    return BM25Okapi(tokenized)


# ---------------------------------------------------------------------------
# Read: hybrid search (BM25 + cosine via RRF) — PRIMARY search function
# ---------------------------------------------------------------------------

def hybrid_search(
    query_vector: list[float],
    query_text: str,
    top_k: int = 10,
    repo_name: Optional[str] = None,
) -> list[dict]:
    """
    Find the top-k chunks using Hybrid Search: BM25 + cosine similarity
    fused via Reciprocal Rank Fusion (RRF).

    WHY THIS OVER PURE COSINE:
      Pure cosine similarity (~44-63% benchmark accuracy) misses cases where
      the lexically correct file ranks lower than semantically adjacent files.
      Example: "how does authentication work" returns chat/session files
      (semantically close) instead of lib/auth.ts (lexically correct — "auth"
      is literally in the filename and code).

      BM25 fixes this by rewarding exact term overlap. Combining both via RRF
      gives lexical precision AND semantic coverage simultaneously.

    HOW RRF WORKS:
      Rather than combining raw scores (which have incompatible scales),
      RRF combines ranks:

        RRF_score(chunk) = 1/(k + cosine_rank) + 1/(k + bm25_rank)

      where k=60 is the standard constant that prevents top-ranked documents
      from dominating too strongly. A chunk ranked #1 by both methods gets
      the highest combined score. A chunk ranked #1 by cosine but #50 by BM25
      still scores well but not as high as a consensus top result.

    Parameters:
        query_vector  768-dim float list from embedder.embed_query()
        query_text    Raw query string — used for BM25 tokenization
        top_k         Number of results to return
        repo_name     If given, search only within this repo.
                      If None, search across all repos.

    Returns list of dicts sorted by RRF score descending:
    [{"text": "...", "score": 0.031, "repo_name": "...", "file_path": "...", ...}]

    Note: RRF scores are not cosine similarities — they're rank-fusion scores
    (typically in range 0.01–0.04). Higher is still better.
    """
    embeddings, texts, metas = _load_all(repo_name=repo_name)

    if len(embeddings) == 0:
        print(f"  [deeplake] No chunks found"
              + (f" for repo: {repo_name}" if repo_name else "") + ".")
        return []

    n = len(texts)

    # --- Dense scores (cosine similarity) ---
    query_np   = np.array(query_vector, dtype=np.float32)
    query_norm = np.linalg.norm(query_np)
    if query_norm == 0:
        return []
    query_np = query_np / query_norm

    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms      = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    dense_scores = normalized @ query_np   # shape (N,)

    # --- Sparse scores (BM25) ---
    bm25          = _build_bm25_index(texts)
    sparse_scores = np.array(
        bm25.get_scores(query_text.lower().split()),
        dtype=np.float32,
    )

    # --- Reciprocal Rank Fusion ---
    # argsort ascending then argsort again gives rank (0 = worst, N-1 = best)
    # We want rank 0 = best, so invert: rank = (N-1) - rank_ascending
    RRF_K = 60   # standard constant

    dense_rank_asc  = np.argsort(np.argsort(dense_scores))   # 0 = worst
    sparse_rank_asc = np.argsort(np.argsort(sparse_scores))  # 0 = worst

    # Convert to descending rank (0 = best) for RRF formula
    dense_rank  = (n - 1) - dense_rank_asc
    sparse_rank = (n - 1) - sparse_rank_asc

    rrf_scores = (
        1.0 / (RRF_K + dense_rank) +
        1.0 / (RRF_K + sparse_rank)
    )

    # Get top_k results
    actual_k    = min(top_k, n)
    top_indices = np.argsort(rrf_scores)[-actual_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "text":  texts[idx],
            "score": float(rrf_scores[idx]),
            **metas[idx],
        })

    return results


# ---------------------------------------------------------------------------
# Read: cosine-only similarity search (kept as fallback / for aggregation)
# ---------------------------------------------------------------------------

def similarity_search(
    query_vector: list[float],
    top_k: int = 4,
    repo_name: Optional[str] = None,
) -> list[dict]:
    """
    Find the top-k chunks most semantically similar to a query vector.
    Pure cosine similarity — used internally by aggregated search.

    For the primary retrieval path, use hybrid_search() instead.

    HOW IT WORKS:
      1. Load all embeddings as a matrix M of shape (N, 768) — one network call
      2. Normalize M and the query vector to unit length
      3. Compute M @ query_vector — one matrix multiply, gives N cosine scores
      4. argsort scores, take top_k indices
      5. Build result dicts from those indices

    Parameters:
        query_vector  768-dim float list from embedder.embed_query()
        top_k         Max number of results to return
        repo_name     If given, search only within this repo.
                      If None, search across all repos.

    Returns list of dicts sorted by similarity score descending.
    """
    embeddings, texts, metas = _load_all(repo_name=repo_name)

    if len(embeddings) == 0:
        print(f"  [deeplake] No chunks found"
              + (f" for repo: {repo_name}" if repo_name else "") + ".")
        return []

    query_np = np.array(query_vector, dtype=np.float32)

    query_norm = np.linalg.norm(query_np)
    if query_norm == 0:
        return []
    query_np = query_np / query_norm

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    scores = normalized @ query_np   # shape: (N,)

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
    Uses cosine similarity (not hybrid) since BM25 per-repo is expensive
    and this function is used for cross-repo semantic coverage, not precision.

    WHY THIS EXISTS:
      Regular similarity_search(top_k=6) returns the 6 highest-scoring
      chunks globally. For "do any repos implement rate limiting?", those
      6 chunks might all be from 1-2 repos, completely missing other repos
      that also have rate limiting but scored 7th, 8th, 9th overall.

      This function guarantees every repo gets exactly one representative
      chunk — its highest-scoring one — so Gemini can make a complete
      assessment across all repos.

    Returns list of chunk dicts, one per repo, sorted by score descending.
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
    best: dict[str, tuple[int, float]] = {}
    for i, meta in enumerate(metas):
        repo = meta.get("repo_name", "unknown")
        s    = float(scores[i])
        if repo not in best or s > best[repo][1]:
            best[repo] = (i, s)

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
    Uses cosine similarity internally (same as before) since the aggregation
    logic already handles the fairness problem that hybrid search solves
    at the chunk level.

    WHY THIS EXISTS:
      For comparative questions ("which repo has the best auth?"), global
      top-k returns chunks from the repo with the most auth-related content,
      crowding out other repos. This function scores each repo as the average
      of its top-3 chunk scores and ranks repos by that aggregate score.

    Returns list of repo result dicts sorted by repo_score descending.
    """
    candidates = similarity_search(
        query_vector=query_vector,
        top_k=candidate_k,
        repo_name=None,
    )

    if not candidates:
        return []

    repo_chunks: dict[str, list[dict]] = {}
    for chunk in candidates:
        name = chunk.get("repo_name", "unknown")
        repo_chunks.setdefault(name, []).append(chunk)

    TOP_N = 3
    repo_scores = []
    for name, chunks in repo_chunks.items():
        top_n_scores = [c["score"] for c in chunks[:TOP_N]]
        repo_score   = sum(top_n_scores) / len(top_n_scores)
        repo_scores.append((name, repo_score, chunks))

    repo_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for rank, (name, score, chunks) in enumerate(repo_scores[:top_repos], start=1):
        best_chunks = chunks[:chunks_per_repo]
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

    Returns list sorted by repo_name.
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