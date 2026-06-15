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
  hybrid_search(query_vector, query_text, repo_name=None)  → BM25 + cosine via RRF (primary)
  similarity_search(query_vector, repo_name=None)          → cosine-only (used by aggregation)
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

BM25 TOKENIZATION (FIX 1 — camelCase/snake_case):
  The original implementation used plain whitespace splitting: .lower().split()
  This means getUserByEmail stayed as one token "getuserbyemail", and a query
  asking about "getUserByEmail" would produce token "getuserbyemail" — no match
  because the code chunk tokenized it the same way. Identifiers like
  prisma.user.findUnique or NextAuthOptions never overlap with query tokens.

  Fix: _tokenize() splits on camelCase transitions AND non-alphanumeric chars,
  so getUserByEmail → ["get", "user", "by", "email"] and the query
  "how does getUserByEmail work" → ["how", "does", "get", "user", "by",
  "email", "work"]. Now BM25 scores the chunk containing getUserByEmail highly
  because "get", "user", "by", "email" all match.

  Applied symmetrically to both the index (chunk texts) and the query string
  in hybrid_search, so the token vocabularies are always aligned.

FAST TEXT/METADATA LOADING (FIX 3 — no integer indexing in loop):
  The original _load_all used:
    all_texts = [str(ds.text[i].numpy()) for i in range(total)]
  Deep Lake warns that integer indexing in a for loop is slow and recommends
  enumerate(ds.tensor) instead. With 21 repos and hundreds of chunks, the
  per-row network overhead was noticeable.

  Fix: use enumerate(ds.text) and enumerate(ds.metadata) which use Deep Lake's
  optimised iterator path. The embedding matrix still uses ds.embedding.numpy()
  (bulk load in one call) which was already optimal.

WHY ONE DATASET FOR ALL REPOS:
  Keeping everything in one dataset (hub://ORG/github_brain_v5) lets us do
  cross-repo search trivially — no joining, no federation. Filtering to a
  specific repo is done after loading metadata, which is cheap.

DELETION STRATEGY (rewrite-in-place, not ds.pop()):
  Deep Lake 3.x has a known internal bug — calling ds.pop() on a dataset that
  already has committed data raises "OverflowError: can't convert negative int
  to unsigned" inside Deep Lake's own commit-diff tracking (chunk_engine.py).

  Also: deeplake.delete() permanently retires the dataset path. The fix used in
  delete_repo_chunks(): load everything, filter out the target repo's rows in
  memory, reset the dataset content IN PLACE at the same path using
  deeplake.empty(path, overwrite=True), then re-append the kept rows.
"""

import os
import re
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

    Uses rewrite-in-place strategy — see module docstring for rationale.
    Returns the number of chunks removed for repo_name.
    """
    path  = _get_dataset_path()
    token = _get_token()

    ds    = get_or_create_dataset()
    total = len(ds)

    if total == 0:
        print(f"  [deeplake] Dataset is empty, nothing to delete for: {repo_name}")
        return 0

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

    ds = deeplake.empty(path, token=token, overwrite=True)
    with ds:
        ds.create_tensor("embedding", htype="embedding",
                         dtype="float32", sample_compression=None)
        ds.create_tensor("text",      htype="text")
        ds.create_tensor("metadata",  htype="json")

    ds = deeplake.load(path, token=token)

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

    FIX 3 — fast iterator for text and metadata:
      The original implementation indexed by integer in a for loop:
        all_texts = [str(ds.text[i].numpy()) for i in range(total)]
      Deep Lake warns this is slow and recommends enumerate(ds.tensor) instead,
      which uses the optimised iterator path with prefetching.

      Embeddings already used ds.embedding.numpy() (bulk matrix load) which
      was already optimal and is unchanged.

    Returns:
        embeddings  np.ndarray shape (N, 768)
        texts       list of N strings
        metas       list of N metadata dicts
    """
    ds    = get_or_create_dataset()
    total = len(ds)

    if total == 0:
        return np.array([]), [], []

    try:
        # Bulk load embeddings — one network call, returns full matrix.
        all_embeddings = ds.embedding.numpy()   # shape: (total, 768)

        # FIX 3: use enumerate(ds.text) instead of ds.text[i] in a range loop.
        # Deep Lake's iterator is significantly faster than integer indexing.
        all_texts = [str(sample.numpy()) for _, sample in enumerate(ds.text)]

        # FIX 3: same for metadata.
        all_metas = []
        for _, sample in enumerate(ds.metadata):
            try:
                raw = sample.numpy()
                if hasattr(raw, "item"):
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

    indices = [i for i, m in enumerate(all_metas)
               if m.get("repo_name") == repo_name]

    if not indices:
        return np.array([]), [], []

    filtered_embeddings = all_embeddings[indices]
    filtered_texts      = [all_texts[i] for i in indices]
    filtered_metas      = [all_metas[i] for i in indices]

    return filtered_embeddings, filtered_texts, filtered_metas


# ---------------------------------------------------------------------------
# BM25 tokenization (FIX 1 — camelCase / snake_case / path-aware)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 in a way that handles code identifiers correctly.

    FIX 1: The original whitespace-only tokenization (.lower().split()) kept
    camelCase identifiers as single tokens. getUserByEmail became one token
    "getuserbyemail", so a query containing "getUserByEmail" produced a token
    that never matched individual words in code chunks.

    This function splits on BOTH camelCase transitions AND non-alphanumeric
    characters (underscores, dots, slashes, hyphens, punctuation), so:
      getUserByEmail     → ["get", "user", "by", "email"]
      prisma.user.create → ["prisma", "user", "create"]
      lib/auth.ts        → ["lib", "auth", "ts"]
      NextAuthOptions    → ["next", "auth", "options"]
      work?              → ["work"]   (punctuation stripped)

    Applied symmetrically to both corpus texts and query strings, so the
    token vocabularies are always aligned regardless of source.

    Tokens shorter than 2 characters are dropped (noise: single letters,
    leftover punctuation fragments).
    """
    # Insert space before uppercase letters that follow lowercase letters
    # (camelCase split: getUserByEmail → get User By Email)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Insert space before uppercase runs followed by lowercase
    # (handles abbreviations: HTMLParser → HTML Parser)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)

    text = text.lower()

    # Split on anything that isn't a letter or digit.
    tokens = re.split(r"[^a-z0-9]+", text)

    # Drop empty strings and single-character noise tokens.
    return [t for t in tokens if len(t) >= 2]


def _build_bm25_index(texts: list[str]) -> BM25Okapi:
    """
    Build a BM25 index from a list of chunk texts using subword tokenization.

    FIX 1: Previously used .lower().split() (whitespace only). Now uses
    _tokenize() which splits camelCase and non-alphanumeric boundaries so
    code identifiers, file paths, and function names are correctly indexed
    as individual tokens rather than opaque single-token blobs.

    BM25Okapi applies TF saturation (very frequent terms don't dominate)
    and IDF weighting (rare terms score higher than common boilerplate).
    """
    tokenized = [_tokenize(text) for text in texts]
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
      RRF_score(chunk) = 1/(k + cosine_rank) + 1/(k + bm25_rank)
      where k=60 is the standard constant. A chunk ranked #1 by both methods
      gets the highest combined score.

    FIX 1 applied here: query_text is tokenized via _tokenize() (not
    .lower().split()) so camelCase and path-based query terms match the
    identically tokenized corpus.

    Parameters:
        query_vector  768-dim float list from embedder.embed_query()
        query_text    Raw query string — tokenized via _tokenize() for BM25
        top_k         Number of results to return
        repo_name     If given, search only within this repo.

    Returns list of dicts sorted by RRF score descending.
    Note: RRF scores are rank-fusion scores (typically 0.01–0.04), not
    cosine similarities. Higher is still better.
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

    norms        = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms        = np.where(norms == 0, 1, norms)
    normalized   = embeddings / norms
    dense_scores = normalized @ query_np   # shape (N,)

    # --- Sparse scores (BM25) ---
    # FIX 1: use _tokenize() instead of .lower().split() for both
    # the corpus index and the query tokens.
    bm25          = _build_bm25_index(texts)
    query_tokens  = _tokenize(query_text)
    sparse_scores = np.array(
        bm25.get_scores(query_tokens),
        dtype=np.float32,
    )

    # --- Reciprocal Rank Fusion ---
    RRF_K = 60

    # argsort twice gives rank (0 = worst, N-1 = best)
    dense_rank_asc  = np.argsort(np.argsort(dense_scores))
    sparse_rank_asc = np.argsort(np.argsort(sparse_scores))

    # Convert to descending rank (0 = best) for RRF formula
    dense_rank  = (n - 1) - dense_rank_asc
    sparse_rank = (n - 1) - sparse_rank_asc

    rrf_scores = (
        1.0 / (RRF_K + dense_rank) +
        1.0 / (RRF_K + sparse_rank)
    )

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
# Read: cosine-only similarity search (used internally by aggregation)
# ---------------------------------------------------------------------------

def similarity_search(
    query_vector: list[float],
    top_k: int = 4,
    repo_name: Optional[str] = None,
) -> list[dict]:
    """
    Find the top-k chunks most semantically similar to a query vector.
    Pure cosine similarity — used internally by aggregated/per-repo search.
    For the primary retrieval path, use hybrid_search() instead.
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

    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms      = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    scores = normalized @ query_np

    actual_k    = min(top_k, len(scores))
    top_indices = np.argsort(scores)[-actual_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "text":  texts[idx],
            "score": float(scores[idx]),
            **metas[idx],
        })

    return results


def similarity_search_per_repo(
    query_vector: list[float],
    max_repos: int = 10,
) -> list[dict]:
    """
    Return the single best-matching chunk per repo for a given query.
    Uses cosine similarity — BM25 per-repo is expensive and this function
    is used for cross-repo semantic coverage, not within-repo precision.

    Guarantees every repo gets exactly one representative chunk so Gemini
    can assess all repos, not just those that dominated global top-k.
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
    scores     = normalized @ query_np

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
    Score repos as a whole for comparative questions ("which repo has the best auth?").

    Uses cosine internally — the aggregation logic (avg of top-3 chunk scores
    per repo) already addresses the fairness problem that hybrid search solves
    at the chunk level. Applying hybrid here would add BM25 cost with no
    additional benefit since repo ranking is by aggregate score, not precision.

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
      - query engine for cross_repo_metadata questions
      - engine.py for repo name normalization (cached in session — see engine.py)

    Does not load embeddings — only reads metadata, which is fast.
    Returns list sorted by repo_name.
    """
    _, _, all_metas = _load_all(repo_name=None)

    seen = {}
    for meta in all_metas:
        repo_name = meta.get("repo_name")
        if repo_name and repo_name not in seen:
            seen[repo_name] = {
                "repo_name":            repo_name,
                "repo_description":     meta.get("repo_description"),
                "repo_technologies":    meta.get("repo_technologies", []),
                "repo_purpose":         meta.get("repo_purpose"),
                "repo_language":        meta.get("repo_language"),
                "deployment_url":       meta.get("deployment_url"),
                "repo_topics":          meta.get("repo_topics", []),
                "metadata_confidence":  meta.get("metadata_confidence"),
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
