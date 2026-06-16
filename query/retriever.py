"""
query/retriever.py
------------------
PURPOSE:
  Translates a classified question into actual search results from Deep Lake.
  Sits between the engine (which decides what to search) and the store
  (which executes the search). Owns query enrichment, re-ranking,
  and deduplication.

HOW IT WORKS:
  Four retrieval strategies — one per query type:

  retrieve_cross_repo_metadata()
    Does NOT do vector search. Calls list_all_repos() to get every repo's
    metadata summary. The engine passes it to Gemini which filters and
    answers in natural language. Fast — no embeddings involved.

  retrieve_cross_repo_semantic(question)
    Embeds the question and runs similarity_search_per_repo() across ALL
    chunks with no repo_name filter. Returns one best chunk per repo so
    Gemini can assess every repo, not just those that dominated top-k.

  retrieve_cross_repo_comparative(question)
    Embeds the question and runs similarity_search_aggregated() which scores
    each repo as the average of its top-3 chunk scores — fair ranking
    regardless of repo size or chunk count.

  retrieve_repo_specific(question, repo_name, conversation_history, seen_chunk_ids)
    Uses HYBRID SEARCH (BM25 + cosine via RRF) on the enriched query,
    filtered to the target repo, followed by CROSS-ENCODER RE-RANKING.
    This is the primary search path and the one that benefits most from
    both hybrid retrieval and re-ranking.

FIX 4 — CROSS-ENCODER RE-RANKER:
  After hybrid_search returns 20 candidates, a cross-encoder model
  (cross-encoder/ms-marco-MiniLM-L-6-v2) re-scores each (query, chunk)
  pair jointly. Unlike bi-encoders which embed query and chunk separately,
  a cross-encoder sees both together and captures their precise interaction.

  This directly addresses the chunk boundary problem (weakness 4): even if
  a chunk's cosine/BM25 scores were mediocre because the function signature
  was in a neighbouring chunk, the cross-encoder can recognise that the
  chunk's CONTENT is highly relevant to the question and boost its rank.

  Pipeline for retrieve_repo_specific:
    hybrid_search (top 20) → cross-encoder re-rank → deduplication → top 5

  The cross-encoder model is loaded lazily on first use (not at import time)
  so startup is not penalised for query types that don't use it.
  Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~22MB, CPU-only, ~50ms/pair)

FIX 7 — SEEN_CHUNK_IDS STARVATION:
  The original implementation used a plain set() for seen_chunk_ids that
  grew indefinitely across turns. In a long session (10+ turns) over a small
  repo (~40-50 chunks), 5 chunks × 10 turns = 50 seen IDs can exhaust the
  entire repo's chunk pool. When hybrid_search returns 20 candidates and all
  20 are in seen_chunk_ids, fresh returns empty and the user gets
  "I couldn't find relevant code" even for valid questions.

  Fix: seen_chunk_ids is now a collections.deque with maxlen=40. When the
  deque is full, the oldest entry is automatically evicted to make room for
  the newest. This means chunks from 8+ turns ago become eligible for
  retrieval again, while chunks from the last 8 turns are still suppressed
  to avoid immediate repetition.

  The deque is stored in the session dict in engine.py. Membership testing
  (cid not in seen_chunk_ids) works identically for deque as for set.
  Adding new IDs uses seen_chunk_ids.append(cid) instead of .add(cid).

  MAXLEN=40 rationale: 5 chunks/turn × 8 turns = 40 — suppresses the most
  recent 8 turns of chunks while keeping the deque small. For a repo with
  40 total chunks this means at worst 1-2 turns of "already seen" overlap
  before chunks cycle back in, which is acceptable.

TOP-K VALUES:
  cross_repo_semantic  → per-repo (one chunk per repo, up to 10 repos)
  repo_specific        → 5 final chunks after re-rank + dedup
                         (fetches 20 raw via hybrid, re-ranks all 20,
                         deduplicates down to 5)
"""

import hashlib
import os
import requests
from collections import deque
from typing import Optional

from indexer.embedder       import embed_query
from indexer.deeplake_store import (hybrid_search,
                                    similarity_search_per_repo,
                                    similarity_search_aggregated,
                                    list_all_repos)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_K_REPO_SPECIFIC  = 5    # final chunks returned per turn in a deep dive
TOP_K_HYBRID_FETCH   = 20   # raw candidates fetched before re-rank + dedup
                             # wider than pure cosine (was 10) because hybrid
                             # scoring changes rank ordering significantly,
                             # and re-ranker needs headroom to promote chunks

SEEN_CHUNK_MAXLEN    = 40   # FIX 7: cap on seen_chunk_ids deque
                             # 5 chunks/turn × 8 turns = 40
                             # oldest entries evicted automatically when full


# ---------------------------------------------------------------------------
# Re-ranker (FIX 4)
# ---------------------------------------------------------------------------



def _rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Re-rank chunks using the hosted Jina API while preserving existing logic."""
    if not chunks:
        return chunks
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        print("  [retriever] JINA_API_KEY not set; skipping rerank.")
        return chunks
    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["text"] for c in chunks],
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("results", []):
            chunks[item["index"]]["rerank_score"] = float(item["relevance_score"])
        ranked = sorted(chunks, key=lambda c: c.get("rerank_score", float("-inf")), reverse=True)
        print("  [retriever] Re-ranked.")
        return ranked
    except Exception as exc:
        print(f"  [retriever] Jina rerank failed ({exc}); using hybrid ranking.")
        return chunks

# ---------------------------------------------------------------------------
# Query enrichment (repo-specific only)
# ---------------------------------------------------------------------------

def _build_enriched_query(question: str, conversation_history: list[dict]) -> str:
    """
    Combine the current question with the last 2 conversation turns.

    WHY: Follow-up questions like "how does that feed into verification?"
    have no semantic meaning on their own. The embedding model would search
    for chunks about "feed into verification" rather than the actual topic.
    By prepending the prior exchange, the embedding captures the right meaning.

    We use at most 2 turns (1 user + 1 assistant) to avoid dragging in noise
    from earlier parts of the conversation that may be on a different topic.

    Assistant answers are truncated at 300 chars to avoid bloating the
    query with verbose explanations.

    NOTE: The enriched query is used for BOTH the cosine embedding AND the
    BM25 tokenization in hybrid_search, AND as the query for cross-encoder
    re-ranking. This is intentional — enrichment adds context words that
    help all three components find relevant follow-up chunks.
    """
    if not conversation_history:
        return question

    recent = conversation_history[-2:]
    parts  = []
    for turn in recent:
        role    = turn.get("role", "").capitalize()
        content = turn.get("content", "")
        if turn.get("role") == "assistant" and len(content) > 300:
            content = content[:300] + "..."
        parts.append(f"{role}: {content}")

    return f"Previous context:\n{chr(10).join(parts)}\n\nCurrent question: {question}"


# ---------------------------------------------------------------------------
# Chunk deduplication
# ---------------------------------------------------------------------------

def _chunk_id(chunk: dict) -> str:
    """
    Stable ID for a chunk: hash of repo_name + file_path + chunk_index.

    Used to track which chunks have already been sent to Gemini this session
    so we don't repeat the same code on every follow-up question.

    The hash is deterministic and collision-resistant for our use case
    (MD5 is fine for deduplication — not used for security).
    """
    key = (f"{chunk.get('repo_name')}::"
           f"{chunk.get('file_path')}::"
           f"{chunk.get('chunk_index')}")
    return hashlib.md5(key.encode()).hexdigest()


def make_seen_chunk_ids() -> deque:
    """
    Create a new seen_chunk_ids deque for a fresh session.

    FIX 7: Returns a deque with maxlen=SEEN_CHUNK_MAXLEN instead of a plain
    set. The deque automatically evicts the oldest entry when full, so the
    pool of suppressed chunks is always bounded. This prevents the starvation
    issue where a long session exhausts all chunks in a small repo.

    Called by engine.py when starting a new session (in _new_session).
    """
    return deque(maxlen=SEEN_CHUNK_MAXLEN)


# ---------------------------------------------------------------------------
# Public retrieval functions
# ---------------------------------------------------------------------------

def retrieve_cross_repo_metadata() -> list[dict]:
    """
    Return metadata summaries for all indexed repos.
    Used for cross_repo_metadata questions.

    No embedding, no vector search. Just reads the metadata that was
    generated at index time and stored on every chunk.

    Returns list of repo summary dicts (see list_all_repos() in deeplake_store).
    """
    print("  [retriever] Cross-repo metadata lookup (no vector search)")
    repos = list_all_repos()
    print(f"  [retriever] Found {len(repos)} repos in metadata.")
    return repos


def retrieve_cross_repo_comparative(question: str) -> list[dict]:
    """
    Rank repos fairly for comparative questions like "which has the best auth?".

    Uses cosine-only aggregated search (similarity_search_aggregated) — the
    aggregation logic (avg of top-3 chunk scores per repo) already handles the
    fairness problem at the repo level. Hybrid search is applied at the chunk
    level in retrieve_repo_specific; for cross-repo ranking, cosine aggregation
    is the right tool.

    Returns list of repo result dicts from similarity_search_aggregated().
    """
    q = question[:80] + "..." if len(question) > 80 else question
    print(f"  [retriever] Cross-repo comparative search: '{q}'")

    query_vector = embed_query(question)
    if query_vector is None:
        print("  [retriever] Failed to embed query.")
        return []

    results = similarity_search_aggregated(
        query_vector=query_vector,
        top_repos=3,
        chunks_per_repo=3,
        candidate_k=50,
    )

    print(f"  [retriever] Ranked {len(results)} repos for comparison.")
    return results


def retrieve_cross_repo_semantic(question: str) -> list[dict]:
    """
    Find the best matching chunk per repo for a semantic question.

    Uses cosine-only per-repo search (similarity_search_per_repo). The goal
    here is COVERAGE across all repos — ensuring every repo gets a
    representative chunk so Gemini can determine which repos are relevant.
    Hybrid search is used for within-repo precision (retrieve_repo_specific);
    cross-repo coverage is better served by pure cosine which ranks globally.

    Returns list of chunk dicts, one per repo, sorted by score descending.
    """
    q = question[:80] + "..." if len(question) > 80 else question
    print(f"  [retriever] Cross-repo semantic search (per-repo): '{q}'")

    query_vector = embed_query(question)
    if query_vector is None:
        print("  [retriever] Failed to embed query.")
        return []

    results = similarity_search_per_repo(
        query_vector=query_vector,
        max_repos=10,
    )

    print(f"  [retriever] Got best chunk from {len(results)} repos.")
    return results


def retrieve_repo_specific(
    question: str,
    repo_name: str,
    conversation_history: list[dict],
    seen_chunk_ids: deque,
) -> list[dict]:
    """
    Hybrid search (BM25 + cosine via RRF) + cross-encoder re-ranking within
    a single repo, with query enrichment and deduplication.

    This is the primary retrieval path. The full pipeline:
      1. Enrich the query with recent conversation context
      2. Embed the enriched query (for cosine component of hybrid search)
      3. Run hybrid_search: BM25 + cosine + RRF, filtered to repo_name
         → returns TOP_K_HYBRID_FETCH (20) candidates
      4. Re-rank all 20 candidates with the cross-encoder model
         → reorders by precise (query, chunk) relevance score
      5. Deduplicate against seen_chunk_ids (capped deque — FIX 7)
         → removes chunks already seen this session
      6. Return up to TOP_K_REPO_SPECIFIC (5) fresh chunks

    WHY HYBRID + RE-RANK (not just one or the other):
      Hybrid search (step 3) gives strong recall — the right chunks are very
      likely to be in the top 20 because both BM25 (lexical) and cosine
      (semantic) are working together. Re-ranking (step 4) gives strong
      precision — from those 20, the cross-encoder picks the 5 that best
      answer this specific question. The two are complementary.

    Parameters:
        question               Current user question.
        repo_name              Which repo to search within.
        conversation_history   Full session history for query enrichment.
        seen_chunk_ids         Capped deque of chunk IDs already seen this
                               session (FIX 7). Mutated in place — new IDs
                               are appended by this function.

    Returns fresh chunk dicts sorted by re-rank score descending.
    """
    enriched = _build_enriched_query(question, conversation_history)
    print(f"  [retriever] Repo-specific hybrid search in '{repo_name}'")

    query_vector = embed_query(enriched)
    if query_vector is None:
        print("  [retriever] Failed to embed query.")
        return []

    # Step 3: Hybrid search — fetch TOP_K_HYBRID_FETCH candidates.
    raw = hybrid_search(
        query_vector=query_vector,
        query_text=enriched,
        top_k=TOP_K_HYBRID_FETCH,
        repo_name=repo_name,
    )

    if not raw:
        return []

    # Step 4: Re-rank with cross-encoder.
    # The enriched query is used (not just the bare question) so the
    # re-ranker has the same conversational context as the embedding.
    reranked = _rerank(enriched, raw)

    # Step 5: Deduplicate against seen_chunk_ids.
    # FIX 7: seen_chunk_ids is a deque with maxlen=SEEN_CHUNK_MAXLEN.
    # Membership testing works identically to a set. Appending evicts
    # the oldest entry automatically when the deque is full.
    fresh = []
    for chunk in reranked:
        cid = _chunk_id(chunk)
        if cid not in seen_chunk_ids:
            fresh.append(chunk)
            seen_chunk_ids.append(cid)   # .append() not .add() — it's a deque
        if len(fresh) >= TOP_K_REPO_SPECIFIC:
            break

    duped = len(reranked) - len(fresh)
    print(f"  [retriever] {len(fresh)} fresh chunks "
          f"({duped} already seen this session, "
          f"{len(seen_chunk_ids)}/{SEEN_CHUNK_MAXLEN} IDs tracked).")
    return fresh
