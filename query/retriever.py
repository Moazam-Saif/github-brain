"""
query/retriever.py
------------------
PURPOSE:
  Translates a classified question into actual search results from Deep Lake.
  Sits between the engine (which decides what to search) and the store
  (which executes the search). Owns query enrichment and deduplication.

HOW IT WORKS:
  Four retrieval strategies — one per query type:

  retrieve_cross_repo_metadata(question)
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
    filtered to the target repo. This is the primary search path and the
    one that benefits most from hybrid retrieval.

    WHY HYBRID HERE SPECIFICALLY:
      This is where the pure cosine failure mode showed up — "how does
      authentication work" returning session/chat files instead of lib/auth.ts.
      BM25 catches exact term matches (e.g. "auth" in lib/auth.ts filename and
      code), while cosine catches semantic intent. RRF fuses both rankings so
      a file that's lexically AND semantically relevant rises to the top.

    WHY ENRICHMENT:
      Follow-up questions like "how does that connect to the DB?" have no
      meaning in isolation. Adding prior exchange as context makes the
      embedding capture the right semantic direction.

    WHY DEDUPLICATION:
      Without it, the same highly-relevant chunk gets returned every turn,
      wasting context space and making answers repetitive.

TOP-K VALUES:
  cross_repo_semantic → per-repo (one chunk per repo, up to 10 repos)
  repo_specific       → 5 final chunks (fetches 20 raw via hybrid, deduplicates down to 5)
"""

import hashlib
from typing import Optional

from indexer.embedder       import embed_query
from indexer.deeplake_store import (hybrid_search,
                                    similarity_search,
                                    similarity_search_per_repo,
                                    similarity_search_aggregated,
                                    list_all_repos)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_K_REPO_SPECIFIC      = 5    # final chunks returned per turn in a deep dive
TOP_K_HYBRID_FETCH       = 20   # raw candidates fetched before deduplication
                                 # wider than before (was 10) because hybrid
                                 # scoring changes rank ordering significantly


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
    BM25 tokenization in hybrid_search. This is intentional — enrichment
    adds context words that help BM25 find relevant follow-up chunks too.
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
    """
    key = (f"{chunk.get('repo_name')}::"
           f"{chunk.get('file_path')}::"
           f"{chunk.get('chunk_index')}")
    return hashlib.md5(key.encode()).hexdigest()


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
    Uses cosine-only aggregated search (similarity_search_aggregated) since
    the aggregation logic already handles the fairness problem at the repo level.

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
    Uses cosine-only per-repo search (similarity_search_per_repo) since
    the goal here is coverage across repos, not precision within one repo.

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
    seen_chunk_ids: set,
) -> list[dict]:
    """
    Hybrid search (BM25 + cosine via RRF) within a single repo,
    with query enrichment and deduplication.

    This is the primary retrieval path and the one most improved by
    hybrid search. Pure cosine was missing lexically correct files
    (e.g. lib/auth.ts for "how does authentication work") because
    semantically adjacent files (session/chat code) ranked higher.
    BM25 catches the exact term match; RRF fuses both rankings.

    Parameters:
        question               Current user question.
        repo_name              Which repo to search within.
        conversation_history   Full session history for query enrichment.
        seen_chunk_ids         Chunks already used this session (mutated
                               in place — new IDs added by this function).

    Process:
      1. Enrich the query with recent conversation context
      2. Embed the enriched query (for cosine component)
      3. Run hybrid_search (BM25 + cosine + RRF) filtered to repo_name
      4. Filter out chunks already seen this session
      5. Return up to TOP_K_REPO_SPECIFIC fresh chunks

    Returns fresh chunk dicts sorted by RRF score descending.
    """
    enriched = _build_enriched_query(question, conversation_history)
    print(f"  [retriever] Repo-specific hybrid search in '{repo_name}'")

    query_vector = embed_query(enriched)
    if query_vector is None:
        print("  [retriever] Failed to embed query.")
        return []

    # Hybrid search: BM25 + cosine via RRF
    # Fetch TOP_K_HYBRID_FETCH candidates before deduplication to ensure
    # we have enough headroom after filtering already-seen chunks.
    raw = hybrid_search(
        query_vector=query_vector,
        query_text=enriched,
        top_k=TOP_K_HYBRID_FETCH,
        repo_name=repo_name,
    )

    fresh = []
    for chunk in raw:
        cid = _chunk_id(chunk)
        if cid not in seen_chunk_ids:
            fresh.append(chunk)
            seen_chunk_ids.add(cid)
        if len(fresh) >= TOP_K_REPO_SPECIFIC:
            break

    duped = len(raw) - len(fresh)
    print(f"  [retriever] {len(fresh)} fresh chunks "
          f"({duped} already seen this session).")
    return fresh