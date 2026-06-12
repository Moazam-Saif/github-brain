"""
query/retriever.py
------------------
PURPOSE:
  Translates a classified question into actual search results from Deep Lake.
  Sits between the engine (which decides what to search) and the store
  (which executes the search). Owns query enrichment and deduplication.

HOW IT WORKS:
  Three search strategies, one per cross-repo type plus repo-specific:

  retrieve_cross_repo_metadata(question)
    Does NOT do vector search. Instead it calls list_all_repos() to get
    every repo's metadata summary, then returns that list to the engine.
    The engine passes it to Gemini which filters and answers in natural language.
    Fast — no embeddings involved.

    WHY: Questions like "do I have anything using Redis?" are answered
    deterministically by checking if "redis" appears in repo_technologies.
    No need to search code chunks for this.

  retrieve_cross_repo_semantic(question)
    Embeds the question and runs similarity_search across ALL chunks with
    no repo_name filter. Returns top-k chunks from whatever repos scored
    highest. The engine feeds these to Gemini.

    WHY: Questions like "which repo implements rate limiting?" can't be
    answered from metadata — the answer is in the code itself.

  retrieve_repo_specific(question, repo_name, conversation_history, seen_chunk_ids)
    Embeds an ENRICHED query (question + last 2 conversation turns) and
    runs similarity_search filtered to that repo only. Deduplicates against
    chunks already seen in this session so the same code isn't repeated.

    WHY ENRICHMENT: Follow-up questions like "how does that connect to the DB?"
    have no meaning in isolation. Adding the prior exchange as context makes
    the embedding capture the right semantic direction.

    WHY DEDUPLICATION: Without it, the same highly-relevant chunk gets returned
    every turn, wasting context space and making answers repetitive.

TOP-K VALUES:
  cross_repo_semantic → k = 6  (wider net across all repos)
  repo_specific       → k = 4  (focused, one repo)
  The repo_specific search fetches k*2 then deduplicates down to k.
"""

import hashlib
from typing import Optional

from indexer.embedder       import embed_query
from indexer.deeplake_store import (similarity_search,
                                    similarity_search_per_repo,
                                    similarity_search_aggregated,
                                    list_all_repos)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_K_REPO_SPECIFIC = 4   # chunks per turn in a repo deep-dive session


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

    The engine receives this list and passes it to Gemini along with the
    question. Gemini does the filtering (e.g. "which ones use Redis?")
    in its response rather than us doing it programmatically — this is
    more flexible because Gemini can handle fuzzy matches, synonyms, etc.

    Returns list of repo summary dicts (see list_all_repos() in deeplake_store).
    """
    print("  [retriever] Cross-repo metadata lookup (no vector search)")
    repos = list_all_repos()
    print(f"  [retriever] Found {len(repos)} repos in metadata.")
    return repos


def retrieve_cross_repo_comparative(question: str) -> list[dict]:
    """
    Rank repos fairly for comparative questions like "which has the best auth?".
    Used for cross_repo_comparative questions.

    WHY NOT regular similarity_search:
      Global top-k returns the highest-scoring chunks regardless of repo.
      A repo with many auth-related chunks floods the top-k and crowds out
      other repos even if those repos have equally good auth implementations.
      The answer ends up biased towards whichever repo has the most chunks
      on the topic, not whichever actually has the best implementation.

    HOW AGGREGATED SEARCH FIXES THIS:
      1. Fetch top-50 chunks globally (candidate pool)
      2. Group chunks by repo_name
      3. For each repo: repo_score = average of its top-3 chunk scores
         (top-3 average is fair across repo sizes — not skewed by repo having
         many chunks or only one fluke high-scoring chunk)
      4. Rank repos by repo_score
      5. Return the top-3 repos, each with their best 3 chunks

      Now Gemini receives a fair representation of each repo's relevant code
      and can make a genuine comparison.

    Returns list of repo result dicts from similarity_search_aggregated():
    [
      {
        "repo_name":   "claimsense",
        "repo_score":  0.89,
        "repo_rank":   1,
        "chunks":      [{"text": "...", "score": 0.91, ...}, ...],
        "repo_description": "...",
        ...
      },
      ...
    ]
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
    Used for cross_repo_semantic questions.

    WHY PER-REPO INSTEAD OF GLOBAL TOP-K:
      The old approach: similarity_search(top_k=6) returns the 6 highest-
      scoring chunks globally. For "do any repos implement rate limiting?",
      those 6 might all be from 1-2 repos with many relevant chunks,
      completely missing other repos that also have rate limiting but
      scored 7th or 8th overall.

      The new approach: similarity_search_per_repo() returns the single
      best chunk from EVERY repo. Gemini now sees a representative sample
      from all repos and can answer "skillswap, claimsense, and car-rental
      all implement it — corplaw does not" instead of missing repos entirely.

    HOW IT WORKS:
      1. Score all chunks against the query (same batch matrix multiply)
      2. For each repo, find its single highest-scoring chunk
      3. Return all repos sorted by their best score descending
      4. Gemini reads one chunk per repo and determines which ones are relevant

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
    Semantic search within a single repo, with enrichment and deduplication.
    Used for repo_specific questions (deep dives).

    Parameters:
        question               Current user question.
        repo_name              Which repo to search within.
        conversation_history   Full session history for query enrichment.
        seen_chunk_ids         Chunks already used this session (mutated in place —
                               new chunk IDs are added to this set by this function).

    Process:
      1. Enrich the query with recent conversation context
      2. Embed the enriched query
      3. Search Deep Lake filtered to repo_name
      4. Filter out chunks already seen this session
      5. Return up to TOP_K_REPO_SPECIFIC fresh chunks

    Returns fresh chunk dicts sorted by similarity score descending.
    """
    enriched     = _build_enriched_query(question, conversation_history)
    print(f"  [retriever] Repo-specific search in '{repo_name}'")

    query_vector = embed_query(enriched)
    if query_vector is None:
        print("  [retriever] Failed to embed query.")
        return []

    # Fetch double so we have headroom after deduplication.
    raw = similarity_search(
        query_vector=query_vector,
        top_k=TOP_K_REPO_SPECIFIC * 2,
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
