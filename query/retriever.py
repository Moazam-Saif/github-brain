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

  retrieve_cross_repo_comparative(question, named_repos=None)
    When named_repos is None (general comparison — "which repo has the
    best auth?"): runs similarity_search_aggregated() which scores each
    repo as the average of its top-3 chunk scores — fair global ranking.

    When named_repos is provided (explicit comparison — "compare CorpLaw-AI
    and skillswap on auth"): fetches chunks directly from each named repo
    using hybrid_search(), bypassing the global ranking entirely. This
    guarantees the repos the user asked about are always represented,
    regardless of how they score globally.

  retrieve_repo_specific(question, repo_name, conversation_history, seen_chunk_ids)
    Uses HYBRID SEARCH (BM25 + cosine via RRF) on the enriched+HyDE query,
    filtered to the target repo, followed by CROSS-ENCODER RE-RANKING.
    This is the primary search path and the one that benefits most from
    both hybrid retrieval and re-ranking.

QUERY REWRITING (_rewrite_for_retrieval):
  Used by cross_repo_semantic and cross_repo_comparative before embedding.
  Strips conversational framing ("do any of my repos", "which project",
  "can you tell me") and returns only the core technical concept.
  Closes the gap between how users phrase questions and how code is indexed.
  Falls back to the original question if the rewrite is longer (expansion
  detected) or if Gemini fails.

HyDE — HYPOTHETICAL DOCUMENT EMBEDDINGS (_generate_hyde_query):
  Used by retrieve_repo_specific before embedding.
  Generates a short, plausible code snippet that WOULD answer the question,
  then embeds that snippet instead of the natural language question.
  Bridges the semantic gap between question-space and code-space — the
  hypothetical snippet's vector aligns far better with indexed code chunk
  vectors than a natural language question would.
  Falls back to the enriched query if generation fails or produces prose.
  The enriched query (not HyDE) is still used for Jina reranking, since
  cross-encoders work better with natural language intent.

FIX 4 — CROSS-ENCODER RE-RANKER:
  After hybrid_search returns 20 candidates, a cross-encoder model
  re-scores each (query, chunk) pair jointly via the Jina rerank API.
  Unlike bi-encoders which embed query and chunk separately, a cross-encoder
  sees both together and captures their precise interaction.

  Pipeline for retrieve_repo_specific:
    hybrid_search (top 20) → Jina rerank → deduplication → top 5

  If JINA_API_KEY is not set, re-ranking is skipped and hybrid ranking
  is used as-is.

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
  recent 8 turns of chunks while keeping the deque small.

NAMED REPO FILTERING (cross_repo_comparative):
  When the user explicitly names repos in a comparative question
  (e.g. "compare CorpLaw-AI and Claim-Verification on database design"),
  the router extracts those names into route["repos"]. The engine passes
  them here as named_repos.

  Instead of running similarity_search_aggregated() (global ranking that
  can displace named repos with higher-scoring irrelevant repos), we run
  hybrid_search() filtered to each named repo individually, then build
  the ranked_repos structure that the engine's _build_comparative_prompt()
  expects. This guarantees the repos the user asked about always appear
  in the comparison, with full chunk context.

TOP-K VALUES:
  cross_repo_semantic  → per-repo (one chunk per repo, up to 10 repos)
  cross_repo_comparative (named) → 3 chunks per named repo via hybrid_search
  cross_repo_comparative (global) → top_repos=3, chunks_per_repo=3 via aggregation
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
from gemini_client          import get_client, GEMINI_MODEL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_K_REPO_SPECIFIC        = 5    # final chunks returned per turn in a deep dive
TOP_K_HYBRID_FETCH         = 20   # raw candidates fetched before re-rank + dedup
TOP_K_COMPARATIVE_PER_REPO = 3    # chunks fetched per named repo in comparative

# Candidate pool sizing for global comparative search.
# With a fixed candidate_k=50 across 20 repos, the pool averages 2.5 chunks
# per repo before grouping — not enough for a fair top-3 average per repo.
# Fix: scale candidate_k with the number of indexed repos so every repo gets
# a statistically fair number of shots at the pool.
# CANDIDATE_K_PER_REPO × repo_count = pool size. 5 per repo means each repo
# has ~5 chunks competing before grouping, which is enough margin for the
# top-3 average to be meaningful. Minimum of 50 preserves the old behavior
# when there are only a handful of repos.
CANDIDATE_K_PER_REPO      = 5    # pool slots per repo for global comparative
CANDIDATE_K_MIN           = 50   # floor — never fetch fewer than this

SEEN_CHUNK_MAXLEN    = 40   # FIX 7: cap on seen_chunk_ids deque
                             # 5 chunks/turn × 8 turns = 40
                             # oldest entries evicted automatically when full


# ---------------------------------------------------------------------------
# Re-ranker (FIX 4 — Jina API)
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
# Query rewriting (cross-repo paths)
# ---------------------------------------------------------------------------

def _rewrite_for_retrieval(question: str) -> str:
    """
    Strip conversational framing from a question to produce a dense,
    retrieval-optimized query for embedding.

    WHY: Embedding models compute cosine similarity between the query vector
    and chunk vectors. Code chunks don't contain question framing — they
    contain function names, variable names, comments, and logic. Phrases like
    "do any of my repos", "can you tell me", "which project" add noise to the
    query vector that pulls it away from the actual concept being searched.

    WHAT THIS DOES:
      Remove question structure, filler words, and ownership framing.
      Keep only the core technical concept(s).
      Do NOT add words that aren't implied by the question.

    Examples:
      "do any of my repos implement rate limiting?"
        → "rate limiting"
      "which repo has the best authentication flow?"
        → "authentication flow"
      "which of my projects uses Redis for caching?"
        → "Redis caching"
      "do I have anything that handles file uploads?"
        → "file upload handling"
      "which project has the most thorough error handling?"
        → "error handling"

    FALLBACK: if Gemini fails or returns something longer than the original
    question (a sign it expanded rather than stripped), the original question
    is used as-is. Bad rewrite is worse than no rewrite.

    Used by: retrieve_cross_repo_semantic, retrieve_cross_repo_comparative
    NOT used by: retrieve_repo_specific (which uses HyDE instead)
    """
    try:
        client = get_client()
        prompt = (
            "Extract the core technical concept from this question as a short, "
            "dense search phrase. Remove all question framing, conversational "
            "words, and ownership references ('my repos', 'my projects', 'do I have', "
            "'which project', 'can you', 'tell me'). Return ONLY the concept — "
            "no explanation, no punctuation, no added words.\n\n"
            f"Question: {question}\n"
            "Core concept:"
        )
        response = get_client().models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        rewritten = response.text.strip().strip('"').strip("'")
        # Sanity check: rewrite must be shorter than the original.
        # If Gemini expanded instead of stripped, discard it.
        if rewritten and len(rewritten) < len(question):
            print(f"  [retriever] Rewritten query: '{question[:60]}' → '{rewritten}'")
            return rewritten
        else:
            print(f"  [retriever] Rewrite discarded (not shorter). Using original.")
            return question
    except Exception as e:
        print(f"  [retriever] Query rewrite failed ({e}). Using original.")
        return question


# ---------------------------------------------------------------------------
# HyDE — Hypothetical Document Embedding (repo-specific path)
# ---------------------------------------------------------------------------

def _generate_hyde_query(question: str, repo_name: str, enriched: str) -> str:
    """
    Generate a hypothetical code snippet that would answer the question,
    then use that as the embedding query instead of the natural language question.

    WHY (HyDE — Hypothetical Document Embeddings):
      There is a fundamental semantic gap between a natural language question
      and a code chunk. "How does this repo handle JWT authentication?" lives
      in question-space; the actual indexed chunk lives in code-space:
      function signatures, variable names, decorators, inline comments.
      Embedding a question produces a vector that may not align well with
      the vectors of the actual code chunks that answer it.

      HyDE closes this gap: generate a plausible code snippet that WOULD
      answer the question, then embed that. The resulting vector lives in
      code-space and aligns far better with the indexed chunk vectors.
      The hypothetical code doesn't need to be correct — it just needs to
      use the right vocabulary (function names, patterns, keywords) so the
      embedding lands near real chunks.

    WHAT THIS GENERATES:
      A short, plausible code snippet (5-15 lines) using realistic variable
      and function names, in the likely language of the repo, that represents
      what the implementation of the answer would look like.

    RELATIONSHIP TO _build_enriched_query:
      _build_enriched_query adds conversation history to resolve pronoun
      references ("how does THAT work?" → prepend prior turns so the
      embedding knows what "that" refers to). HyDE replaces the question
      with a hypothetical code answer. These compose: the enriched query
      (with history context) is passed into this function as `enriched`,
      so HyDE generates a hypothetical snippet informed by the full
      conversational context, not just the raw question.

    FALLBACK: if generation fails or produces something that looks like
    prose rather than code (no common code tokens detected), fall back
    to the enriched query. A bad HyDE snippet would steer retrieval worse
    than the enriched natural language query.

    Parameters:
        question   Raw current question (for the prompt instruction).
        repo_name  Repo being searched (included in the prompt so Gemini
                   can infer the likely language/framework).
        enriched   Output of _build_enriched_query — history-enriched query.
                   Used as context and as the fallback.

    Returns the hypothetical code string, or `enriched` on failure.
    """
    CODE_TOKENS = {"def ", "function ", "class ", "return ", "const ", "=>",
                   "import ", "from ", "async ", "await ", "if ", "for "}

    try:
        prompt = (
            f"You are helping retrieve code from a repository called '{repo_name}'.\n"
            f"Write a short, plausible code snippet (5-15 lines) that would implement "
            f"the answer to this question. Use realistic function/variable names and "
            f"patterns. The code doesn't need to be runnable — it just needs to use "
            f"the right vocabulary so it can be matched against real code chunks.\n"
            f"Return ONLY the code — no explanation, no markdown, no backticks.\n\n"
            f"Context:\n{enriched}\n\n"
            f"Question: {question}\n"
            f"Hypothetical code snippet:"
        )
        response = get_client().models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        hyde = response.text.strip().strip("`")

        # Sanity check: must look like code, not a prose answer.
        if hyde and any(token in hyde for token in CODE_TOKENS):
            print(f"  [retriever] HyDE snippet generated ({len(hyde)} chars).")
            return hyde
        else:
            print(f"  [retriever] HyDE output looks like prose — falling back to enriched query.")
            return enriched
    except Exception as e:
        print(f"  [retriever] HyDE generation failed ({e}). Using enriched query.")
        return enriched




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


def retrieve_cross_repo_comparative(
    question: str,
    named_repos: Optional[list[str]] = None,
) -> list[dict]:
    """
    Retrieve chunks for comparative questions.

    Two modes depending on whether specific repos were named:

    MODE 1 — named_repos is None (general comparison):
      Runs similarity_search_aggregated() — global ranking by top-3 average
      chunk score per repo. Fair across all indexed repos. The top 3 repos
      by score are returned with their best 3 chunks each.

    MODE 2 — named_repos provided (explicit comparison):
      Bypasses global ranking entirely. For each named repo, runs
      hybrid_search() filtered to that repo and takes the top
      TOP_K_COMPARATIVE_PER_REPO (3) chunks. Builds the same ranked_repos
      structure that _build_comparative_prompt() in engine.py expects, with
      repo_rank assigned in the order the user named them (not by score).

      WHY THIS IS BETTER FOR NAMED REPOS:
        Global ranking scores ALL repos against the question. A repo that
        happens to have many semantically adjacent chunks (but isn't one the
        user named) can score higher and displace a named repo from the top 3.
        Example: "compare CorpLaw-AI and Claim-Verification on database design"
        → github-brain scored 0.546 and took the #2 slot, pushing CorpLaw-AI
        (0.513) to #3 with only 3 weak chunks from lib/prompts.ts.
        With named_repos mode, CorpLaw-AI is fetched directly and gets its
        best 3 database-relevant chunks via hybrid_search — the schema file
        scores high on BM25 for "database design".

      CASING: named_repos comes from the router which preserves the user's
      casing. The engine normalizes these against indexed repo names before
      passing them here, so exact casing is guaranteed.

    Parameters:
        question     The user's natural language question.
        named_repos  Optional list of repo names explicitly mentioned in the
                     question. Provided by engine.py after case normalization.

    Returns list of repo result dicts matching the structure expected by
    _build_comparative_prompt() in engine.py:
    [
      {
        "repo_name":         "CorpLaw-AI",
        "repo_score":        0.031,       ← RRF score (named mode) or cosine avg (global)
        "repo_rank":         1,
        "chunks":            [...],
        "repo_description":  "...",
        "repo_technologies": [...],
        "repo_purpose":      "...",
        "repo_language":     "...",
        "deployment_url":    "...",
      },
      ...
    ]
    """
    q = question[:80] + "..." if len(question) > 80 else question

    # -------------------------------------------------------------------
    # MODE 2: Named repos — fetch directly, bypass global ranking
    # -------------------------------------------------------------------
    if named_repos:
        print(f"  [retriever] Cross-repo comparative (named): {named_repos}")

        rewritten    = _rewrite_for_retrieval(question)
        query_vector = embed_query(rewritten)
        if query_vector is None:
            print("  [retriever] Failed to embed query.")
            return []

        results = []
        for rank, repo_name in enumerate(named_repos, start=1):
            chunks = hybrid_search(
                query_vector=query_vector,
                query_text=rewritten,
                top_k=TOP_K_COMPARATIVE_PER_REPO,
                repo_name=repo_name,
            )

            if not chunks:
                print(f"  [retriever] No chunks found for named repo: {repo_name}")
                # Still include the repo in results so Gemini knows it was requested
                # but had no retrievable chunks — better than silently omitting it.
                results.append({
                    "repo_name":         repo_name,
                    "repo_score":        0.0,
                    "repo_rank":         rank,
                    "chunks":            [],
                    "repo_description":  None,
                    "repo_technologies": [],
                    "repo_purpose":      None,
                    "repo_language":     None,
                    "deployment_url":    None,
                })
                continue

            # Pull repo-level metadata from the first chunk.
            first = chunks[0]
            results.append({
                "repo_name":         repo_name,
                "repo_score":        round(float(chunks[0].get("score", 0)), 4),
                "repo_rank":         rank,
                "chunks":            chunks,
                "repo_description":  first.get("repo_description"),
                "repo_technologies": first.get("repo_technologies", []),
                "repo_purpose":      first.get("repo_purpose"),
                "repo_language":     first.get("repo_language"),
                "deployment_url":    first.get("deployment_url"),
            })

        print(f"  [retriever] Named comparative: fetched chunks for "
              f"{sum(1 for r in results if r['chunks'])} / {len(named_repos)} repos.")
        return results

    # -------------------------------------------------------------------
    # MODE 1: General comparison — global aggregated ranking
    # -------------------------------------------------------------------
    print(f"  [retriever] Cross-repo comparative (global): '{q}'")

    rewritten    = _rewrite_for_retrieval(question)
    repo_count   = len(list_all_repos())
    candidate_k  = max(CANDIDATE_K_MIN, repo_count * CANDIDATE_K_PER_REPO)
    print(f"  [retriever] {repo_count} repos indexed → candidate_k={candidate_k}")

    query_vector = embed_query(rewritten)
    if query_vector is None:
        print("  [retriever] Failed to embed query.")
        return []

    results = similarity_search_aggregated(
        query_vector=query_vector,
        top_repos=3,
        chunks_per_repo=3,
        candidate_k=candidate_k,
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

    rewritten    = _rewrite_for_retrieval(question)
    query_vector = embed_query(rewritten)
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
    a single repo, with query enrichment, HyDE, and deduplication.

    This is the primary retrieval path. The full pipeline:
      1. Enrich the query with recent conversation context (_build_enriched_query)
      2. Generate a hypothetical code snippet via HyDE (_generate_hyde_query)
         — closes the semantic gap between natural language questions and code
      3. Embed the HyDE snippet (for cosine component of hybrid search)
      4. Run hybrid_search: BM25 + cosine + RRF, filtered to repo_name
         → returns TOP_K_HYBRID_FETCH (20) candidates
      5. Re-rank all 20 candidates with the Jina rerank API using the
         ENRICHED query (not HyDE) — cross-encoders work better with
         natural language intent than with hypothetical code snippets
      6. Deduplicate against seen_chunk_ids (capped deque — FIX 7)
      7. Return up to TOP_K_REPO_SPECIFIC (5) fresh chunks

    WHY HyDE FOR EMBEDDING BUT ENRICHED FOR RERANKING:
      HyDE bridges the question→code semantic gap at embedding time — a
      hypothetical snippet lives in the same vector space as indexed code.
      But the Jina reranker is a cross-encoder: it reads both query and
      chunk text together and scores their interaction. Natural language
      intent is a clearer signal for that comparison than a hypothetical
      snippet that may use different naming than the actual chunks.

    Parameters:
        question               Current user question.
        repo_name              Which repo to search within.
        conversation_history   Full session history for query enrichment.
        seen_chunk_ids         Capped deque of chunk IDs already seen this
                               session (FIX 7). Mutated in place — new IDs
                               are appended by this function.

    Returns fresh chunk dicts sorted by re-rank score descending.
    """
    enriched     = _build_enriched_query(question, conversation_history)
    hyde_query   = _generate_hyde_query(question, repo_name, enriched)
    print(f"  [retriever] Repo-specific hybrid search in '{repo_name}'")

    query_vector = embed_query(hyde_query)
    if query_vector is None:
        print("  [retriever] Failed to embed query.")
        return []

    # Step 3: Hybrid search — fetch TOP_K_HYBRID_FETCH candidates.
    raw = hybrid_search(
        query_vector=query_vector,
        query_text=hyde_query,
        top_k=TOP_K_HYBRID_FETCH,
        repo_name=repo_name,
    )

    if not raw:
        return []

    # Step 4: Re-rank with Jina API.
    # Use the enriched natural language query for re-ranking, not the HyDE
    # snippet. The reranker is a cross-encoder that compares query and chunk
    # text directly — natural language is a better signal for that comparison
    # than a hypothetical code snippet which may use different naming than
    # the actual chunks. HyDE's value is in the embedding space; once we
    # have the candidates, the reranker works better with the original intent.
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
