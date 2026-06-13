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
    Uses a two-layer retrieval strategy:

    Layer 1 — Keyword-to-file mapping (up to 2 slots):
      Before running cosine similarity, the question is scanned for strong
      signal keywords (auth, database, route, etc.). If a match is found,
      chunks from files whose paths contain the mapped hint strings are
      fetched and reserved. This guarantees structurally relevant files
      (e.g. auth.ts for an auth question) are always in context even if
      their cosine score isn't the highest.

    Layer 2 — Cosine similarity (remaining slots up to TOP_K_REPO_SPECIFIC):
      The remaining slots are filled by standard cosine similarity search,
      excluding files already covered by Layer 1 to avoid duplication.

    WHY TWO LAYERS: Dense configuration code (e.g. a NextAuth config file)
    doesn't naturally score high against a natural language question even
    though it directly answers it. The keyword layer captures structural
    knowledge (file path = file purpose) that cosine similarity can't.

    Deduplication against seen_chunk_ids ensures the same chunk isn't
    repeated across turns in a session.

TOP-K VALUES:
  Total chunks per turn:     5  (2 keyword-reserved + 3 cosine)
  Raw cosine fetch:          10 (double, to have headroom after dedup)
  Keyword fetch per group:   4  (enough to cover a 2-3 chunk file fully)
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

TOP_K_REPO_SPECIFIC    = 5   # total chunks per turn in a repo deep-dive session
KEYWORD_RESERVED_SLOTS = 2   # max slots reserved for keyword-matched chunks
KEYWORD_FETCH_K        = 4   # how many chunks to fetch per keyword-matched file group


# ---------------------------------------------------------------------------
# Keyword-to-file mapping
# ---------------------------------------------------------------------------

# Maps sets of question keywords to file path hint strings.
# A chunk is "keyword-matched" if its file_path contains any of the hint strings
# for the matched keyword group (case-insensitive substring match).
#
# Structure:
#   keywords (tuple) → file_hints (list)
#
# When a question contains any word from `keywords`, chunks whose file_path
# contains any string from `file_hints` are fetched for the reserved slots.
#
# Multiple keyword groups can match a single question — reserved slots are
# filled in group match order until KEYWORD_RESERVED_SLOTS is reached.

KEYWORD_FILE_MAP = [
    (
        ("auth", "authentication", "login", "logout", "oauth", "jwt",
         "session", "credential", "signup", "register", "password", "token"),
        ["auth", "login", "credential", "oauth", "session"],
    ),
    (
        ("database", "db", "schema", "model", "prisma", "query",
         "migration", "orm", "sql", "mongo", "postgres", "firestore",
         "supabase", "firebase", "mongoose"),
        ["schema", "prisma", "db", "model", "migration", "database"],
    ),
    (
        ("route", "endpoint", "api", "request", "response",
         "handler", "middleware", "controller"),
        ["route", "api", "middleware", "handler", "controller", "endpoint"],
    ),
    (
        ("state", "store", "redux", "slice", "context",
         "dispatch", "action", "reducer"),
        ["store", "slice", "context", "reducer", "state"],
    ),
    (
        ("component", "render", "ui", "page", "view",
         "layout", "style", "css", "tailwind"),
        ["component", "page", "layout", "view", "style"],
    ),
    (
        ("config", "setup", "environment", "env",
         "initialize", "bootstrap", "init"),
        ["config", "setup", "env", "init", "bootstrap"],
    ),
    (
        ("test", "spec", "unit", "integration", "mock", "fixture"),
        ["test", "spec", "mock", "fixture", "__test__"],
    ),
    (
        ("stream", "socket", "websocket", "realtime", "event",
         "queue", "worker", "job", "cron"),
        ["stream", "socket", "queue", "worker", "job", "event"],
    ),
    (
        ("upload", "file", "storage", "image", "media", "cloud"),
        ["upload", "storage", "media", "cloud", "file"],
    ),
    (
        ("embed", "vector", "embedding", "similarity", "search", "retrieval", "index"),
        ["embed", "vector", "retriev", "search", "index"],
    ),
]


def _get_keyword_file_hints(question: str) -> list[list[str]]:
    """
    Scan the question for keyword matches and return the matched file hint groups.

    Returns a list of hint lists — one per matched keyword group — in match order.
    Each hint list contains file path substrings to search for.

    Example:
      question = "how does authentication work?"
      → matches group ("auth", "authentication", ...)
      → returns [["auth", "login", "credential", "oauth", "session"]]

    Multiple groups can match:
      question = "how does auth connect to the database?"
      → returns [["auth", ...], ["schema", "prisma", "db", ...]]
    """
    question_lower = question.lower()
    question_words = set(question_lower.replace("?", "").replace(",", "").split())

    matched_hint_groups = []
    for keywords, file_hints in KEYWORD_FILE_MAP:
        if any(kw in question_words for kw in keywords):
            matched_hint_groups.append(file_hints)

    return matched_hint_groups


def _file_matches_hints(file_path: str, hints: list[str]) -> bool:
    """
    Return True if file_path contains any of the hint strings (case-insensitive).

    Example:
      file_path = "lib/auth.ts", hints = ["auth", "login"]  → True
      file_path = "lib/prisma.ts", hints = ["auth", "login"] → False
    """
    path_lower = file_path.lower()
    return any(hint in path_lower for hint in hints)


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
    Two-layer semantic search within a single repo, with enrichment and deduplication.
    Used for repo_specific questions (deep dives).

    Parameters:
        question               Current user question.
        repo_name              Which repo to search within.
        conversation_history   Full session history for query enrichment.
        seen_chunk_ids         Chunks already used this session (mutated in place —
                               new chunk IDs are added to this set by this function).

    Process:
      Layer 1 — Keyword-to-file mapping (up to KEYWORD_RESERVED_SLOTS = 2):
        1. Scan question for keyword matches
        2. For each matched keyword group, fetch KEYWORD_FETCH_K chunks from
           files whose path contains the hint strings
        3. Add fresh (not yet seen) keyword-matched chunks to results first
        4. Stop once KEYWORD_RESERVED_SLOTS fresh keyword chunks are collected

      Layer 2 — Cosine similarity (remaining slots up to TOP_K_REPO_SPECIFIC = 5):
        5. Enrich query with recent conversation context
        6. Embed enriched query
        7. Run cosine similarity search (fetch TOP_K_REPO_SPECIFIC * 2 = 10 raw)
        8. Skip chunks already seen OR already covered by keyword layer
        9. Fill remaining slots with fresh cosine chunks

    Returns up to TOP_K_REPO_SPECIFIC fresh chunk dicts sorted:
      keyword-matched chunks first, then cosine chunks.
    """
    print(f"  [retriever] Repo-specific search in '{repo_name}'")

    # -----------------------------------------------------------------------
    # Layer 1: Keyword-to-file mapping
    # -----------------------------------------------------------------------
    keyword_chunks = []
    keyword_covered_file_paths = set()  # file paths already covered by keyword layer

    hint_groups = _get_keyword_file_hints(question)

    if hint_groups:
        print(f"  [retriever] Keyword hint groups matched: {len(hint_groups)}")

        # Embed the raw question (no enrichment) for keyword-layer search.
        # We use the raw question here because enrichment adds prior context
        # which could dilute the keyword signal for file matching.
        kw_query_vector = embed_query(question)

        if kw_query_vector is not None:
            # Fetch a broad pool of chunks from this repo to filter by file path.
            all_repo_chunks = similarity_search(
                query_vector=kw_query_vector,
                top_k=len(hint_groups) * KEYWORD_FETCH_K * 2,
                repo_name=repo_name,
            )

            for hints in hint_groups:
                if len(keyword_chunks) >= KEYWORD_RESERVED_SLOTS:
                    break

                # Find chunks from files matching this hint group.
                matched = [
                    c for c in all_repo_chunks
                    if _file_matches_hints(c.get("file_path", ""), hints)
                ]

                for chunk in matched:
                    if len(keyword_chunks) >= KEYWORD_RESERVED_SLOTS:
                        break
                    cid = _chunk_id(chunk)
                    if cid not in seen_chunk_ids:
                        keyword_chunks.append(chunk)
                        seen_chunk_ids.add(cid)
                        keyword_covered_file_paths.add(chunk.get("file_path", ""))

        matched_files = list(keyword_covered_file_paths)
        print(f"  [retriever] Keyword layer: {len(keyword_chunks)} chunks "
              f"from {matched_files}")

    # -----------------------------------------------------------------------
    # Layer 2: Cosine similarity for remaining slots
    # -----------------------------------------------------------------------
    remaining_slots = TOP_K_REPO_SPECIFIC - len(keyword_chunks)
    cosine_chunks   = []

    if remaining_slots > 0:
        enriched     = _build_enriched_query(question, conversation_history)
        query_vector = embed_query(enriched)

        if query_vector is not None:
            raw = similarity_search(
                query_vector=query_vector,
                top_k=TOP_K_REPO_SPECIFIC * 2,
                repo_name=repo_name,
            )

            for chunk in raw:
                if len(cosine_chunks) >= remaining_slots:
                    break
                cid = _chunk_id(chunk)
                if cid not in seen_chunk_ids:
                    cosine_chunks.append(chunk)
                    seen_chunk_ids.add(cid)

        print(f"  [retriever] Cosine layer: {len(cosine_chunks)} chunks")

    # Keyword chunks first (structurally guaranteed), then cosine chunks.
    results = keyword_chunks + cosine_chunks

    total_raw  = (len(hint_groups) * KEYWORD_FETCH_K * 2) + (TOP_K_REPO_SPECIFIC * 2)
    total_seen = total_raw - len(results)
    print(f"  [retriever] {len(results)} total chunks "
          f"({len(keyword_chunks)} keyword + {len(cosine_chunks)} cosine, "
          f"{total_seen} skipped/seen).")

    return results
