"""
query/engine.py
---------------
PURPOSE:
  The main orchestrator for answering user questions. This is the only
  module that cli.py calls. It wires together the router, retriever,
  session management, and Gemini answer generation into one clean flow.

HOW IT WORKS (one turn):
  1. Router classifies the question into one of five types
  2. The appropriate retriever fetches context (metadata or code chunks)
  3. A prompt is built from that context + session history
  4. Gemini generates the answer
  5. The answer is returned to cli.py alongside the updated session

FIVE QUESTION TYPES:
  list_repos           → metadata from list_all_repos() → Gemini answers
  cross_repo_metadata  → metadata from list_all_repos() → Gemini filters + answers
  cross_repo_semantic  → code chunks from all repos → Gemini answers
  cross_repo_comparative → ranked repo chunks → Gemini compares
  repo_specific        → code chunks from one repo + session history → Gemini answers

FIX 2 — CACHED list_all_repos():
  The original code called list_all_repos() inside the repo_specific branch
  on EVERY turn for repo name normalization. list_all_repos() calls _load_all()
  which pulls the entire dataset from Deep Lake. Combined with the _load_all()
  call inside hybrid_search(), that was TWO full dataset loads per turn.

  Fix: list_all_repos() is called ONCE at the start of query() and its result
  is stored as session["all_repos_cache"]. On subsequent turns in the same
  session, the cache is reused. The cache is invalidated (refreshed) when a
  session resets (new repo or cross-repo question), which is the only time
  the repo list could meaningfully change mid-conversation.

  For cross-repo questions (metadata, semantic, comparative) the cache is
  populated fresh at the start of that turn and not stored — those paths
  already called list_all_repos() exactly once and are unchanged.

FIX 6 — CASING GUARD WITH EXPLICIT LOG:
  The repo name normalization (case-insensitive match against indexed repos)
  was already in the right position before the session continuation check.
  However, when list_all_repos() returned empty (e.g. Deep Lake load failure),
  matched_repo fell through to None, and the session was created with a
  barebones {repo_name: repo_name} metadata dict. Every subsequent turn then
  showed "stack: unknown, deployed: not deployed" in the prompt, silently
  producing degraded answers.

  Fix: explicit guard — if list_all_repos() returns empty or the repo name
  doesn't match any indexed repo, log a clear warning and return an
  informative message to the user rather than creating a broken session.

FIX 7 — SESSION USES DEQUE FOR seen_chunk_ids:
  seen_chunk_ids is now a collections.deque(maxlen=40) instead of a plain set.
  Created via retriever.make_seen_chunk_ids() so the maxlen constant is owned
  by retriever.py where the deduplication logic lives.

  The deque is stored in session["seen_chunk_ids"] and passed directly to
  retrieve_repo_specific(), which uses .append() instead of .add() and
  membership-tests with `in` (works identically for deque as for set).

SESSION MANAGEMENT (repo_specific only):
  A session is a dict that persists in memory across turns in cli.py.
  It holds: active_repo, conversation_history, seen_chunk_ids (deque),
  repo_metadata, summarized_history, and all_repos_cache.

  Sessions start when a repo_specific question is first asked.
  Sessions continue as long as questions stay within the same repo.
  Sessions reset when the user asks a cross-repo or list question,
  or explicitly types "reset" in the CLI.
"""

import time
from typing import Optional
from gemini_client import get_client, GEMINI_MODEL

from query.router    import classify_question
from query.retriever import (retrieve_cross_repo_metadata,
                             retrieve_cross_repo_semantic,
                             retrieve_cross_repo_comparative,
                             retrieve_repo_specific,
                             make_seen_chunk_ids)
from indexer.deeplake_store import list_all_repos


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

MAX_VERBATIM_TURNS = 4


def _new_session(repo_name: str, repo_metadata: dict, all_repos_cache: list) -> dict:
    """
    Create a fresh session for a repo deep dive.

    FIX 7: seen_chunk_ids is now a deque(maxlen=40) via make_seen_chunk_ids().
    FIX 2: all_repos_cache stores the result of list_all_repos() so it isn't
           re-fetched on every subsequent turn in this session.
    """
    return {
        "active_repo":          repo_name,
        "conversation_history": [],
        "seen_chunk_ids":       make_seen_chunk_ids(),   # FIX 7: deque, not set
        "repo_metadata":        repo_metadata,
        "summarized_history":   None,
        "all_repos_cache":      all_repos_cache,         # FIX 2: cached repo list
    }


def _new_comparison_session(
    comparison_repos: list[str],
    all_repos_cache: list,
) -> dict:
    """
    Create a session for a cross_repo_comparative exchange.

    Structurally distinct from a repo_specific session — "active_repo" stays
    None, so repo_specific code (which checks session["active_repo"]) never
    mistakes this for a repo deep-dive. "comparison_repos" records which
    repos were just compared, in case a follow-up needs that context.

    NOTE: query() reads session["comparison_repos"] and passes it to
    classify_question() as active_comparison, so a follow-up like "what
    about the UI?" is correctly routed back to these SAME repos. However,
    comparison_history is reset fresh on every comparative turn right now
    (see the caller) — prior Q&A within the same comparison is NOT currently
    carried into _build_comparative_prompt(), so Gemini won't see what was
    already said about e.g. database design when answering a UI follow-up.
    That's a separate, not-yet-made change if continuity of content (not
    just correct routing) is needed.
    """
    return {
        "active_repo":         None,              # never set — distinguishes from repo_specific
        "comparison_repos":    comparison_repos,   # repos just compared
        "comparison_history":  [],                 # filled in by the caller after this returns
        "all_repos_cache":     all_repos_cache,     # FIX 2 pattern: cached repo list, reused
    }


def _normalize_repo_names(
    names: list[str],
    all_repos: list[dict],
) -> list[str]:
    """
    Normalize a list of repo names (from the router) against the exact
    casing of indexed repos.

    THE BUG THIS FIXES: the router correctly extracts repo names from the
    question (e.g. "compare CorpLaw-AI and Claim-Verification-Automation on
    database design" → repos: ["CorpLaw-AI", "Claim-Verification-Automation"]),
    but if this list is never passed down to retrieve_cross_repo_comparative(),
    the retriever has no way to know specific repos were named and falls back
    to similarity_search_aggregated() — a GLOBAL ranking across all indexed
    repos. That's how an unrelated repo (e.g. github-brain) can outscore one
    of the actually-named repos and take a slot in the comparison.

    This function ensures the names extracted by the router match the EXACT
    casing stored in the index (the router may return "Corplaw-AI" while the
    index has "CorpLaw-AI"), so the named-repo filter in the retriever works
    correctly instead of silently matching nothing.

    Names that don't match any indexed repo are logged and dropped rather
    than silently passed through (which would cause hybrid_search to filter
    a nonexistent repo_name and return zero chunks for that slot).

    Parameters:
        names      List of repo name strings from route["repos"].
        all_repos  Full list of indexed repo dicts from list_all_repos().

    Returns list of normalized repo name strings (may be shorter than input
    if some names didn't match anything indexed).
    """
    normalized = []
    for name in names:
        match = next(
            (r["repo_name"] for r in all_repos
             if r["repo_name"].lower() == name.lower()),
            None,
        )
        if match:
            if match != name:
                print(f"[engine] Normalized named repo: '{name}' → '{match}'")
            normalized.append(match)
        else:
            print(f"[engine] WARNING: named repo '{name}' not found in index. "
                  f"Available: {[r['repo_name'] for r in all_repos]}")
    return normalized


def _manage_context_window(session: dict, client) -> dict:
    """
    Enforce the sliding window on conversation history.

    When history exceeds MAX_VERBATIM_TURNS × 2 entries (each turn is
    1 user + 1 assistant = 2 entries), summarize the oldest entries into
    one paragraph and replace them with that summary.

    This keeps prompts from growing indefinitely while preserving the gist
    of earlier exchanges. Summarization failure is non-fatal — a placeholder
    is used so the session can continue.
    """
    history = session["conversation_history"]
    if len(history) <= MAX_VERBATIM_TURNS * 2:
        return session

    split        = len(history) - (MAX_VERBATIM_TURNS * 2)
    old_turns    = history[:split]
    recent_turns = history[split:]

    old_text = "\n".join(
        f"{t['role'].capitalize()}: {t['content']}" for t in old_turns
    )
    summary_prompt = (
        f"Summarize this conversation segment in 2-3 sentences, "
        f"keeping key technical facts:\n\n{old_text}"
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=summary_prompt
        )
        summary = response.text.strip()
        time.sleep(4)
    except Exception as e:
        summary = (f"[Earlier conversation about "
                   f"{session['active_repo']} codebase]")
        print(f"  [engine] Context summarization failed: {e}")

    session["summarized_history"]   = summary
    session["conversation_history"] = recent_turns
    return session


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_metadata_prompt(question: str, repos: list[dict]) -> str:
    """
    Prompt for both list_repos and cross_repo_metadata questions.

    Passes every repo's full metadata to Gemini including the richer
    structural fields so it can answer questions like "which repos have auth?"
    without touching code chunks.
    """
    lines = []
    for r in repos:
        parts = [f"• {r['repo_name']}"]
        if r.get("repo_description"):
            parts.append(r["repo_description"])
        if r.get("deployment_url"):
            parts.append(f"deployed: {r['deployment_url']}")
        if r.get("repo_technologies"):
            parts.append(f"stack: {', '.join(r['repo_technologies'][:6])}")
        if r.get("repo_language"):
            parts.append(f"language: {r['repo_language']}")
        if r.get("repo_purpose"):
            parts.append(f"type: {r['repo_purpose']}")
        if r.get("architecture_pattern"):
            parts.append(f"architecture: {r['architecture_pattern']}")
        if r.get("database_type"):
            parts.append(f"database: {r['database_type']}")
        if r.get("api_style"):
            parts.append(f"api: {r['api_style']}")
        if r.get("frontend_framework"):
            parts.append(f"ui: {r['frontend_framework']}")
        if r.get("external_services"):
            parts.append(f"services: {', '.join(r['external_services'][:4])}")
        if r.get("key_features"):
            parts.append(f"features: {', '.join(r['key_features'][:4])}")

        flags = []
        if r.get("has_authentication"): flags.append("auth")
        if r.get("has_database"):       flags.append("database")
        if r.get("has_api"):            flags.append("api")
        if r.get("has_frontend"):       flags.append("frontend")
        if r.get("has_tests"):          flags.append("tests")
        if flags:
            parts.append(f"has: {', '.join(flags)}")

        lines.append(" | ".join(parts))

    repos_text = "\n".join(lines) if lines else "No repos indexed yet."

    return f"""You are GitHub Brain, an assistant that helps a developer understand their GitHub repositories.

Answer the following question using ONLY the repository metadata provided below.
Be specific — mention repo names and relevant details.
If a deployment URL is present and relevant, include it in your answer.

Question: {question}

Repository metadata:
{repos_text}"""


def _build_semantic_prompt(question: str, chunks: list[dict]) -> str:
    """Prompt for cross_repo_semantic questions."""
    chunks_text = "\n\n---\n\n".join(
        f"Repo: {c.get('repo_name')} | File: {c.get('file_path')}\n"
        f"Similarity score: {c.get('score', 0):.2f}\n\n{c.get('text', '')}"
        for c in chunks
    )

    return f"""You are GitHub Brain, an assistant that helps a developer understand their GitHub repositories.

Answer the following question using ONLY the code context provided below.
Reference specific repo names, file paths, and code where relevant.
If the answer cannot be determined from the provided context, say so honestly.

Question: {question}

Retrieved code context:
{chunks_text}"""


def _build_comparative_prompt(question: str, ranked_repos: list[dict]) -> str:
    """Prompt for cross_repo_comparative questions."""
    sections = []
    for repo in ranked_repos:
        chunks_text = "\n\n".join(
            f"  [{c.get('file_path')} chunk {c.get('chunk_index')}]"
            f" (score {c.get('score', 0):.2f})\n  {c.get('text', '')}"
            for c in repo.get("chunks", [])
        )
        techs    = ", ".join(repo.get("repo_technologies", [])) or "unknown"
        deployed = repo.get("deployment_url") or "not deployed"

        section = (
            f"=== Repo #{repo['repo_rank']}: {repo['repo_name']} "
            f"(relevance score: {repo['repo_score']:.3f}) ===\n"
            f"Description: {repo.get('repo_description') or 'none'}\n"
            f"Stack: {techs} | Deployed: {deployed}\n\n"
            f"Most relevant code:\n{chunks_text}"
        )
        sections.append(section)

    repos_text = "\n\n".join(sections) if sections else "No repos found."

    return f"""You are GitHub Brain, an assistant that helps a developer understand their GitHub repositories.

The developer is asking a comparative question about their repos.
Below are the most relevant repos ranked by how closely their code matches the topic,
along with their most relevant code chunks.

Compare these repos honestly based on the code shown. Reference specific file paths,
function names, and implementation details. If one repo clearly does something better,
say so and explain why. If they're comparable, say that too.

Question: {question}

Ranked repositories (by topic relevance):
{repos_text}"""


def _build_repo_specific_prompt(
    question: str,
    chunks: list[dict],
    session: dict,
) -> str:
    """
    Prompt for repo_specific deep-dive questions.

    Includes three layers of context:
      1. Repo metadata summary — reminds Gemini what this repo is
      2. Conversation history — prior turns so follow-ups make sense
      3. Retrieved chunks — the actual code relevant to this question
    """
    meta     = session["repo_metadata"]
    techs    = ", ".join(meta.get("repo_technologies", [])) or "unknown"
    deployed = meta.get("deployment_url") or "not deployed"

    repo_summary = (
        f"Repo: {meta.get('repo_name')} | "
        f"Language: {meta.get('repo_language')} | "
        f"Type: {meta.get('repo_purpose')} | "
        f"Stack: {techs} | Deployed: {deployed}\n"
        f"Description: {meta.get('repo_description') or 'none'}"
    )

    history_parts = []
    if session.get("summarized_history"):
        history_parts.append(
            f"[Summary of earlier conversation]\n{session['summarized_history']}"
        )
    for turn in session["conversation_history"]:
        history_parts.append(f"{turn['role'].capitalize()}: {turn['content']}")
    history_str = "\n\n".join(history_parts) if history_parts else "None"

    # Show re-rank score if available, otherwise fall back to cosine score.
    chunks_text = "\n\n---\n\n".join(
        f"File: {c.get('file_path')} (chunk {c.get('chunk_index')})\n"
        f"Relevance: {c.get('rerank_score', c.get('score', 0)):.2f}\n\n"
        f"{c.get('text', '')}"
        for c in chunks
    )

    return f"""You are GitHub Brain, an assistant helping a developer deeply understand one of their repositories.

Repository:
{repo_summary}

Conversation so far:
{history_str}

Relevant code for this question:
{chunks_text}

Current question: {question}

Answer thoroughly. Reference specific file paths and function/class names.
If the answer spans multiple files, explain how they connect.
If something is unclear from the retrieved code, say so — don't guess."""


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def _generate_answer(prompt: str, client, max_retries: int = 3) -> str:
    """
    Call Gemini to generate a natural language answer from a built prompt.

    Retries on rate limit (429) errors with exponential backoff.
    The sleep(4) after each successful call keeps us under 15 req/min
    on the free tier (router + answer = 2 calls per turn → 8s minimum per turn).
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
            time.sleep(4)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"  [engine] Rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [engine] Generation error (attempt {attempt + 1}): {e}")

    return "I encountered an error generating a response. Please try again."


# ---------------------------------------------------------------------------
# Public: main query function
# ---------------------------------------------------------------------------

def query(
    question: str,
    session: Optional[dict] = None,
) -> tuple[str, Optional[dict]]:
    """
    Process a user question end-to-end and return an answer.

    This is the only function cli.py needs to call.

    Parameters:
        question  The user's natural language question.
        session   Current session dict or None.

    Returns:
        (answer_str, updated_session)

    FIX 2 — list_all_repos() caching:
      list_all_repos() is called once per query() invocation and its result
      is used for:
        (a) repo name normalization in the repo_specific branch
        (b) building the metadata prompt for list_repos / cross_repo_metadata
      For repo_specific turns after the first, the result is stored in
      session["all_repos_cache"] and reused — no second Deep Lake load.

    FIX 6 — casing guard:
      If list_all_repos() returns empty or the repo name can't be matched,
      we log a warning and return an informative error rather than silently
      creating a broken session with empty metadata.
    """
    client      = get_client()
    active_repo = session["active_repo"] if session else None

    # Pull comparison_repos from the session (set by _new_comparison_session
    # on the previous turn, if the last question was cross_repo_comparative).
    # Passed to the router as active_comparison so a follow-up like "what
    # about the UI?" is classified as a continuation of the SAME comparison
    # instead of a fresh global search.
    active_comparison = session.get("comparison_repos") if session else None

    print(f"\n[engine] Question: '{question}'")
    route      = classify_question(
        question,
        active_repo=active_repo,
        active_comparison=active_comparison,
    )
    query_type = route["type"]
    print(f"[engine] Routed as: {route}")

    # -----------------------------------------------------------------------
    # list_repos
    # -----------------------------------------------------------------------
    if query_type == "list_repos":
        repos  = list_all_repos()
        prompt = _build_metadata_prompt(question, repos)
        answer = _generate_answer(prompt, client)
        return answer, None

    # -----------------------------------------------------------------------
    # cross_repo_metadata
    # -----------------------------------------------------------------------
    if query_type == "cross_repo_metadata":
        repos = retrieve_cross_repo_metadata()
        if not repos:
            return (
                "No repos are indexed yet. "
                "Run `python cli.py index --mode full` first.",
                None,
            )
        prompt = _build_metadata_prompt(question, repos)
        answer = _generate_answer(prompt, client)
        return answer, None

    # -----------------------------------------------------------------------
    # cross_repo_semantic
    # -----------------------------------------------------------------------
    if query_type == "cross_repo_semantic":
        chunks = retrieve_cross_repo_semantic(question)
        if not chunks:
            return (
                "I couldn't find relevant code for that question across your repos. "
                "Make sure your repos are indexed with "
                "`python cli.py index --mode full`.",
                None,
            )
        prompt = _build_semantic_prompt(question, chunks)
        answer = _generate_answer(prompt, client)
        return answer, None

    # -----------------------------------------------------------------------
    # cross_repo_comparative
    # -----------------------------------------------------------------------
    if query_type == "cross_repo_comparative":
        # THE FIX: the router already extracts named repos into route["repos"]
        # (visible in the logs: {'type': 'cross_repo_comparative', 'repos':
        # ['CorpLaw-AI', 'Claim-Verification-Automation']}) but this branch was
        # discarding that field entirely and calling retrieve_cross_repo_comparative
        # with ONLY the question — which forces the retriever into its global
        # ranking fallback (no named_repos = no filter = searches ALL 21 repos,
        # letting unrelated repos like github-brain outscore the ones actually
        # asked about).
        #
        # Fix: extract route.get("repos"), normalize casing against the index,
        # and pass the result as named_repos= so retrieve_cross_repo_comparative
        # takes its MODE 2 path — fetching and ranking ONLY the named repos.
        raw_named   = route.get("repos")
        named_repos = None

        if raw_named:
            all_repos_for_cmp = list_all_repos()
            if all_repos_for_cmp:
                named_repos = _normalize_repo_names(raw_named, all_repos_for_cmp)
                if not named_repos:
                    print("[engine] All named repos failed normalization. "
                          "Falling back to global ranking.")
            else:
                print("[engine] WARNING: list_all_repos() returned empty during "
                      "comparative. Falling back to global ranking.")

        ranked_repos = retrieve_cross_repo_comparative(
            question,
            named_repos=named_repos,   # None → global ranking, list → ONLY these repos
        )
        if not ranked_repos:
            return (
                "I couldn't find enough relevant code across your repos to "
                "make a comparison. Make sure your repos are indexed with "
                "`python cli.py index --mode full`.",
                None,
            )
        prompt = _build_comparative_prompt(question, ranked_repos)
        answer = _generate_answer(prompt, client)

        # Build a comparison session so the repos just compared are recorded.
        # Uses named_repos if the router named specific repos; otherwise falls
        # back to whichever repos the global ranking actually returned, so the
        # session always reflects the repos this answer was actually about.
        comparison_repos_for_session = (
            named_repos if named_repos
            else [r["repo_name"] for r in ranked_repos]
        )
        # all_repos_for_cmp was already fetched above when raw_named was set;
        # if it wasn't (global comparison, no names given), fetch it now so
        # the session's all_repos_cache is populated either way.
        cache_for_session = (
            all_repos_for_cmp if raw_named and all_repos_for_cmp
            else list_all_repos()
        )

        new_session = _new_comparison_session(
            comparison_repos_for_session,
            cache_for_session,
        )
        new_session["comparison_history"] = [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
        return answer, new_session

    # -----------------------------------------------------------------------
    # repo_specific — deep dive with session management
    # -----------------------------------------------------------------------
    repo_name = route.get("repo", "")

    # FIX 2: Use cached repo list from session if available, otherwise fetch once.
    # This avoids a second _load_all() call on top of the one inside hybrid_search.
    if session and session.get("all_repos_cache"):
        all_repos = session["all_repos_cache"]
        print(f"[engine] Using cached repo list ({len(all_repos)} repos)")
    else:
        all_repos = list_all_repos()
        print(f"[engine] Fetched repo list ({len(all_repos)} repos)")

    # FIX 6: Guard against empty repo list — log clearly and return informative error.
    if not all_repos:
        return (
            "I couldn't load the indexed repo list from Deep Lake. "
            "Check your ACTIVELOOP_TOKEN and ACTIVELOOP_ORG, then try again.",
            session,
        )

    # FIX 6: Case-insensitive normalization — BEFORE the session continuation check.
    # If the router returns "corplaw-ai" but the dataset has "CorpLaw-AI",
    # normalize to the exact indexed casing NOW so session["active_repo"] comparisons
    # work correctly. Log a warning if normalization changed the name.
    matched_repo = next(
        (r for r in all_repos if r["repo_name"].lower() == repo_name.lower()),
        None,
    )

    if matched_repo is None:
        # FIX 6: Explicit warning — don't silently create a broken session.
        print(f"[engine] WARNING: repo '{repo_name}' from router not found in "
              f"indexed repos. Available: {[r['repo_name'] for r in all_repos]}")
        return (
            f"I couldn't find a repo named '{repo_name}' in your indexed repos. "
            f"Try asking about one of your indexed repos by its exact name, or "
            f"run `python cli.py index --mode full` if you haven't indexed yet.",
            None,
        )

    if matched_repo["repo_name"] != repo_name:
        print(f"[engine] Normalized repo name: '{repo_name}' → '{matched_repo['repo_name']}'")
    repo_name = matched_repo["repo_name"]

    # Start new session or continue existing one.
    # Normalization above guarantees repo_name casing matches session["active_repo"].
    if session is None or session.get("active_repo") != repo_name:
        session = _new_session(repo_name, matched_repo, all_repos)
        print(f"[engine] New session: {repo_name}")
    else:
        print(f"[engine] Continuing session: {repo_name}")

    # Compress old history if needed.
    session = _manage_context_window(session, client)

    # Retrieve relevant chunks via hybrid search + re-rank.
    # seen_chunk_ids is the capped deque from the session (FIX 7).
    chunks = retrieve_repo_specific(
        question=question,
        repo_name=repo_name,
        conversation_history=session["conversation_history"],
        seen_chunk_ids=session["seen_chunk_ids"],
    )

    if not chunks:
        answer = (
            f"I couldn't find relevant code for that in '{repo_name}'. "
            f"The repo may not be indexed, or try rephrasing the question."
        )
    else:
        prompt = _build_repo_specific_prompt(question, chunks, session)
        answer = _generate_answer(prompt, client)

    # Append this turn to session history.
    session["conversation_history"].append(
        {"role": "user",      "content": question}
    )
    session["conversation_history"].append(
        {"role": "assistant", "content": answer}
    )

    return answer, session
