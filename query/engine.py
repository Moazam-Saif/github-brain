"""
query/engine.py
---------------
PURPOSE:
  The main orchestrator for answering user questions. This is the only
  module that cli.py calls. It wires together the router, retriever,
  session management, and Gemini answer generation into one clean flow.

HOW IT WORKS (one turn):
  1. Router classifies the question into one of four types
  2. The appropriate retriever fetches context (metadata or code chunks)
  3. A prompt is built from that context + session history
  4. Gemini generates the answer
  5. The answer is returned to cli.py alongside the updated session

FOUR QUESTION TYPES:
  list_repos           → metadata from list_all_repos() → Gemini answers
  cross_repo_metadata  → metadata from list_all_repos() → Gemini filters + answers
  cross_repo_semantic  → code chunks from all repos → Gemini answers
  repo_specific        → code chunks from one repo + session history → Gemini answers

SESSION MANAGEMENT (repo_specific only):
  A session is a dict that persists in memory across turns in cli.py.
  It holds: active_repo, conversation_history, seen_chunk_ids, repo_metadata,
  and a summarized_history for older turns.

  Sessions start when a repo_specific question is first asked.
  Sessions continue as long as questions stay within the same repo.
  Sessions reset when the user asks a cross-repo or list question,
  or explicitly types "reset" in the CLI.

  Why: Without session state, every follow-up question is answered in
  isolation. With it, "how does that connect to the DB?" knows what
  "that" refers to because the prior exchange is included in context.

CONTEXT WINDOW MANAGEMENT:
  conversation_history grows with every turn. After MAX_VERBATIM_TURNS (4),
  older turns are summarized into a paragraph by Gemini rather than kept
  verbatim. This prevents the prompt from growing unboundedly while
  preserving conversational continuity.

RATE LIMITING:
  Each query turn uses 2 Gemini calls: router + answer generation.
  With sleep(4) after each generate_content call, we stay under
  15 requests/minute on the free tier.
"""

import json
import time
from typing import Optional
from gemini_client import get_client, GEMINI_MODEL

from query.router    import classify_question
from query.retriever import (retrieve_cross_repo_metadata,
                             retrieve_cross_repo_semantic,
                             retrieve_cross_repo_comparative,
                             retrieve_repo_specific)
from indexer.deeplake_store import list_all_repos


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

MAX_VERBATIM_TURNS = 4


def _new_session(repo_name: str, repo_metadata: dict) -> dict:
    """Create a fresh session for a repo deep dive."""
    return {
        "active_repo":          repo_name,
        "conversation_history": [],
        "seen_chunk_ids":       set(),
        "repo_metadata":        repo_metadata,
        "summarized_history":   None,
    }


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

    split         = len(history) - (MAX_VERBATIM_TURNS * 2)
    old_turns     = history[:split]
    recent_turns  = history[split:]

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
# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_metadata_prompt(question: str, repos: list[dict]) -> str:
    """
    Prompt for both list_repos and cross_repo_metadata questions.

    Passes every repo's full metadata to Gemini including the new richer
    fields (has_authentication, has_database, key_features, etc.) so it
    can answer questions like "which repos have auth?" without touching chunks.
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

        # Boolean flags as a compact summary.
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
    """
    Prompt for cross_repo_semantic questions.

    Gives Gemini retrieved code chunks from across repos. Each chunk
    is labelled with its repo name and file path so Gemini can reference
    them precisely in the answer.
    """
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
    """
    Prompt for cross_repo_comparative questions.

    Unlike the semantic prompt which gives Gemini a flat list of chunks,
    this prompt is structured around repos. Each repo gets a section with:
      - Its rank and aggregate score (so Gemini knows the ordering)
      - Its metadata (description, stack)
      - Its best matching code chunks

    This structure tells Gemini: "here are the top repos ranked by
    relevance to this topic — now make a qualitative comparison."

    The score is shown explicitly so Gemini can factor in how closely
    each repo matched the topic, not just answer based on code quality alone.
    """
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

    The summarized_history field replaces older verbatim turns once
    the context window management has compressed them.
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
        history_parts.append(
            f"{turn['role'].capitalize()}: {turn['content']}"
        )
    history_str = "\n\n".join(history_parts) if history_parts else "None"

    chunks_text = "\n\n---\n\n".join(
        f"File: {c.get('file_path')} (chunk {c.get('chunk_index')})\n"
        f"Similarity: {c.get('score', 0):.2f}\n\n{c.get('text', '')}"
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
        session   Current session dict or None. For repo_specific questions,
                  pass the returned session back on the next call to preserve
                  conversation history, seen chunks, and active repo context.

    Returns:
        (answer_str, updated_session)

        updated_session is None for list_repos, cross_repo_metadata, and
        cross_repo_semantic — these don't maintain session state.

        updated_session is the mutated session dict for repo_specific queries.
        Pass it back on the next call to continue the conversation.

    Full flow per call:
        router → classify question type
        retriever → fetch appropriate context
        prompt builder → assemble Gemini prompt
        Gemini → generate answer
        session update → append turn to history (repo_specific only)
    """
    client      = get_client()
    active_repo = session["active_repo"] if session else None

    print(f"\n[engine] Question: '{question}'")
    route      = classify_question(question, active_repo=active_repo)
    query_type = route["type"]
    print(f"[engine] Routed as: {route}")

    # -----------------------------------------------------------------------
    # list_repos — show all repos from metadata
    # -----------------------------------------------------------------------
    if query_type == "list_repos":
        repos  = list_all_repos()
        prompt = _build_metadata_prompt(question, repos)
        answer = _generate_answer(prompt, client)
        return answer, None   # resets session

    # -----------------------------------------------------------------------
    # cross_repo_metadata — answer from stored metadata, no vector search
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
        return answer, None   # resets session

    # -----------------------------------------------------------------------
    # cross_repo_semantic — semantic search across all code chunks
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
        return answer, None   # resets session

    # -----------------------------------------------------------------------
    # cross_repo_comparative — rank repos fairly, then compare
    # -----------------------------------------------------------------------
    if query_type == "cross_repo_comparative":
        ranked_repos = retrieve_cross_repo_comparative(question)
        if not ranked_repos:
            return (
                "I couldn't find enough relevant code across your repos to "
                "make a comparison. Make sure your repos are indexed with "
                "`python cli.py index --mode full`.",
                None,
            )
        prompt = _build_comparative_prompt(question, ranked_repos)
        answer = _generate_answer(prompt, client)
        return answer, None   # resets session

    # -----------------------------------------------------------------------
    # repo_specific — deep dive with session management
    # -----------------------------------------------------------------------
    repo_name = route["repo"]

# Normalize repo name casing against what's actually indexed.
    all_repos = list_all_repos()
    matched_repo = next(
        (r for r in all_repos if r["repo_name"].lower() == repo_name.lower()),
        None,
    )
    if matched_repo:
        repo_name = matched_repo["repo_name"]  # use the exact casing from the dataset

    # Start new session or continue existing one.
    if session is None or session.get("active_repo") != repo_name:
        repo_meta = matched_repo or {"repo_name": repo_name}
        session = _new_session(repo_name, repo_meta)
        print(f"[engine] New session: {repo_name}")
    else:
        print(f"[engine] Continuing session: {repo_name}")

    # Compress old history if needed.
    session = _manage_context_window(session, client)

    # Retrieve relevant chunks.
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
