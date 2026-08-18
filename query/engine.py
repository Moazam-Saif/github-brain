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

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from gemini_client import get_client, GEMINI_MODEL

from query.router    import classify_question
from query.retriever import (retrieve_cross_repo_metadata,
                             retrieve_cross_repo_semantic,
                             retrieve_cross_repo_comparative,
                             retrieve_repo_specific,
                             make_seen_chunk_ids)
from indexer.deeplake_store import list_all_repos
from indexer.github_client  import GitHubClient

import contextlib

@contextlib.contextmanager
def _timed(label: str):
    """
    TEMPORARY diagnostic timer — prints how long a block took. Added to
    find the real source of the "one query takes 5 minutes" report (see
    INTEGRATION_PROGRESS.md) before proposing any performance fix, rather
    than guessing. Safe to leave in permanently (negligible overhead) or
    strip out once the bottleneck is identified and fixed.
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  [TIMING] {label}: {elapsed:.2f}s")

# Default branch used when fetching file content for the result dict.
# Neither list_all_repos() nor per-chunk metadata carry a branch field
# today, so there's nothing more specific to use. If get_file_content()
# 404s because a repo's default branch isn't "main", _build_result()
# just returns an empty FileItem for that file rather than crashing —
# see the "Failure handling" rule in INTEGRATION_PLAN.md Section 5.
DEFAULT_BRANCH = "main"


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
    about the UI?" is correctly routed back to these SAME repos. The caller
    in query() also reads session["comparison_history"] when the follow-up
    targets the same repo set, and passes it into _build_comparative_prompt()
    so Gemini has continuity with what was already discussed in this
    comparison rather than starting fresh every turn.
    """
    return {
        "active_repo":         None,              # never set — distinguishes from repo_specific
        "comparison_repos":    comparison_repos,   # repos just compared
        "comparison_history":  [],                 # filled in by the caller after this returns
        "all_repos_cache":     all_repos_cache,     # FIX 2 pattern: cached repo list, reused
    }


def _new_context_session(question: str, answer: str) -> dict:
    """
    Create a minimal session after a list_repos, cross_repo_metadata, or
    cross_repo_semantic answer, so the NEXT turn has cross-turn context.

    THE PROBLEM THIS FIXES:
      These three query types previously returned session=None, discarding
      the just-answered exchange entirely. If the next turn was repo_specific
      (e.g. "tell me how Encrypted-Virtual-Drive applies it"), the session
      started fresh with empty conversation_history. The enriched query was
      built from that empty history — so "it" had no referent, and the
      embedding searched for "how Encrypted-Virtual-Drive applies it" with
      no idea what "it" meant. The retriever found generic tech-stack chunks
      instead of authentication-specific ones.

    THE FIX:
      Return a lightweight session carrying just the prior Q&A in
      prior_exchange. When the next turn starts a repo_specific session
      (_new_session), the repo_specific branch checks for prior_exchange and
      seeds conversation_history with it before the first retrieval call.
      _build_enriched_query() then sees the prior exchange as context and
      resolves pronouns/references correctly.

    This session carries no active_repo, no comparison_repos — it's purely
    a cross-turn context bridge. The repo_specific branch already handles the
    case where active_repo is None (starts a new session), so nothing else
    in the codebase needs to change.
    """
    return {
        "active_repo":      None,
        "comparison_repos": None,
        "prior_exchange":   [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer[:300] + "..." if len(answer) > 300 else answer},
        ],
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


RESPONSE_STRUCTURE_INSTRUCTION = """

Structure your ENTIRE response using exactly these five labeled parts, in
this order, each starting on its own line with the marker shown:

---SUMMARY---
Write 2-4 full sentences summarizing the answer. This must stand alone and
make sense without reading the rest of the response — it will be shown to
the user separately, before anything else. Do not just repeat the first
sentence of your answer here; write a real, self-contained summary.

---SECTIONS---
List each section heading you will use in your answer below, one per line,
like:
[S1] First section heading
[S2] Second section heading
Use short, descriptive headings (a few words each). Use as many or as few
sections as the answer actually needs — at least one.

---ANSWER---
Write your full answer here, organized under markdown headings that exactly
match the section headings listed above (## First section heading, etc).
Do NOT repeat the summary here. When you reference a specific code chunk
(numbered below), cite it inline like [N] right after the reference, e.g.
"the Posts model [1] defines each blog entry". Additionally, when you
mention a SPECIFIC, REAL expression from the code you were given — a
variable name, function/method call, config key, class name, etc., taken
VERBATIM from a chunk's text below — mark it inline like {R1}, {R2}, ...
immediately after the expression, e.g. "calls request.form.get{R1} to read
the field". Each {RN} marker must have a matching entry in the
---REFERENCES--- section below. Do NOT mark generic English words, file
names mentioned only in passing, or anything you are paraphrasing rather
than quoting — only real code tokens that appear verbatim in the chunk text
you were given, because each one will become a link to that EXACT line in
the source file, and a wrong or vague reference is worse than none.

---CHUNKS---
For each numbered code chunk provided to you, write:
[N] Short heading (a few words)
One to three sentences explaining what that chunk does and how it relates
to the question.
Include exactly one [N] block per chunk, in the same order and using the
same chunk numbers given to you.

---REFERENCES---
For each {RN} marker you used in your answer, write one line in this exact
format:
[RN] expression | chunk N | line L
- "expression" is the EXACT verbatim text you quoted (matching {RN})
- "chunk N" is the chunk number (matching the [N] chunks given to you)
- "line L" is the SINGLE line number where that expression literally
  appears — read it directly from the "line_number: code" annotations in
  the chunk text you were given; do not guess or estimate.
If you used no {RN} markers, leave this section empty (just the marker
line, nothing after it) — do not invent references to fill it."""


def _build_semantic_prompt(question: str, chunks: list[dict]) -> str:
    """Prompt for cross_repo_semantic questions."""
    chunks_text = "\n\n---\n\n".join(
        f"[{i + 1}] Repo: {c.get('repo_name')} | File: {c.get('file_path')} "
        f"(lines {c.get('start_line', '?')}-{c.get('end_line', '?')})\n"
        f"Similarity score: {c.get('score', 0):.2f}\n\n{_annotate_chunk_with_line_numbers(c)}"
        for i, c in enumerate(chunks)
    )

    return f"""You are GitHub Brain, an assistant that helps a developer understand their GitHub repositories.

Answer the following question using ONLY the code context provided below.
Reference specific repo names, file paths, and code where relevant.
If the answer cannot be determined from the provided context, say so honestly.

Question: {question}

Retrieved code context:
{chunks_text}
{RESPONSE_STRUCTURE_INSTRUCTION}"""


def _build_comparative_prompt(
    question: str,
    ranked_repos: list[dict],
    comparison_history: Optional[list[dict]] = None,
) -> str:
    """
    Prompt for cross_repo_comparative questions.

    comparison_history (optional): prior Q&A turns from THIS SAME comparison
    session (e.g. the database-design answer when this turn asks about the
    UI instead). When provided, it's rendered above the current question so
    Gemini has continuity — knows what was already said about these same two
    repos and doesn't need to be told again, can reference it, and won't
    contradict itself. Defaults to None for the first turn of a comparison,
    where there's nothing to include yet.
    """
    sections = []
    chunk_counter = 0  # global sequential numbering, must match flat_chunks order
    for repo in ranked_repos:
        chunk_lines = []
        for c in repo.get("chunks", []):
            chunk_counter += 1
            chunk_lines.append(
                f"  [{chunk_counter}] {c.get('file_path')} chunk {c.get('chunk_index')}"
                f" (lines {c.get('start_line', '?')}-{c.get('end_line', '?')}, "
                f"score {c.get('score', 0):.2f})\n  {_annotate_chunk_with_line_numbers(c)}"
            )
        chunks_text = "\n\n".join(chunk_lines)
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

    # Render prior turns in this comparison, if any.
    history_block = ""
    if comparison_history:
        history_lines = "\n\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in comparison_history
        )
        history_block = (
            f"\nEarlier in this comparison:\n{history_lines}\n"
            f"\nDon't repeat what was already covered above — focus on the new question.\n"
        )

    return f"""You are GitHub Brain, an assistant that helps a developer understand their GitHub repositories.

The developer is asking a comparative question about their repos.
Below are the most relevant repos ranked by how closely their code matches the topic,
along with their most relevant code chunks.

Compare these repos honestly based on the code shown. Reference specific file paths,
function names, and implementation details. If one repo clearly does something better,
say so and explain why. If they're comparable, say that too.
{history_block}
Question: {question}

Ranked repositories (by topic relevance):
{repos_text}
{RESPONSE_STRUCTURE_INSTRUCTION}"""


def _annotate_chunk_with_line_numbers(chunk: dict) -> str:
    """
    Render a chunk's text with each CODE line prefixed by its real 1-indexed
    line number in the source file, so Gemini can cite exact lines when it
    references specific code (see RESPONSE_STRUCTURE_INSTRUCTION's
    ---REFERENCES--- block, and INTEGRATION_PROGRESS.md's "how the text is
    going to be connected to the source files" discussion).

    chunker.py prepends a one-line ("File: ... | Role: ...") or two-line
    ("File: ... | Role: ... | Purpose: ...") context header before the
    actual code, followed by a blank line. That header is NOT part of the
    file's real content — it has no meaningful line number — so it's kept
    unannotated; numbering starts at chunk["start_line"] from the first
    real code line onward.

    Falls back to the chunk's raw text, unannotated, if start_line/end_line
    aren't present (e.g. data indexed before this feature existed and not
    yet re-indexed) — never crashes on missing fields.
    """
    text = chunk.get("text", "")
    start_line = chunk.get("start_line")
    end_line = chunk.get("end_line")

    if start_line is None or end_line is None:
        return text

    # Split off the header: everything up to and including the first blank
    # line (chunker.py always inserts "\n\n" between header and code).
    if "\n\n" in text:
        header, _, code = text.partition("\n\n")
    else:
        header, code = "", text

    code_lines = code.split("\n")
    numbered = []
    line_num = start_line
    for line in code_lines:
        numbered.append(f"{line_num}: {line}")
        line_num += 1

    annotated_code = "\n".join(numbered)
    return f"{header}\n\n{annotated_code}" if header else annotated_code


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
        f"[{i + 1}] File: {c.get('file_path')} (chunk {c.get('chunk_index')}, "
        f"lines {c.get('start_line', '?')}-{c.get('end_line', '?')})\n"
        f"Relevance: {c.get('rerank_score', c.get('score', 0)):.2f}\n\n"
        f"{_annotate_chunk_with_line_numbers(c)}"
        for i, c in enumerate(chunks)
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
If something is unclear from the retrieved code, say so — don't guess.
{RESPONSE_STRUCTURE_INSTRUCTION}"""


# ---------------------------------------------------------------------------
# Structured result building (for the API layer — see INTEGRATION_PLAN.md)
# ---------------------------------------------------------------------------

SECTION_MARKERS = ["---SUMMARY---", "---SECTIONS---", "---ANSWER---", "---CHUNKS---", "---REFERENCES---"]
CHUNK_BLOCK_RE   = re.compile(r"^\[(\d+)\]\s*(.*)$")
SECTION_LIST_RE  = re.compile(r"^\[S(\d+)\]\s*(.*)$")
# "[RN] expression | chunk N | line L" — three pipe-separated fields after
# the [RN] marker. Whitespace around pipes is tolerant since Gemini's exact
# spacing isn't guaranteed.
REFERENCE_RE     = re.compile(
    r"^\[R(\d+)\]\s*(.+?)\s*\|\s*chunk\s*(\d+)\s*\|\s*line\s*(\d+)\s*$",
    re.IGNORECASE,
)


def _parse_references(block_text: str) -> dict:
    """
    Parse the ---REFERENCES--- block into
    {int_r_number: {"expression": str, "chunk": int, "line": int}}.

    Unlike _parse_numbered_blocks, this is a single-line-per-entry format
    (no multi-line body), so it needs its own parser. Malformed lines
    (wrong field count, non-numeric chunk/line) are silently skipped rather
    than raising — a bad reference line just means that {RN} marker won't
    resolve to a link in the frontend (renders as plain text), which is
    the correct partial-degrade behavior: better to lose one clickable
    reference than to crash the whole response over it.
    """
    references: dict[int, dict] = {}
    for line in block_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = REFERENCE_RE.match(line)
        if not m:
            continue
        r_num, expression, chunk_num, line_num = m.groups()
        references[int(r_num)] = {
            "expression": expression.strip(),
            "chunk": int(chunk_num),
            "line": int(line_num),
        }
    return references


def _parse_numbered_blocks(block_text: str, key_re: "re.Pattern") -> dict:
    """
    Shared helper: parse a block of "[key] heading\ntext..." groups into
    {int_key: {"heading": str, "text": str}}, skipping any group missing a
    non-empty heading or non-empty text (partial degrade — see
    INTEGRATION_PROGRESS.md's "chunk text should be prose" / course-correction
    entries for why a malformed individual group is dropped rather than
    guessed at or allowed to break the whole parse).
    """
    results: dict[int, dict] = {}
    current_num = None
    current_heading = ""
    current_lines: list[str] = []

    def _flush():
        text = " ".join(l.strip() for l in current_lines if l.strip()).strip()
        if current_num is not None and current_heading.strip() and text:
            results[current_num] = {"heading": current_heading.strip(), "text": text}

    for line in block_text.split("\n"):
        m = key_re.match(line.strip())
        if m:
            _flush()
            current_num = int(m.group(1))
            current_heading = m.group(2)
            current_lines = []
        elif current_num is not None:
            current_lines.append(line)
    _flush()
    return results


def _parse_structured_response(raw: str) -> dict:
    """
    Parse a full (non-streaming) Gemini response in the RESPONSE_STRUCTURE_INSTRUCTION
    format into its five parts.

    Returns:
      {
        "summary":  str,                          # from ---SUMMARY---, or "" if missing
        "sections": [{"heading": str}, ...],       # from ---SECTIONS---, in order
        "answer":   str,                           # from ---ANSWER---
        "chunk_blocks": {int: {"heading","text"}}, # from ---CHUNKS---
        "references": {int: {"expression","chunk","line"}}, # from ---REFERENCES---
      }

    Fallback behavior (partial degrade, matches the streaming parser's
    behavior for consistency): if ---SUMMARY--- or ---SECTIONS--- markers are
    missing, those fields come back empty/[] and the frontend falls back to
    its existing heuristics (e.g. no jump-nav shown). If ---ANSWER--- is
    missing entirely (total malformation), the raw text is used as-is as the
    answer, matching the old CHUNK_BLOCK_MARKER-missing fallback. Missing or
    empty ---REFERENCES--- just means no {RN} markers resolve to links —
    the frontend treats those as plain text, never a crash.
    """
    parts = {"summary": "", "sections": [], "answer": raw.strip(), "chunk_blocks": {}, "references": {}}

    if "---SUMMARY---" not in raw:
        return parts

    # Split on all markers we recognize, keeping track of which section each
    # piece belongs to.
    pattern = "(" + "|".join(re.escape(m) for m in SECTION_MARKERS) + ")"
    pieces = re.split(pattern, raw)
    # pieces alternates: [pre-marker junk, marker, text, marker, text, ...]

    current_marker = None
    buffers = {m: "" for m in SECTION_MARKERS}
    for piece in pieces:
        if piece in SECTION_MARKERS:
            current_marker = piece
        elif current_marker:
            buffers[current_marker] += piece

    parts["summary"] = buffers["---SUMMARY---"].strip()

    sections_raw = _parse_numbered_blocks(buffers["---SECTIONS---"], SECTION_LIST_RE)
    parts["sections"] = [
        {"heading": sections_raw[k]["heading"]}
        for k in sorted(sections_raw)
    ] if sections_raw else _parse_section_headings_fallback(buffers["---SECTIONS---"])

    parts["answer"] = buffers["---ANSWER---"].strip() or parts["answer"]
    parts["chunk_blocks"] = _parse_numbered_blocks(buffers["---CHUNKS---"], CHUNK_BLOCK_RE)
    parts["references"] = _parse_references(buffers["---REFERENCES---"])

    return parts


def _parse_section_headings_fallback(sections_text: str) -> list[dict]:
    """
    ---SECTIONS--- entries have no body text (just "[S1] Heading" lines), so
    _parse_numbered_blocks's "needs non-empty text too" rule would drop every
    entry. Parse headings directly instead — same [S1]/[S2] regex, but a
    heading alone is a complete, valid entry here.
    """
    headings = []
    for line in sections_text.split("\n"):
        m = SECTION_LIST_RE.match(line.strip())
        if m and m.group(2).strip():
            headings.append({"heading": m.group(2).strip()})
    return headings


def _split_answer_into_sections(answer: str, section_list: list[dict]) -> list[dict]:
    """
    Split the ---ANSWER--- text into per-section slices, keyed by its own
    "## Heading" markdown lines — NOT by re-asking Gemini for anything new,
    just parsing structure that's already there (the RESPONSE_STRUCTURE_INSTRUCTION
    prompt already asks for section headings in ---ANSWER--- to match
    ---SECTIONS---).

    Returns a list of:
      {"heading": str, "body": str, "chunk_indices": [int, ...]}
    in the same order as `section_list` (falls back to the order headings
    actually appear in `answer` if that differs — e.g. Gemini reordered or
    dropped one; safer than assuming perfect agreement between the two
    independently-parsed parts of the same response).

    chunk_indices are which [N] markers appear in that section's own body
    text — this is what SourceViewer/App.jsx use to auto-select a topic tab
    when a chunk/source file is clicked (see INTEGRATION_PROGRESS.md's
    "topics horizontal bar" entry).

    If ---ANSWER--- has no "## " headings at all (total malformation),
    returns a single section with an empty heading and the whole answer as
    its body — the frontend's tab bar just won't render in that case and it
    falls back to showing the answer as one block, same partial-degrade
    philosophy as everywhere else in this file.
    """
    HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(HEADING_RE.finditer(answer))

    if not matches:
        return [{"heading": "", "body": answer.strip(), "chunk_indices": []}]

    slices = []
    for i, m in enumerate(matches):
        heading    = m.group(1).strip()
        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(answer)
        body       = answer[body_start:body_end].strip()
        chunk_indices = sorted({int(n) for n in re.findall(r"\[(\d+)\]", body)})
        slices.append({
            "heading": heading,
            "body": body,
            "chunk_indices": chunk_indices,
        })
    return slices


def _build_result(
    query_type: str,
    answer: str,
    chunks: list[dict],
    repo: Optional[str],
    repo_metadata: Optional[dict],
    github_client: GitHubClient,
) -> dict:
    """
    Assemble the structured result dict consumed by api/server.py.

    Fetches file content from GitHub for every unique file referenced by
    `chunks`, de-duplicated by (repo_name, file_path), CONCURRENTLY via a
    thread pool (github_client is sync/requests-based, so this is I/O-bound
    parallelism, not true async) — this is "Option C" from the streaming
    design discussion: files are fetched in parallel rather than serially,
    so the wait between "chunks known" and "files ready" is roughly the
    slowest single fetch, not the sum of all of them.

    Never raises — a GitHub fetch failure for one file just leaves that
    FileItem empty; the text answer is always returned regardless.

    summary/sections/answer/ChunkItem.heading&text all come from parsing the
    SAME Gemini call that produced `answer` (see RESPONSE_STRUCTURE_INSTRUCTION
    / prompt builders, and _parse_structured_response) — no extra API call.
    If a section is missing/malformed, it falls back to an empty value rather
    than a fabricated one — partial degrade, see INTEGRATION_PROGRESS.md.
    """
    parsed = _parse_structured_response(answer)

    chunk_items = []
    # chunk 0-1 -> "a", chunk 2-3 -> "b", chunk 4+ -> "c" (plan Section 4, ChunkItem.col)
    col_map = {0: "a", 1: "a", 2: "b", 3: "b"}
    for i, chunk in enumerate(chunks):
        block = parsed["chunk_blocks"].get(i + 1)
        chunk_items.append({
            "index":       i + 1,
            "heading":     block["heading"] if block else "",
            "text":        block["text"] if block else "",
            "file_path":   chunk.get("file_path", ""),
            "file_role":   chunk.get("file_role", "other"),
            "chunk_index": chunk.get("chunk_index", 0),
            "repo_name":   chunk.get("repo_name", ""),
            "score":       chunk.get("rerank_score", chunk.get("score", 0.0)),
            "col":         col_map.get(i, "c"),
            # Real line range from chunker.py (see start_line/end_line
            # addition, INTEGRATION_PROGRESS.md's "how the text is going to
            # be connected to the source files" work) — None for chunks
            # from data indexed before this field existed, in which case
            # the frontend falls back to its old Strategy A estimate.
            "start_line":  chunk.get("start_line"),
            "end_line":    chunk.get("end_line"),
        })

    # Unique (repo, file_path) pairs to fetch, de-duplicated.
    to_fetch = {}
    for chunk in chunks:
        repo_n    = chunk.get("repo_name", "")
        file_path = chunk.get("file_path", "")
        key       = f"{repo_n}::{file_path}"
        if key not in to_fetch and file_path:
            to_fetch[key] = (repo_n, file_path)

    def _fetch_one(key_repo_path):
        key, (repo_n, file_path) = key_repo_path
        try:
            content = github_client.get_file_content(
                repo_n, file_path, DEFAULT_BRANCH
            ) or ""
        except Exception as e:
            print(f"  [engine] get_file_content failed for "
                  f"{repo_n}/{file_path}: {e}")
            content = ""
        return file_path, content

    files = {}
    if to_fetch:
        with ThreadPoolExecutor(max_workers=min(8, len(to_fetch))) as pool:
            for file_path, content in pool.map(_fetch_one, to_fetch.items()):
                files[file_path] = {
                    "name":     file_path,
                    "content":  content,
                    "language": file_path.rsplit(".", 1)[-1] if "." in file_path else "text",
                    "lines":    content.splitlines() if content else [],
                }

    section_content = _split_answer_into_sections(parsed["answer"], parsed["sections"])

    # Resolve each {RN} reference: translate its "chunk N" (1-indexed
    # position in the prompt, matching chunk_items above) into the real
    # file_path, and sanity-check the claimed line actually falls within
    # that chunk's real start_line/end_line range — Gemini could hallucinate
    # a line number even when told to read it from the annotated text, so
    # this is a real validation, not just a lookup. A reference that fails
    # this check is dropped entirely (not clamped/guessed) — a reference
    # pointing at the wrong place is worse than no reference at all, same
    # partial-degrade philosophy as everywhere else in this file.
    references = {}
    for r_num, ref in parsed["references"].items():
        chunk_num = ref["chunk"]
        if chunk_num < 1 or chunk_num > len(chunks):
            continue
        source_chunk = chunks[chunk_num - 1]
        start_line = source_chunk.get("start_line")
        end_line = source_chunk.get("end_line")
        line = ref["line"]
        if start_line is None or end_line is None:
            # Chunk predates the start_line/end_line field (not yet
            # re-indexed) — can't validate, so don't emit a reference that
            # might point at the wrong line.
            continue
        if not (start_line <= line <= end_line):
            print(f"  [engine] Dropping reference R{r_num}: line {line} is "
                  f"outside chunk {chunk_num}'s real range "
                  f"({start_line}-{end_line}) — likely hallucinated.")
            continue
        references[r_num] = {
            "expression": ref["expression"],
            "file_path": source_chunk.get("file_path", ""),
            "line": line,
        }

    return {
        "query_type":       query_type,
        "repo":             repo,
        "answer":           parsed["answer"],
        "summary":          parsed["summary"] or _fallback_summary(parsed["answer"]),
        "sections":         parsed["sections"],
        "references":       references,
        "section_content":  section_content,
        "chunks":           chunk_items,
        "files":            files,
        "repo_metadata":    repo_metadata,
    }


def _fallback_summary(answer: str) -> str:
    """
    Only used when Gemini's response is missing the ---SUMMARY--- marker
    entirely (total malformation, not the normal path). Same heuristic the
    old _extract_summary used — a short string cut, not a real summary — so
    even in this rare failure case the frontend has SOMETHING to show rather
    than an empty summary card.
    """
    for sep in [". ", "\n"]:
        idx = answer.find(sep)
        if idx != -1 and idx < 200:
            return answer[:idx + 1].strip()
    return answer[:200].strip()


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


def generate_answer_stream(prompt: str, client, max_retries: int = 3):
    """
    Streaming counterpart to _generate_answer. Calls Gemini's
    generate_content_stream() and yields SEMANTIC events as each of the four
    RESPONSE_STRUCTURE_INSTRUCTION sections is confirmed complete — NOT raw
    token-by-token text. This is what api/server.py's /ask SSE endpoint
    consumes directly.

    Yields dicts of the form:
      {"event": "summary",  "data": {"summary": str}}
      {"event": "sections", "data": {"sections": [{"heading": str}, ...]}}
      {"event": "answer",   "data": {"answer": str}}   # full ---ANSWER--- text
      {"event": "chunk_blocks", "data": {chunk_num: {"heading","text"}}}
      {"event": "error", "data": {"message": str}}      # on failure, terminal

    A section is only "confirmed complete" once the NEXT marker (or stream
    end, for the last section) has been seen in the buffer — we can't know
    ---SUMMARY--- is finished until ---SECTIONS--- starts arriving, since the
    model could still be mid-sentence. This means the summary event fires as
    soon as the first few sentences plus the next marker have streamed in —
    typically a small fraction of the total response — rather than waiting
    for the whole answer, which is the whole point of streaming this at all.

    Same retry/rate-limit handling as _generate_answer, but retries restart
    the WHOLE stream (partial output from a failed attempt is discarded) —
    there's no way to resume a partial Gemini stream.
    """
    for attempt in range(max_retries):
        try:
            buffer = ""
            emitted = set()  # which of the 4 markers we've already emitted for

            for chunk in client.models.generate_content_stream(
                model=GEMINI_MODEL, contents=prompt
            ):
                buffer += chunk.text or ""

                # SUMMARY is complete once ---SECTIONS--- has started arriving.
                if "summary" not in emitted and "---SECTIONS---" in buffer:
                    summary_text = buffer.split("---SUMMARY---", 1)[-1]
                    summary_text = summary_text.split("---SECTIONS---", 1)[0].strip()
                    if summary_text:
                        emitted.add("summary")
                        yield {"event": "summary", "data": {"summary": summary_text}}

                # SECTIONS is complete once ---ANSWER--- has started arriving.
                if "sections" not in emitted and "---ANSWER---" in buffer:
                    sections_text = buffer.split("---SECTIONS---", 1)[-1]
                    sections_text = sections_text.split("---ANSWER---", 1)[0].strip()
                    parsed_sections = _parse_section_headings_fallback(sections_text)
                    emitted.add("sections")
                    yield {"event": "sections", "data": {"sections": parsed_sections}}

                # ANSWER is complete once ---CHUNKS--- has started arriving.
                if "answer" not in emitted and "---CHUNKS---" in buffer:
                    answer_text = buffer.split("---ANSWER---", 1)[-1]
                    answer_text = answer_text.split("---CHUNKS---", 1)[0].strip()
                    if answer_text:
                        emitted.add("answer")
                        yield {"event": "answer", "data": {"answer": answer_text}}

            time.sleep(4)

            # Stream ended — emit whatever sections never got a "next marker"
            # (typically just CHUNKS, sometimes more if the model omitted
            # markers near the end). Re-parse the full buffer for anything
            # not yet emitted rather than guessing at partial boundaries.
            full = _parse_structured_response(buffer)
            if "summary" not in emitted and full["summary"]:
                yield {"event": "summary", "data": {"summary": full["summary"]}}
            if "sections" not in emitted and full["sections"]:
                yield {"event": "sections", "data": {"sections": full["sections"]}}
            if "answer" not in emitted:
                yield {"event": "answer", "data": {"answer": full["answer"]}}
            yield {"event": "chunk_blocks", "data": {"chunk_blocks": full["chunk_blocks"]}}
            yield {"event": "references", "data": {"references": full["references"]}}
            return

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"  [engine] Rate limit on stream. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [engine] Streaming generation error (attempt {attempt + 1}): {e}")

    yield {"event": "error", "data": {"message": "I encountered an error generating a response. Please try again."}}


# ---------------------------------------------------------------------------
# Public: main query function
# ---------------------------------------------------------------------------

def query(
    question: str,
    session: Optional[dict] = None,
    on_event=None,
) -> tuple[str, Optional[dict], dict]:
    """
    Process a user question end-to-end and return an answer.

    This is the only function cli.py (and now api/server.py) needs to call.

    Parameters:
        question  The user's natural language question.
        session   Current session dict or None.
        on_event  Optional callback(event: dict) for streaming. When provided,
          the three chunk-bearing branches (cross_repo_semantic,
          cross_repo_comparative, repo_specific) call Gemini via
          generate_answer_stream() instead of _generate_answer(), and invoke
          on_event(...) for each semantic event (summary/sections/answer/
          chunk_blocks/error) AS IT ARRIVES — see generate_answer_stream's
          docstring for the event shapes. The full answer text is still
          accumulated and used exactly as before for session history,
          _build_result(), etc. — every line of branch logic below this
          point is UNCHANGED whether on_event is set or not; only the
          generation call itself differs. When on_event is None (cli.py's
          case), this is byte-for-byte the same as before streaming existed.
          list_repos/cross_repo_metadata and all error/guard paths ignore
          on_event entirely — they're fast and don't stream.

    Returns:
        (answer_str, updated_session, result)

        `result` is the structured dict consumed by the API layer — see
        INTEGRATION_PLAN.md Section 4. cli.py doesn't need it and discards
        it: `answer, session, _ = query(...)`.

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
    client        = get_client()
    github_client = GitHubClient(token=os.getenv("GITHUB_TOKEN"))
    active_repo   = session["active_repo"] if session else None

    def _generate(prompt: str) -> str:
        """
        Shared generation step for the 3 chunk-bearing branches. Streams via
        on_event when provided, otherwise calls _generate_answer exactly as
        before. Always returns the full answer text either way, so callers
        don't need an if/else — this is the ONLY thing that changes between
        streaming and non-streaming mode.
        """
        if on_event is None:
            return _generate_answer(prompt, client)

        full_answer = ""
        for event in generate_answer_stream(prompt, client):
            on_event(event)
            if event["event"] == "error":
                return event["data"]["message"]
            # Reconstruct the full text so downstream logic (session
            # history, _build_result's re-parse) works unchanged. answer
            # event carries the ---ANSWER--- section text; we still need
            # summary/sections/chunk_blocks folded back in so
            # _build_result's _parse_structured_response call (which
            # re-parses the FULL text from scratch) finds all 4 markers.
            if event["event"] == "summary":
                full_answer += f"---SUMMARY---\n{event['data']['summary']}\n\n"
            elif event["event"] == "sections":
                full_answer += "---SECTIONS---\n" + "\n".join(
                    f"[S{i+1}] {s['heading']}"
                    for i, s in enumerate(event["data"]["sections"])
                ) + "\n\n"
            elif event["event"] == "answer":
                full_answer += f"---ANSWER---\n{event['data']['answer']}\n\n"
            elif event["event"] == "chunk_blocks":
                full_answer += "---CHUNKS---\n" + "\n\n".join(
                    f"[{num}] {block['heading']}\n{block['text']}"
                    for num, block in event["data"]["chunk_blocks"].items()
                ) + "\n\n"
            elif event["event"] == "references":
                full_answer += "---REFERENCES---\n" + "\n".join(
                    f"[R{num}] {ref['expression']} | chunk {ref['chunk']} | line {ref['line']}"
                    for num, ref in event["data"]["references"].items()
                )
        return full_answer

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
        result = _build_result("list_repos", answer, [], None, None, github_client)
        return answer, _new_context_session(question, answer), result

    # -----------------------------------------------------------------------
    # cross_repo_metadata
    # -----------------------------------------------------------------------
    if query_type == "cross_repo_metadata":
        repos = retrieve_cross_repo_metadata()
        if not repos:
            answer = ("No repos are indexed yet. "
                       "Run `python cli.py index --mode full` first.")
            result = _build_result(
                "cross_repo_metadata", answer, [], None, None, github_client
            )
            result["error"] = "not_indexed"
            return answer, None, result
        prompt = _build_metadata_prompt(question, repos)
        answer = _generate_answer(prompt, client)
        result = _build_result(
            "cross_repo_metadata", answer, [], None, None, github_client
        )
        return answer, _new_context_session(question, answer), result

    # -----------------------------------------------------------------------
    # cross_repo_semantic
    # -----------------------------------------------------------------------
    if query_type == "cross_repo_semantic":
        chunks = retrieve_cross_repo_semantic(question)
        if not chunks:
            answer = ("I couldn't find relevant code for that question across your repos. "
                       "Make sure your repos are indexed with "
                       "`python cli.py index --mode full`.")
            result = _build_result(
                "cross_repo_semantic", answer, [], None, None, github_client
            )
            result["error"] = "not_indexed"
            return answer, None, result
        prompt = _build_semantic_prompt(question, chunks)
        answer = _generate(prompt)
        result = _build_result(
            "cross_repo_semantic", answer, chunks, None, None, github_client
        )
        return answer, _new_context_session(question, answer), result

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

                # FIX 13: detect a PARTIAL match — some names matched, some
                # didn't (e.g. "compare CorpLaw-AI and Claim-Verification-AI"
                # where the second is a typo for Claim-Verification-Automation).
                # Previously this silently proceeded with just the matched
                # repo(s), producing a confusing answer that looks like a
                # comparison but is actually only describing one repo, with
                # no indication to the user that a name was dropped.
                if named_repos and len(named_repos) < len(raw_named):
                    unmatched = [
                        n for n in raw_named
                        if n.lower() not in {m.lower() for m in named_repos}
                    ]
                    available = [r["repo_name"] for r in all_repos_for_cmp]
                    print(f"[engine] WARNING: partial named-repo match. "
                          f"Matched: {named_repos}, unmatched: {unmatched}")
                    answer = (
                        f"I couldn't find a repo named "
                        f"'{', '.join(unmatched)}' in your indexed repos, so I "
                        f"can't run this comparison. Your indexed repos are: "
                        f"{', '.join(available)}. Check the spelling and try again."
                    )
                    result = _build_result(
                        "cross_repo_comparative", answer, [], None, None, github_client
                    )
                    result["error"] = "repo_not_found"
                    # preserve whatever session existed — don't wipe it
                    return answer, session, result

                if not named_repos:
                    print("[engine] All named repos failed normalization. "
                          "Falling back to global ranking.")
            else:
                print("[engine] WARNING: list_all_repos() returned empty during "
                      "comparative. Falling back to global ranking.")

        # Is this a follow-up on an EXISTING comparison session, talking
        # about the SAME repos? If so, pull its prior history so the prompt
        # has continuity instead of starting from scratch every turn.
        # Compared as sets so repo order/casing differences don't break the
        # match (named_repos is already normalized; session repos were too).
        prior_history = []
        is_followup = bool(
            session
            and session.get("comparison_repos")
            and named_repos
            and set(session["comparison_repos"]) == set(named_repos)
        )
        if is_followup:
            prior_history = session.get("comparison_history", [])
            print(f"[engine] Continuing comparison session: {named_repos} "
                  f"({len(prior_history)} prior turns)")

        ranked_repos = retrieve_cross_repo_comparative(
            question,
            named_repos=named_repos,   # None → global ranking, list → ONLY these repos
        )
        if not ranked_repos:
            answer = ("I couldn't find enough relevant code across your repos to "
                       "make a comparison. Make sure your repos are indexed with "
                       "`python cli.py index --mode full`.")
            result = _build_result(
                "cross_repo_comparative", answer, [], None, None, github_client
            )
            result["error"] = "not_indexed"
            return answer, (session if is_followup else None), result
        prompt = _build_comparative_prompt(
            question,
            ranked_repos,
            comparison_history=prior_history,
        )
        answer = _generate(prompt)

        # Flatten all chunks from all ranked repos for the result dict
        # (plan Section 7: cross_repo_comparative chunks are "all chunks
        # from all ranked repos, flattened").
        flat_chunks = [c for repo in ranked_repos for c in repo.get("chunks", [])]
        result = _build_result(
            "cross_repo_comparative", answer, flat_chunks, None, None, github_client
        )

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
        # Extend prior history on a follow-up turn instead of discarding it.
        # prior_history is [] on the first turn of a comparison (is_followup
        # was False), so this naturally degrades to "just this turn" then.
        new_session["comparison_history"] = prior_history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
        return answer, new_session, result

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
        with _timed("list_all_repos() [full _load_all scan]"):
            all_repos = list_all_repos()
        print(f"[engine] Fetched repo list ({len(all_repos)} repos)")

    # FIX 6: Guard against empty repo list — log clearly and return informative error.
    if not all_repos:
        answer = ("I couldn't load the indexed repo list from Deep Lake. "
                   "Check your ACTIVELOOP_TOKEN and ACTIVELOOP_ORG, then try again.")
        result = _build_result(
            "repo_specific", answer, [], repo_name, None, github_client
        )
        result["error"] = "not_indexed"
        return answer, session, result

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
        # FIX 12: preserve whatever session already existed instead of wiping
        # it with None. A typo'd repo name on a follow-up question (e.g. user
        # is mid-conversation about CorpLaw-AI and asks about a misspelled
        # repo) should fail THIS turn only — not destroy the active session.
        # The user can correct the name and continue where they left off.
        answer = (
            f"I couldn't find a repo named '{repo_name}' in your indexed repos. "
            f"Try asking about one of your indexed repos by its exact name, or "
            f"run `python cli.py index --mode full` if you haven't indexed yet."
        )
        result = _build_result(
            "repo_specific", answer, [], None, None, github_client
        )
        result["error"] = "repo_not_found"
        return answer, session, result

    if matched_repo["repo_name"] != repo_name:
        print(f"[engine] Normalized repo name: '{repo_name}' → '{matched_repo['repo_name']}'")
    repo_name = matched_repo["repo_name"]

    # Start new session or continue existing one.
    # Normalization above guarantees repo_name casing matches session["active_repo"].
    if session is None or session.get("active_repo") != repo_name:
        # Capture prior_exchange BEFORE overwriting session — _new_session()
        # returns a fresh dict and the incoming session reference is lost after.
        prior_exchange = session.get("prior_exchange", []) if session else []
        session = _new_session(repo_name, matched_repo, all_repos)
        # If the previous turn was a metadata/semantic/list answer, it left a
        # prior_exchange in the session. Seed conversation_history with it so
        # _build_enriched_query() can resolve pronouns and references (e.g.
        # "tell me how Encrypted-Virtual-Drive applies IT" → "it" = authentication
        # from the prior metadata answer). Already truncated to 300 chars in
        # _new_context_session() so it doesn't bloat the enriched query.
        if prior_exchange:
            session["conversation_history"] = prior_exchange
            print(f"[engine] Seeded session with prior context ({len(prior_exchange)} turns)")
        print(f"[engine] New session: {repo_name}")
    else:
        print(f"[engine] Continuing session: {repo_name}")

    # Compress old history if needed.
    with _timed("_manage_context_window()"):
        session = _manage_context_window(session, client)

    # Retrieve relevant chunks via hybrid search + re-rank.
    # seen_chunk_ids is the capped deque from the session (FIX 7).
    with _timed("retrieve_repo_specific() [enrich+HyDE+embed+hybrid_search+rerank]"):
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
        with _timed("_generate() [Gemini answer generation, streamed or not]"):
            answer = _generate(prompt)

    # Append this turn to session history.
    session["conversation_history"].append(
        {"role": "user",      "content": question}
    )
    session["conversation_history"].append(
        {"role": "assistant", "content": answer}
    )

    with _timed("_build_result() [GitHub file fetch + response parsing]"):
        result = _build_result(
            "repo_specific", answer, chunks, repo_name,
            session.get("repo_metadata"), github_client
        )
    return answer, session, result
