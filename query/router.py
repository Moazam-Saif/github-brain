"""
query/router.py
---------------
PURPOSE:
  Classifies every incoming user question into one of five types before
  any search happens. The classification decides which search strategy
  the engine uses, so getting it right is critical.

HOW IT WORKS:
  One Gemini call per question. The prompt describes all five types with
  examples, and Gemini returns a small JSON object. The engine reads that
  JSON and branches accordingly.

  The router also receives the active_repo from the current session (if any).
  This lets it correctly attribute follow-up questions — if the user is
  already deep-diving claimsense and asks "how does the parser work?",
  the router knows that means claimsense, not a cross-repo search.

QUERY TYPES:
  list_repos             → user wants to see/count their repos
                           Handled by: metadata lookup, no vector search
                           Example: "what repos do I have?"

  cross_repo_metadata    → question about a technology, feature, or attribute
                           answerable from stored metadata without reading code
                           Handled by: filter list_all_repos() in Python
                           Example: "do I have anything using Redis?"
                                    "which repos are deployed?"

  cross_repo_semantic    → question about whether a concept/feature exists
                           anywhere across repos — needs code, not metadata
                           Handled by: vector search across all chunks, top-6
                           Example: "do any of my repos implement rate limiting?"
                                    "do I have anything using WebSockets?"

  cross_repo_comparative → question that RANKS or COMPARES repos against each
                           other on some quality or implementation — needs
                           fair per-repo scoring, not global top-k chunks
                           Handled by: aggregated repo-level scoring
                           Example: "which repo has the best auth?"
                                    "which project handles errors most thoroughly?"
                                    "which of my repos is most production-ready?"

  repo_specific          → question targets one specific repo, either by name
                           or because there is an active session for that repo
                           Handled by: vector search within that repo only,
                                       with session context and deduplication
                           Example: "how does auth work in claimsense?"
                                    "walk me through the parser" (active session)

FALLBACK:
  If classification fails after retries, defaults to cross_repo_semantic.
  This is the safest fallback — broad search is never silently wrong,
  it might just be slightly slower than a metadata lookup would have been.
"""

import os
import json
import time
from typing import Optional
from google import genai


# ---------------------------------------------------------------------------
# Router prompt
# ---------------------------------------------------------------------------

ROUTER_PROMPT = """You are a routing assistant for a GitHub repository search tool.

Given a user question, classify it into exactly one of five types and return ONLY valid JSON.
No explanation, no markdown, no backticks.

Question: "{question}"
Active repo (if user is currently exploring one): {active_repo}

TYPES:

list_repos
  User wants to list, count, or browse their repos.
  Examples: "what repos do I have?", "show me my projects", "how many repos?"

cross_repo_metadata
  Question asks about a technology, language, deployment status, or project
  category answerable by checking stored metadata — WITHOUT reading source code.
  Metadata contains: repo name, description, technologies list, purpose,
  primary language, deployment URL, topics.
  Examples:
    "do I have anything using Redis?"
    "which repos are deployed?"
    "do I have any Java projects?"
    "which projects use Docker?"
    "what languages do I use across my repos?"

cross_repo_semantic
  Question asks whether a concept, pattern, or feature EXISTS anywhere across
  repos — requires reading actual source code, but does NOT rank or compare repos.
  The answer is find/not-find, not a ranking.
  Examples:
    "do any of my repos implement rate limiting?"
    "do I have anything using WebSockets?"
    "which repos handle file uploads?"

cross_repo_comparative
  Question RANKS or COMPARES repos against each other on some quality or
  implementation. The answer is a ranking or winner, not just find/not-find.
  These CANNOT be answered by metadata alone and require reading code from
  multiple repos fairly.
  Examples:
    "which repo has the best authentication?"
    "which project handles errors most thoroughly?"
    "which of my repos is most production-ready?"
    "which project has the cleanest code structure?"
    "compare how claimsense and skillswap handle database access"

repo_specific
  Question targets one specific repo by name, OR there is an active repo
  and the question is a follow-up that doesn't name a different repo.
  Examples:
    "how does auth work in claimsense?"
    "walk me through the parser" (with active repo set)
    "what does Database.java do?" (with active repo set)

RULES:
  - If repo_specific: extract the repo name, or use active_repo if set
  - If repo_specific but no repo name and no active_repo: use cross_repo_semantic
  - cross_repo_comparative vs cross_repo_semantic:
      comparative = user wants a ranking/winner/comparison
      semantic    = user wants to find IF something exists
  - cross_repo_metadata vs cross_repo_semantic:
      metadata = answerable from tech stack / description / deployment info
      semantic = requires reading actual implementation code

Return exactly one of:
  {{"type": "list_repos"}}
  {{"type": "cross_repo_metadata"}}
  {{"type": "cross_repo_semantic"}}
  {{"type": "cross_repo_comparative"}}
  {{"type": "repo_specific", "repo": "<name>"}}"""


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required but was not provided.")
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def classify_question(
    question: str,
    active_repo: Optional[str] = None,
    max_retries: int = 3,
) -> dict:
    """
    Classify a user question into one of five query types.

    Parameters:
        question     Raw user question string.
        active_repo  Currently active repo name from session, or None.

    Returns one of:
        {"type": "list_repos"}
        {"type": "cross_repo_metadata"}
        {"type": "cross_repo_semantic"}
        {"type": "cross_repo_comparative"}
        {"type": "repo_specific", "repo": "claimsense"}

    Falls back to {"type": "cross_repo_semantic"} on failure.
    """
    client = _get_client()
    prompt = ROUTER_PROMPT.format(
        question=question,
        active_repo=active_repo if active_repo else "none",
    )

    valid_types = {"list_repos", "cross_repo_metadata",
                   "cross_repo_semantic", "cross_repo_comparative",
                   "repo_specific"}

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash", contents=prompt
            )
            raw = response.text.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)

            if parsed.get("type") not in valid_types:
                raise ValueError(f"Unexpected type: {parsed.get('type')}")

            if parsed["type"] == "repo_specific" and not parsed.get("repo"):
                print("  [router] repo_specific but no repo name. "
                      "Falling back to cross_repo_semantic.")
                return {"type": "cross_repo_semantic"}

            return parsed

        except json.JSONDecodeError as e:
            print(f"  [router] Invalid JSON (attempt {attempt + 1}): {e}")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"  [router] Rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [router] Error (attempt {attempt + 1}): {e}")

    print("  [router] Classification failed. Defaulting to cross_repo_semantic.")
    return {"type": "cross_repo_semantic"}
