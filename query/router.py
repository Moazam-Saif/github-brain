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

                           When specific repos are named in the question, the
                           route includes a "repos" list so the retriever fetches
                           chunks only from those repos, not the global ranking.
                           Example: "compare CorpLaw-AI and skillswap on auth"
                           → {"type": "cross_repo_comparative", "repos": ["CorpLaw-AI", "skillswap"]}

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

import json
import time
from typing import Optional
from gemini_client import get_client, GEMINI_MODEL


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
    "compare CorpLaw-AI and Claim-Verification-Automation on database design"

  If the question explicitly names specific repos to compare, extract those
  repo names into a "repos" list. If no specific repos are named (general
  comparison across all repos), omit the "repos" field entirely.

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
  - cross_repo_comparative with named repos: include "repos" list with the
    exact repo names as mentioned in the question (preserve casing as given)

Return exactly one of:
  {{"type": "list_repos"}}
  {{"type": "cross_repo_metadata"}}
  {{"type": "cross_repo_semantic"}}
  {{"type": "cross_repo_comparative"}}
  {{"type": "cross_repo_comparative", "repos": ["RepoA", "RepoB"]}}
  {{"type": "repo_specific", "repo": "<name>"}}"""


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
        {"type": "cross_repo_comparative", "repos": ["CorpLaw-AI", "skillswap"]}
        {"type": "repo_specific", "repo": "claimsense"}

    For cross_repo_comparative, "repos" is present only when the question
    explicitly names specific repos to compare. When present, the retriever
    fetches chunks only from those repos rather than running a global ranking.

    Falls back to {"type": "cross_repo_semantic"} on failure.
    """
    client = get_client()
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
                model=GEMINI_MODEL, contents=prompt
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

            # Normalize the repos list for cross_repo_comparative if present.
            if parsed["type"] == "cross_repo_comparative":
                repos = parsed.get("repos")
                if repos is not None:
                    if not isinstance(repos, list) or len(repos) == 0:
                        # Malformed — treat as no filter
                        parsed.pop("repos", None)
                    else:
                        # Keep only string entries, drop anything else
                        parsed["repos"] = [r for r in repos if isinstance(r, str)]
                        if not parsed["repos"]:
                            parsed.pop("repos", None)

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
