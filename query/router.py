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
Active comparison (if the user just compared specific repos, these are the repos — if this question is a follow-up that doesn't name different repos, assume it's still about these): {active_comparison}

TYPES:

list_repos
  User wants to list, count, or browse their repos.
  Examples: "what repos do I have?", "show me my projects", "how many repos?"

cross_repo_metadata
  Question asks about ANY attribute that is already extracted and stored as
  metadata at index time — WITHOUT needing to read source code. This is a much
  broader set than just "technology/language/deployment". The stored metadata
  fields are:
    repo_name, repo_description, repo_technologies, repo_purpose,
    repo_language, deployment_url, repo_topics,
    has_authentication (bool), has_database (bool), database_type,
    has_api (bool), api_style, has_frontend (bool), frontend_framework,
    architecture_pattern, key_features (list), external_services (list),
    has_tests (bool)
  If the question can be answered by checking ANY of these fields directly —
  even if the question is phrased as "does X exist" or "do any repos do Y" —
  it is cross_repo_metadata, NOT cross_repo_semantic. This includes questions
  about authentication, databases, APIs, frontend frameworks, testing,
  external services (Redis, Stripe, etc.), and architecture patterns, because
  ALL of these are already booleans/fields in metadata — the system never
  needs to read code to answer them.
  Examples:
    "do I have anything using Redis?"            → external_services
    "which repos are deployed?"                  → deployment_url
    "do I have any Java projects?"                → repo_language
    "which projects use Docker?"                  → repo_technologies / external_services
    "what languages do I use across my repos?"    → repo_language
    "do any of my repos use authentication?"      → has_authentication
    "which repos have a database?"                → has_database
    "do any repos have tests?"                     → has_tests
    "which projects have a REST API?"              → has_api / api_style
    "do any of my repos use Redis for caching?"    → external_services
      (the "for caching" phrase does NOT change this — it's still asking
      "is Redis used anywhere", which is a metadata lookup, not a request
      to read caching implementation code)

cross_repo_semantic
  Question asks about HOW a concept/pattern is implemented, or about a
  specific code-level detail that is NOT one of the stored metadata fields
  above. Requires reading actual source code. The answer is find/not-find
  or "how", not a ranking, and NOT something coverable by has_authentication/
  has_database/has_api/has_frontend/has_tests/external_services/etc.
  Examples:
    "do any of my repos implement rate limiting?"   (no has_rate_limiting field)
    "do I have anything using WebSockets?"           (no has_websockets field)
    "which repos handle file uploads?"                (no has_file_upload field)
    "which repos validate JWT signatures manually?"   (implementation detail,
                                                        not just "has auth")
  RULE OF THUMB: if a yes/no metadata field already exists for the concept
  (see the list above), use cross_repo_metadata even if it's phrased as
  "do any repos do X" or "which repos use X for Y". Only use cross_repo_semantic
  when the concept has NO corresponding stored field and genuinely requires
  reading code to answer.

cross_repo_comparative
  Question RANKS or COMPARES repos against each other, AND the ranking
  criterion requires reading actual implementation code to evaluate — it
  cannot be answered from metadata fields alone.

  DECISION GATE — apply this test before choosing this type:
    Ask: can the ranking/comparison criterion be evaluated using only the
    stored metadata fields (repo_purpose, repo_technologies, key_features,
    repo_description, repo_language, architecture_pattern, etc.)?
    → YES, metadata is enough: use cross_repo_metadata instead.
    → NO, you must read implementation code to judge: use cross_repo_comparative.

  Code-quality criteria that require reading code (→ cross_repo_comparative):
    "which repo has the best authentication?"      (need to read auth code)
    "which project handles errors most thoroughly?" (need to read error handling)
    "which of my repos is most production-ready?"  (need to read code quality)
    "which project has the cleanest code structure?"
    "compare how claimsense and skillswap handle database access"
    "compare CorpLaw-AI and Claim-Verification-Automation on database design"

  Identity/relevance criteria answerable from metadata (→ cross_repo_metadata):
    "which repo is most relevant for an AI engineering role?"
    "suggest my best project for a portfolio"
    "which of my repos is most impressive for a backend interview?"
    "which project best demonstrates my Python skills?"
    These ask WHAT the repo IS (its purpose, stack, domain) — answerable
    from repo_purpose, repo_technologies, key_features, repo_description.
    Route to cross_repo_metadata even though they sound like rankings.

  If the question explicitly names specific repos to compare, extract those
  repo names into a "repos" list. If no specific repos are named (general
  comparison across all repos), omit the "repos" field entirely.

  FOLLOW-UP ON AN ACTIVE COMPARISON: if active_comparison is set (not "none")
  and this question does NOT name different repos — e.g. "what about the
  UI?", "and error handling?", "which one is better at X?" — classify as
  cross_repo_comparative and reuse the SAME repos from active_comparison in
  "repos". Only use different repos if the question explicitly names them
  instead.
  Example: active_comparison = "CorpLaw-AI, Claim-Verification-Automation"
           question = "and what about user interface"
  → {{"type": "cross_repo_comparative", "repos": ["CorpLaw-AI", "Claim-Verification-Automation"]}}

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
  - cross_repo_comparative vs cross_repo_metadata — USE THIS GATE FIRST:
      Before routing to cross_repo_comparative, ask: does the comparison
      criterion require reading implementation code, or is it answerable
      from what the repo IS (purpose, stack, domain, description)?
      → Answerable from metadata: cross_repo_metadata. ALWAYS. Even if
        the question uses ranking words like "best", "most", "suggest".
      → Requires reading code to judge quality/implementation: cross_repo_comparative.
  - cross_repo_comparative vs cross_repo_semantic:
      comparative = user wants a ranking/winner/comparison
      semantic    = user wants to find IF something exists
  - cross_repo_metadata vs cross_repo_semantic — USE THIS EXACT TEST:
      Does a stored metadata field directly answer this? Check this list:
      has_authentication, has_database, database_type, has_api, api_style,
      has_frontend, frontend_framework, architecture_pattern, key_features,
      external_services, has_tests, repo_technologies, repo_language,
      deployment_url, repo_topics, repo_purpose.
      → YES, a field covers it: cross_repo_metadata. ALWAYS, even if phrased
        as "do any repos do X", "does X exist", or "which repos use X for Y".
      → NO field covers it (the concept needs reading actual code to verify,
        e.g. rate limiting, WebSockets, file uploads, specific algorithms):
        cross_repo_semantic.
      Do NOT default to semantic just because the question uses "do any/does
      X exist" phrasing — that phrasing is used by BOTH types. The deciding
      factor is ALWAYS whether a metadata field exists for the concept, not
      the question's grammar.
  - cross_repo_comparative with named repos: include "repos" list with the
    exact repo names as mentioned in the question (preserve casing as given)
  - If active_comparison is set (not "none") and the question is a follow-up
    that does NOT name different repos: classify as cross_repo_comparative
    and set "repos" to the SAME repos listed in active_comparison
  - If active_comparison is set but the question explicitly names NEW/
    different repos to compare instead: use those new repos in "repos",
    ignore active_comparison
  - If active_comparison is set but the question targets ONE specific repo
    from that comparison by name (e.g. "tell me more about CorpLaw-AI"):
    classify as repo_specific with that repo name, NOT comparative

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
    active_comparison: Optional[list[str]] = None,
    max_retries: int = 3,
) -> dict:
    """
    Classify a user question into one of five query types.

    Parameters:
        question           Raw user question string.
        active_repo         Currently active repo name from session, or None.
        active_comparison   List of repo names from the most recent
                            cross_repo_comparative session (session["comparison_repos"]
                            in engine.py), or None. Lets the router classify
                            follow-ups like "what about the UI?" as a
                            continuation of the SAME comparison rather than
                            a fresh global search across all repos.

    Returns one of:
        {"type": "list_repos"}
        {"type": "cross_repo_metadata"}
        {"type": "cross_repo_semantic"}
        {"type": "cross_repo_comparative"}
        {"type": "cross_repo_comparative", "repos": ["CorpLaw-AI", "skillswap"]}
        {"type": "repo_specific", "repo": "claimsense"}

    For cross_repo_comparative, "repos" is present when the question
    explicitly names specific repos OR when it's a follow-up on an active
    comparison (in which case "repos" carries over from active_comparison).

    Falls back to {"type": "cross_repo_semantic"} on failure.
    """
    client = get_client()
    prompt = ROUTER_PROMPT.format(
        question=question,
        active_repo=active_repo if active_repo else "none",
        active_comparison=", ".join(active_comparison) if active_comparison else "none",
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
