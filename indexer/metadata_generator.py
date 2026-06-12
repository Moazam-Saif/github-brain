"""
metadata_generator.py
---------------------
Generates structured metadata for each repo during indexing.

Responsibilities:
  - Implement the tiered fallback to pick the best metadata source
  - Call Gemini once per repo to extract structured fields AND file filter config
  - Merge Gemini output with hard GitHub API fields
  - Handle the case where no metadata source is available (confidence: low)

This is NOT RAG. This is plain structured extraction — a one-time LLM call
at index time. No vector search, no embeddings, no similarity scores.

Field ownership (from blueprint Section 5):
  GitHub API  → repo_name, repo_language, repo_topics  (never overwritten)
  Gemini      → repo_description, repo_technologies, repo_purpose, deployment_url
  System      → metadata_source, metadata_confidence, file_filter

The file_filter field is returned separately from the metadata dict.
It drives get_indexable_files() in github_client.py. It is NOT stored
as chunk metadata — it is only used during indexing to decide which files
to fetch. Once indexing is done it is discarded.
"""

import os
import json
import time
from typing import Optional
from gemini_client import get_client, GEMINI_MODEL


# ---------------------------------------------------------------------------
# Language-based fallback extension sets
# Used when README gives no file listing information.
# ---------------------------------------------------------------------------

LANGUAGE_FALLBACK_EXTENSIONS = {
    "Python":     [".py", ".pyi"],
    "JavaScript": [".js", ".mjs", ".cjs", ".jsx"],
    "TypeScript": [".ts", ".tsx"],
    "Java":       [".java"],
    "C++":        [".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"],
    "C":          [".c", ".h"],
    "C#":         [".cs"],
    "Go":         [".go"],
    "Rust":       [".rs"],
    "Ruby":       [".rb"],
    "PHP":        [".php"],
    "Swift":      [".swift"],
    "Kotlin":     [".kt", ".kts"],
    "Scala":      [".scala"],
    "Shell":      [".sh", ".bash", ".zsh"],
    "CMake":      [".cmake", "CMakeLists.txt"],
    "HTML":       [".html", ".htm"],
    "CSS":        [".css", ".scss", ".sass"],
    "default":    [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp",
                   ".h", ".c", ".go", ".rs", ".rb", ".cs", ".md"],
}

# Always include README regardless of language — it provides context.
ALWAYS_INCLUDE_EXTENSIONS = {".md"}


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are analyzing a GitHub repository README or dependency file.
Extract the following and return ONLY a valid JSON object.
No explanation, no markdown, no backticks.

Source type: {source_type}
Source content:
{source_content}

Return this exact structure:
{{
  "repo_description": "one clear sentence describing what this project does",
  "repo_technologies": ["array", "of", "technologies", "frameworks", "databases"],
  "repo_purpose": "one of: web app / cli tool / library / ml project / api / desktop app / other",
  "deployment_url": "full URL if deployed anywhere, null if not mentioned",

  "has_authentication":   true or false,
  "has_database":         true or false,
  "database_type":        "mysql / postgresql / mongodb / sqlite / supabase / firebase / other / null",
  "has_api":              true or false,
  "api_style":            "REST / GraphQL / gRPC / other / null",
  "has_frontend":         true or false,
  "frontend_framework":   "react / vue / angular / javafx / swing / other / null",
  "architecture_pattern": "MVC / microservices / monolith / agent-based / event-driven / other / null",
  "key_features":         ["short list of 3-6 main features explicitly described in the README"],
  "external_services":    ["third-party APIs, services, or platforms used e.g. stripe, twilio, cloudinary"],
  "has_tests":            true or false,

  "files_to_index": {{
    "found": true or false,
    "paths": ["list of file paths and folder prefixes explicitly mentioned as important/worth looking at"],
    "extensions": ["list of file extensions to include, e.g. .java .cpp .py"]
  }}
}}

Rules for repo_description, repo_technologies, repo_purpose, deployment_url:
- repo_description: one sentence, factual, no hype
- repo_technologies: languages, frameworks, databases, external services
- deployment_url: real URL explicitly found in content, never invented. null if not mentioned.

Rules for the new boolean and categorical fields:
- has_authentication: true if the project has any login, signup, JWT, OAuth, session, or
  credential-checking logic mentioned
- has_database: true if any database, ORM, or data persistence is mentioned
- database_type: the specific database used, null if none
- has_api: true if the project exposes or consumes HTTP endpoints or an API
- api_style: REST if it uses routes/endpoints, null if not applicable
- has_frontend: true if there is any UI — web, desktop, mobile, or otherwise
- frontend_framework: the specific UI framework, null if none or if CLI only
- architecture_pattern: the dominant structural pattern if identifiable, null if unclear
- key_features: extract from the Features section or equivalent. Keep each item short.
- external_services: third-party integrations beyond the main stack (e.g. Cloudinary,
  Lightcast, Zoho, Railway, Jitsi). Do NOT repeat items already in repo_technologies.
- has_tests: true if any test files, test commands, or testing frameworks are mentioned

Rules for files_to_index:
- Set "found" to true ONLY if the README contains an explicit list of files/folders
  described as important or worth reading. Look for sections like "Project Structure",
  "Only the files that matter", code blocks listing source files with descriptions.
- If "found" is true, "paths" must contain every explicitly listed important path.
  Do NOT include paths listed under "not worth looking at" or similar exclusion language.
- "extensions" must list every file extension present among the listed paths.
- If "found" is false, set "paths" to [] and "extensions" to [].
- Only include paths EXPLICITLY mentioned. Never infer or guess."""


SOURCE_LABELS = {
    "readme":       "README file (Markdown)",
    "package":      "package.json (Node.js dependency manifest)",
    "requirements": "requirements.txt (Python dependency list)",
    "pyproject":    "pyproject.toml (Python project configuration)",
}


# ---------------------------------------------------------------------------
# Gemini extraction call
# ---------------------------------------------------------------------------

def _extract_with_gemini(
    source_type: str,
    source_content: str,
    model,
    max_retries: int = 3,
) -> dict:
    """
    Send the extraction prompt to Gemini and parse the JSON response.

    Returns a dict with keys:
      repo_description, repo_technologies, repo_purpose, deployment_url,
      files_to_index { found, paths, extensions }

    Falls back to safe defaults if all retries fail.
    """
    prompt = EXTRACTION_PROMPT.format(
        source_type=SOURCE_LABELS.get(source_type, source_type),
        source_content=source_content[:10000],
    )

    for attempt in range(max_retries):
        try:
            response = model.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
            raw = response.text.strip()

            # Strip accidental markdown fences.
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)

            # Validate top-level keys — only the core ones are required.
            # New fields are optional with safe defaults applied below.
            required = {
                "repo_description", "repo_technologies",
                "repo_purpose", "deployment_url", "files_to_index"
            }
            if not required.issubset(parsed.keys()):
                raise ValueError(f"Missing keys: {required - parsed.keys()}")

            # Normalize technologies.
            if not isinstance(parsed["repo_technologies"], list):
                parsed["repo_technologies"] = []

            # Normalize deployment_url.
            url = parsed.get("deployment_url")
            if url is not None and not isinstance(url, str):
                parsed["deployment_url"] = None

            # Normalize boolean fields — default False if missing or wrong type.
            for bool_field in ("has_authentication", "has_database",
                               "has_api", "has_frontend", "has_tests"):
                parsed[bool_field] = bool(parsed.get(bool_field, False))

            # Normalize string/null categorical fields.
            for str_field in ("database_type", "api_style",
                              "frontend_framework", "architecture_pattern"):
                val = parsed.get(str_field)
                parsed[str_field] = val if isinstance(val, str) else None

            # Normalize list fields.
            for list_field in ("key_features", "external_services"):
                val = parsed.get(list_field, [])
                parsed[list_field] = val if isinstance(val, list) else []

            # Normalize files_to_index.
            fti = parsed.get("files_to_index", {})
            if not isinstance(fti, dict):
                fti = {}
            parsed["files_to_index"] = {
                "found":      bool(fti.get("found", False)),
                "paths":      fti.get("paths", []) if isinstance(fti.get("paths"), list) else [],
                "extensions": fti.get("extensions", []) if isinstance(fti.get("extensions"), list) else [],
            }

            return parsed

        except json.JSONDecodeError as e:
            print(f"  [metadata] Invalid JSON from Gemini (attempt {attempt + 1}): {e}")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"  [metadata] Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [metadata] Extraction error (attempt {attempt + 1}): {e}")

    print("  [metadata] All Gemini attempts failed. Using empty defaults.")
    return {
        "repo_description":   None,
        "repo_technologies":  [],
        "repo_purpose":       "other",
        "deployment_url":     None,
        "has_authentication": False,
        "has_database":       False,
        "database_type":      None,
        "has_api":            False,
        "api_style":          None,
        "has_frontend":       False,
        "frontend_framework": None,
        "architecture_pattern": None,
        "key_features":       [],
        "external_services":  [],
        "has_tests":          False,
        "files_to_index":     {"found": False, "paths": [], "extensions": []},
    }


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_repo_metadata(
    github_repo: dict,
    source_type: Optional[str],
    source_content: Optional[str],
) -> tuple[dict, dict]:
    """
    Build the complete metadata object and file filter config for a repo.

    Parameters:
        github_repo     Repo dict from GitHubClient.get_repos().
        source_type     One of: "readme", "package", "requirements", "pyproject", None.
        source_content  Raw text of the metadata source file, or None.

    Returns:
        (metadata, file_filter)

        metadata    — dict attached to every chunk. Contains repo-level fields.
                      Does NOT contain file_filter (not needed at query time).

        file_filter — dict used only during indexing by get_indexable_files():
          {
            "mode":       "readme" | "language_fallback",
            "paths":      ["app.py", "src/", ...],   # explicit paths from README
            "extensions": [".py", ".java", ...],      # extensions from README or fallback
          }

    Tiered fallback:
        README with file listing → mode: "readme", use Gemini-extracted paths + extensions
        README without listing   → mode: "language_fallback", extensions from language map
        No README (deps file)    → mode: "language_fallback", extensions from language map
        Nothing                  → mode: "language_fallback", extensions from default set
    """
    repo_name     = github_repo["name"]
    repo_language = github_repo.get("language")
    repo_topics   = github_repo.get("topics", [])

    # --- Call Gemini if we have a source file. ---
    if source_type is not None and source_content:
        confidence = "high" if source_type == "readme" else "medium"
        print(f"  [metadata] Extracting from {source_type} (confidence: {confidence})")

        model = get_client()
        time.sleep(4)  # Stay under 15 req/min free tier.

        gemini = _extract_with_gemini(source_type, source_content, model)
    else:
        confidence = "low"
        print(f"  [metadata] No metadata source. Using GitHub API fields only.")
        github_description = github_repo.get("description")
        gemini = {
            "repo_description":   github_description,
            "repo_technologies":  [],
            "repo_purpose":       "other",
            "deployment_url":     None,
            "has_authentication": False,
            "has_database":       False,
            "database_type":      None,
            "has_api":            False,
            "api_style":          None,
            "has_frontend":       False,
            "frontend_framework": None,
            "architecture_pattern": None,
            "key_features":       [],
            "external_services":  [],
            "has_tests":          False,
            "files_to_index":     {"found": False, "paths": [], "extensions": []},
        }

    # --- Build metadata dict (stored on every chunk). ---
    metadata = {
        # Always from GitHub API — never overwritten.
        "repo_name":           repo_name,
        "repo_language":       repo_language,
        "repo_topics":         repo_topics,

        # Core descriptive fields from Gemini.
        "repo_description":    gemini.get("repo_description"),
        "repo_technologies":   gemini.get("repo_technologies", []),
        "repo_purpose":        gemini.get("repo_purpose", "other"),
        "deployment_url":      gemini.get("deployment_url"),

        # Richer structural fields from Gemini — reduce need for vector search.
        "has_authentication":    gemini.get("has_authentication", False),
        "has_database":          gemini.get("has_database", False),
        "database_type":         gemini.get("database_type"),
        "has_api":               gemini.get("has_api", False),
        "api_style":             gemini.get("api_style"),
        "has_frontend":          gemini.get("has_frontend", False),
        "frontend_framework":    gemini.get("frontend_framework"),
        "architecture_pattern":  gemini.get("architecture_pattern"),
        "key_features":          gemini.get("key_features", []),
        "external_services":     gemini.get("external_services", []),
        "has_tests":             gemini.get("has_tests", False),

        # System fields.
        "metadata_source":     source_type if source_type else "github_api",
        "metadata_confidence": confidence,
    }

    # --- Build file_filter (used only during indexing, not stored). ---
    fti = gemini["files_to_index"]

    if source_type == "readme" and fti["found"] and fti["paths"]:
        # README has an explicit file listing — use it strictly.
        file_filter = {
            "mode":       "readme",
            "paths":      fti["paths"],
            "extensions": list(set(fti["extensions"]) | ALWAYS_INCLUDE_EXTENSIONS),
        }
        print(f"  [metadata] File filter: README-driven "
              f"({len(fti['paths'])} paths, {len(fti['extensions'])} extensions)")
    else:
        # No explicit listing — fall back to language-based extension set.
        lang_exts = LANGUAGE_FALLBACK_EXTENSIONS.get(
            repo_language,
            LANGUAGE_FALLBACK_EXTENSIONS["default"]
        )
        file_filter = {
            "mode":       "language_fallback",
            "paths":      [],
            "extensions": list(set(lang_exts) | ALWAYS_INCLUDE_EXTENSIONS),
        }
        print(f"  [metadata] File filter: language fallback "
              f"({repo_language or 'unknown'}, {len(file_filter['extensions'])} extensions)")

    # Log summary.
    techs    = ", ".join(metadata["repo_technologies"][:5]) or "none detected"
    deployed = metadata["deployment_url"] or "not deployed"
    print(f"  [metadata] {repo_name}: {metadata['repo_description'] or 'no description'}")
    print(f"  [metadata] technologies: {techs}")
    print(f"  [metadata] deployment:   {deployed}")

    return metadata, file_filter
