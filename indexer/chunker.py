"""
chunker.py
----------
Splits file content into overlapping chunks for embedding and storage.

Responsibilities:
  - Prepend a context header to each file's content before chunking
  - Split file text into chunks of ~512 tokens
  - Overlap consecutive chunks by ~50 tokens to avoid context loss at boundaries
  - Attach full repo metadata + file-level fields to each chunk
  - Skip files that are empty or whitespace-only after content is fetched

Why 512 tokens / 50 token overlap:
  512 tokens is large enough to contain a complete function or logical block
  with surrounding context, but small enough that retrieved chunks stay focused
  on one concept. 50-token overlap prevents meaning loss at chunk boundaries.
  See blueprint Section 2 for full rationale.

Token estimation:
  We use a simple character-based approximation (1 token ≈ 4 chars) rather than
  loading a full tokenizer. This keeps dependencies minimal and is accurate
  enough for chunking purposes — the goal is approximately 512 tokens, not exactly.

CONTEXT HEADERS (new):
  Before chunking, each file's content is prefixed with a one-line header:

    File: lib/auth.ts | Role: authentication | Purpose: Google OAuth setup and NextAuth adapter configuration

  This makes every chunk's embedding semantically represent what the file DOES,
  not just its raw code. A question like "how does authentication work?" now
  scores highly against lib/auth.ts chunks directly — not just against files
  that happen to talk *about* auth in passing.

  Role and purpose come from one of two sources, in priority order:
    1. README-extracted (file_roles / file_purposes from metadata_generator.py) —
       only available for files in the README's curated "important files" list.
    2. Filename/folder inference (_infer_role_from_path) — a lightweight fallback
       based on the file's own name and immediate parent folder. Used for all
       other files, including repos with no curated README list at all.

  If no role can be determined, the header omits the Role/Purpose fields and
  is just "File: <path>". A header is always present so every chunk at least
  carries its file path in the embedded text.
"""

from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE_TOKENS    = 512
OVERLAP_TOKENS       = 50
CHARS_PER_TOKEN      = 4   # approximation: 1 token ≈ 4 characters

CHUNK_SIZE_CHARS     = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN    # 2048 chars
OVERLAP_CHARS        = OVERLAP_TOKENS   * CHARS_PER_TOKEN     # 200 chars


# ---------------------------------------------------------------------------
# Filename/folder-based role inference (fallback when README has no role)
# ---------------------------------------------------------------------------
#
# Checked in this order for every file NOT covered by README-extracted file_roles:
#   1. Filename match (FILENAME_ROLE_MAP)   — most specific
#   2. Folder match   (FOLDER_ROLE_MAP)     — less specific, broader net
#   3. No match → role is None, header omits Role/Purpose
#
# Roles here use the SAME controlled vocabulary as metadata_generator.py's
# VALID_FILE_ROLES, so headers look identical regardless of source.

FILENAME_ROLE_MAP = [
    (["auth", "login", "logout", "oauth", "jwt", "credential", "password", "signup", "register"], "authentication"),
    (["db", "database", "prisma", "mongoose", "sequelize", "knex", "orm"], "database"),
    (["schema", "model", "entity", "migration"], "database"),
    (["route", "router", "routes"], "routing"),
    (["controller", "handler", "resolver"], "api"),
    (["middleware", "guard", "interceptor"], "middleware"),
    (["store", "redux", "slice", "reducer", "action", "context"], "state management"),
    (["config", "settings", "env", "constants", "bootstrap", "init", "setup"], "configuration"),
    (["test", "spec", "mock", "fixture", "__test__"], "testing"),
    (["queue", "worker", "job", "cron", "scheduler", "task"], "queue/background jobs"),
    (["upload", "storage", "media", "cloudinary", "s3", "bucket"], "file storage"),
    (["embed", "vector", "retriev", "search", "index"], "search/retrieval"),
    (["socket", "websocket", "realtime", "stream"], "realtime"),
    (["email", "mail", "notification", "sms"], "notifications"),
    (["cache", "redis", "memcache"], "caching"),
    (["payment", "stripe", "billing", "invoice"], "payments"),
    (["logger", "logging", "log"], "logging"),
    (["util", "helper", "utils", "helpers", "common", "shared"], "utilities"),
    (["component", "widget"], "ui component"),
    (["page", "view", "screen", "layout"], "ui page"),
    (["style", "theme", "css", "tailwind"], "styling"),
    (["prompt", "llm", "ai", "gemini", "openai", "claude"], "ai/llm"),
    (["api", "client", "service", "sdk"], "api client"),
]

FOLDER_ROLE_MAP = [
    (["auth", "authentication"], "authentication"),
    (["models", "model", "entities", "entity", "schemas"], "database"),
    (["routes", "router", "routers"], "routing"),
    (["controllers", "controller", "handlers"], "api"),
    (["middleware", "middlewares", "guards"], "middleware"),
    (["store", "redux", "slices", "reducers", "context"], "state management"),
    (["config", "configs", "configuration", "settings"], "configuration"),
    (["tests", "test", "__tests__", "spec", "specs", "mocks"], "testing"),
    (["workers", "jobs", "queues", "tasks", "cron"], "queue/background jobs"),
    (["uploads", "storage", "media", "assets"], "file storage"),
    (["components", "component", "widgets"], "ui component"),
    (["pages", "views", "screens", "layouts"], "ui page"),
    (["styles", "themes", "css"], "styling"),
    (["utils", "helpers", "lib", "shared", "common"], "utilities"),
    (["services", "service"], "api client"),
    (["api", "apis"], "api"),
]


def _infer_role_from_path(file_path: str) -> Optional[str]:
    """
    Infer a controlled-vocabulary role for a file based on its filename
    and immediate parent folder, when no README-extracted role is available.

    Matching order:
      1. Filename (without extension) checked against FILENAME_ROLE_MAP —
         e.g. "authController.js" → filename "authcontroller" contains "auth"
         → "authentication"
      2. Parent folder name checked against FOLDER_ROLE_MAP —
         e.g. "server/models/User.js" → folder "models" → "database"
      3. No match → None

    Both maps are checked using case-insensitive substring matching against
    the filename/foldername, so "AuthController.java" and "auth_controller.py"
    both match "auth".

    Returns the role string, or None if no rule matches.
    """
    path = file_path.replace("\\", "/")
    parts = path.split("/")
    filename = parts[-1].lower()

    # Strip extension for filename matching.
    filename_no_ext = filename.rsplit(".", 1)[0] if "." in filename else filename

    # 1. Filename match — most specific.
    for keywords, role in FILENAME_ROLE_MAP:
        if any(kw in filename_no_ext for kw in keywords):
            return role

    # 2. Parent folder match — broader net.
    if len(parts) >= 2:
        parent_folder = parts[-2].lower()
        for keywords, role in FOLDER_ROLE_MAP:
            if any(kw in parent_folder for kw in keywords):
                return role

    # 3. No match.
    return None


# ---------------------------------------------------------------------------
# Context header construction
# ---------------------------------------------------------------------------

def _build_chunk_header(
    file_path: str,
    file_roles: dict,
    file_purposes: dict,
) -> str:
    """
    Build the one-line context header prepended to a file's content
    before chunking.

    Role/purpose resolution order:
      1. README-extracted role/purpose (file_roles / file_purposes, keyed
         by exact file_path) — most accurate, from the repo's own docs.
      2. Filename/folder inference (_infer_role_from_path) — used when the
         file has no README-extracted role.
      3. No role available — header is just "File: <path>".

    Purpose is only ever included if it came from the README (filename/folder
    inference produces a role but not a purpose — there's nothing reliable to
    say beyond the role itself).

    Returns one of:
      "File: lib/auth.ts | Role: authentication | Purpose: Google OAuth setup and NextAuth adapter configuration"
      "File: server/controllers/authController.js | Role: authentication"
      "File: server/app.js"
    """
    role    = file_roles.get(file_path)
    purpose = file_purposes.get(file_path)

    if role is None:
        role = _infer_role_from_path(file_path)
        purpose = None  # inferred roles never carry a purpose

    header = f"File: {file_path}"
    if role:
        header += f" | Role: {role}"
    if purpose:
        header += f" | Purpose: {purpose}"

    return header


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str) -> list[str]:
    """
    Split a text string into overlapping character-window chunks.

    Uses a sliding window of CHUNK_SIZE_CHARS with a step of
    (CHUNK_SIZE_CHARS - OVERLAP_CHARS) so consecutive chunks share
    OVERLAP_CHARS characters of content.

    Empty or whitespace-only input returns an empty list.
    """
    text = text.strip()
    if not text:
        return []

    step = CHUNK_SIZE_CHARS - OVERLAP_CHARS
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE_CHARS
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def chunk_file(
    file: dict,
    repo_metadata: dict,
    file_roles: Optional[dict] = None,
    file_purposes: Optional[dict] = None,
) -> list[dict]:
    """
    Chunk a single file and attach metadata to each chunk.

    Parameters:
        file            Dict with keys: path, content, ext
                        (as returned by GitHubClient.get_indexable_files)
        repo_metadata   The repo-level metadata dict from metadata_generator.py
                        (repo_name, repo_description, repo_technologies, etc.)
        file_roles      Optional dict {file_path: role} from metadata_generator.py.
                        Empty/None if the repo's README had no curated file list.
        file_purposes   Optional dict {file_path: purpose} from metadata_generator.py.
                        Empty/None if the repo's README had no curated file list.

    Before chunking, a one-line context header is prepended to the file's
    content (see _build_chunk_header). This header becomes part of chunk 0's
    text — subsequent chunks are produced by the normal sliding window over
    the header+content string, so the header only meaningfully affects the
    first chunk's embedding.

    Returns a list of chunk dicts. Each chunk dict contains everything needed
    for embedding and Deep Lake storage:

    {
        "text":        <the chunk text, chunk 0 prefixed with the context header>,
        "repo_name":   "claimsense",
        "repo_description": "...",
        "repo_technologies": [...],
        "repo_purpose": "web app",
        "repo_language": "python",
        "deployment_url": "https://...",
        "repo_topics": [...],
        "metadata_source": "readme",
        "metadata_confidence": "high",
        "file_path":   "src/parser.py",
        "file_type":   ".py",
        "file_role":   "authentication" | None,
        "chunk_index": 0,
        "indexed_at":  "2026-06-10T12:00:00+00:00"
    }

    Returns an empty list if the file content produces no chunks
    (empty file, whitespace only).
    """
    content  = file.get("content", "")
    path     = file.get("path", "")
    ext      = file.get("ext", "")

    file_roles    = file_roles or {}
    file_purposes = file_purposes or {}

    # Build the context header and prepend it to the file content.
    header        = _build_chunk_header(path, file_roles, file_purposes)
    resolved_role = file_roles.get(path) or _infer_role_from_path(path)

    content_with_header = f"{header}\n\n{content}"

    raw_chunks = _split_into_chunks(content_with_header)
    if not raw_chunks:
        return []

    timestamp = datetime.now(timezone.utc).isoformat()

    result = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk = {
            # The actual content that gets embedded and stored.
            # Chunk 0 includes the context header; later chunks are raw content
            # as produced by the sliding window over content_with_header.
            "text": chunk_text,

            # Repo-level metadata — copied from repo_metadata.
            # Every field is present on every chunk so retrieval
            # never needs a second lookup.
            "repo_name":           repo_metadata.get("repo_name"),
            "repo_description":    repo_metadata.get("repo_description"),
            "repo_technologies":   repo_metadata.get("repo_technologies", []),
            "repo_purpose":        repo_metadata.get("repo_purpose"),
            "repo_language":       repo_metadata.get("repo_language"),
            "deployment_url":      repo_metadata.get("deployment_url"),
            "repo_topics":         repo_metadata.get("repo_topics", []),
            "metadata_source":     repo_metadata.get("metadata_source"),
            "metadata_confidence": repo_metadata.get("metadata_confidence"),

            # File-level fields.
            "file_path":    path,
            "file_type":    ext,
            "file_role":    resolved_role,

            # Chunk position within this file.
            "chunk_index":  idx,

            # Timestamp of when this chunk was indexed.
            "indexed_at":   timestamp,
        }
        result.append(chunk)

    return result


def chunk_repo_files(
    files: list[dict],
    repo_metadata: dict,
    file_roles: Optional[dict] = None,
    file_purposes: Optional[dict] = None,
) -> list[dict]:
    """
    Chunk all files for a repo and return a flat list of all chunks.

    Parameters:
        files           List of file dicts from GitHubClient.get_indexable_files()
        repo_metadata   Repo-level metadata from metadata_generator.py
        file_roles      Optional dict {file_path: role} from metadata_generator.py
        file_purposes   Optional dict {file_path: purpose} from metadata_generator.py

    Returns a flat list of all chunk dicts across all files.
    Files that produce no chunks (empty, binary, whitespace) are silently skipped.
    """
    all_chunks = []

    for file in files:
        chunks = chunk_file(file, repo_metadata, file_roles, file_purposes)
        if chunks:
            all_chunks.extend(chunks)

    repo_name  = repo_metadata.get("repo_name", "unknown")
    file_count = len(files)
    chunk_count = len(all_chunks)
    print(f"  [chunker] {repo_name}: {file_count} files → {chunk_count} chunks")

    return all_chunks
