"""
chunker.py
----------
Splits file content into overlapping chunks for embedding and storage.

Responsibilities:
  - Prepend context headers to every chunk (FIX 5 — not just chunk 0)
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

CONTEXT HEADERS (FIX 5 — header on EVERY chunk, not just chunk 0):
  Before chunking, each file's content is prefixed with a full context header:

    File: lib/auth.ts | Role: authentication | Purpose: Google OAuth setup and NextAuth adapter configuration

  This was the original behaviour — but the header only appeared in chunk 0,
  because it was prepended to the file content before the sliding window ran.
  Chunk 1, 2, 3, ... started wherever the window landed in the raw code with
  NO role/file context in their embedded text.

  FIX 5: A SHORT header is now prepended to EVERY chunk:
    Chunk 0 (first):     full header — "File: lib/auth.ts | Role: authentication | Purpose: ..."
    Chunk 1+ (rest):     short header — "File: lib/auth.ts | Role: authentication"
                         (no Purpose — that's only needed once per file)

  This ensures BM25 and cosine BOTH get role/file-path signal for ANY chunk
  that the retriever surfaces — not just the first one. A question like
  "how does auth connect to the database?" might find its answer in chunk 3
  of lib/auth.ts (the Prisma adapter methods). Previously chunk 3 had no
  role header; now it has "File: lib/auth.ts | Role: authentication" in its
  embedded text, so it correctly surfaces for auth-related questions.

  The short header adds ~15-20 tokens per chunk — negligible versus 512 token
  chunk size, and well worth the improvement in retrieval accuracy.

Role/purpose resolution order (unchanged from original):
  1. README-extracted (file_roles / file_purposes from metadata_generator.py)
  2. Filename/folder inference (_infer_role_from_path)
  3. Neither available — short header is just "File: <path>"
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
      1. Filename (without extension) checked against FILENAME_ROLE_MAP
      2. Parent folder name checked against FOLDER_ROLE_MAP
      3. No match → None

    Both maps use case-insensitive substring matching.
    """
    path  = file_path.replace("\\", "/")
    parts = path.split("/")
    filename = parts[-1].lower()
    filename_no_ext = filename.rsplit(".", 1)[0] if "." in filename else filename

    for keywords, role in FILENAME_ROLE_MAP:
        if any(kw in filename_no_ext for kw in keywords):
            return role

    if len(parts) >= 2:
        parent_folder = parts[-2].lower()
        for keywords, role in FOLDER_ROLE_MAP:
            if any(kw in parent_folder for kw in keywords):
                return role

    return None


# ---------------------------------------------------------------------------
# Context header construction
# ---------------------------------------------------------------------------

def _build_full_header(
    file_path: str,
    file_roles: dict,
    file_purposes: dict,
) -> tuple[str, Optional[str]]:
    """
    Build the full context header for chunk 0 and resolve the role.

    Returns:
        (full_header_str, resolved_role)

    full_header is one of:
      "File: lib/auth.ts | Role: authentication | Purpose: Google OAuth setup..."
      "File: lib/auth.ts | Role: authentication"
      "File: lib/auth.ts"

    resolved_role is the role string used to build the short header for
    subsequent chunks (chunk 1+), or None if no role could be determined.
    """
    role    = file_roles.get(file_path)
    purpose = file_purposes.get(file_path)

    if role is None:
        role    = _infer_role_from_path(file_path)
        purpose = None   # inferred roles never carry a purpose string

    header = f"File: {file_path}"
    if role:
        header += f" | Role: {role}"
    if purpose:
        header += f" | Purpose: {purpose}"

    return header, role


def _build_short_header(file_path: str, role: Optional[str]) -> str:
    """
    Build the short context header prepended to chunk 1+ of a file.

    FIX 5: Every chunk now carries at least "File: <path>" in its embedded
    text, and "File: <path> | Role: <role>" when a role is known. This
    ensures retrieval accuracy for questions that find their answer in
    later chunks of a multi-chunk file — the role signal is present in
    the embedding regardless of which chunk is returned.

    Purpose is intentionally omitted (only in the full header on chunk 0)
    to keep the overhead minimal — the short header adds ~10-15 tokens.
    """
    header = f"File: {file_path}"
    if role:
        header += f" | Role: {role}"
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

    step   = CHUNK_SIZE_CHARS - OVERLAP_CHARS
    chunks = []
    start  = 0

    while start < len(text):
        end   = start + CHUNK_SIZE_CHARS
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

    FIX 5 — context header on every chunk:
      Previously, the context header ("File: ... | Role: ... | Purpose: ...")
      was prepended to the file content BEFORE the sliding window ran. This
      meant the header only appeared in chunk 0's text. Chunks 1, 2, 3, etc.
      started wherever the window landed in the raw code — no role context.

      Now the process is:
        1. Run the sliding window on the raw file content (no header yet)
           → produces raw_chunks[0], raw_chunks[1], ...
        2. Prepend the FULL header to raw_chunks[0] (includes Purpose if known)
        3. Prepend the SHORT header to raw_chunks[1+] (file path + role only)

      Every chunk now carries file path and role in its embedded text. The
      cross-encoder re-ranker (FIX 4) also benefits from this because it
      sees the file context in every chunk it evaluates.

    Parameters:
        file            Dict with keys: path, content, ext
        repo_metadata   The repo-level metadata dict from metadata_generator.py
        file_roles      Optional dict {file_path: role} from metadata_generator.py
        file_purposes   Optional dict {file_path: purpose} from metadata_generator.py

    Returns a list of chunk dicts. Each chunk dict contains everything needed
    for embedding and Deep Lake storage.

    Returns an empty list if the file content produces no chunks.
    """
    content  = file.get("content", "")
    path     = file.get("path", "")
    ext      = file.get("ext", "")

    file_roles    = file_roles    or {}
    file_purposes = file_purposes or {}

    # Resolve role and build headers.
    full_header,  resolved_role = _build_full_header(path, file_roles, file_purposes)
    short_header                = _build_short_header(path, resolved_role)

    # FIX 5: Run the sliding window on the RAW content (not header+content).
    # We'll prepend the appropriate header to each chunk individually below.
    raw_chunks = _split_into_chunks(content)
    if not raw_chunks:
        return []

    timestamp = datetime.now(timezone.utc).isoformat()

    result = []
    for idx, raw_chunk_text in enumerate(raw_chunks):
        # Prepend full header to chunk 0, short header to all subsequent chunks.
        if idx == 0:
            chunk_text = f"{full_header}\n\n{raw_chunk_text}"
        else:
            chunk_text = f"{short_header}\n\n{raw_chunk_text}"

        chunk = {
            # The actual content that gets embedded and stored.
            # Every chunk now carries a context header (full or short).
            "text": chunk_text,

            # Repo-level metadata — copied from repo_metadata.
            "repo_name":           repo_metadata.get("repo_name"),
            "repo_description":    repo_metadata.get("repo_description"),
            "repo_technologies":   repo_metadata.get("repo_technologies", []),
            "repo_purpose":        repo_metadata.get("repo_purpose"),
            "repo_language":       repo_metadata.get("repo_language"),
            "deployment_url":      repo_metadata.get("deployment_url"),
            "repo_topics":         repo_metadata.get("repo_topics", []),
            "metadata_source":     repo_metadata.get("metadata_source"),
            "metadata_confidence": repo_metadata.get("metadata_confidence"),

            # Richer structural metadata fields.
            "has_authentication":   repo_metadata.get("has_authentication", False),
            "has_database":         repo_metadata.get("has_database", False),
            "database_type":        repo_metadata.get("database_type"),
            "has_api":              repo_metadata.get("has_api", False),
            "api_style":            repo_metadata.get("api_style"),
            "has_frontend":         repo_metadata.get("has_frontend", False),
            "frontend_framework":   repo_metadata.get("frontend_framework"),
            "architecture_pattern": repo_metadata.get("architecture_pattern"),
            "key_features":         repo_metadata.get("key_features", []),
            "external_services":    repo_metadata.get("external_services", []),
            "has_tests":            repo_metadata.get("has_tests", False),

            # File-level fields.
            "file_path":    path,
            "file_type":    ext,
            "file_role":    resolved_role,

            # Chunk position within this file (per-file, starts at 0).
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

    repo_name   = repo_metadata.get("repo_name", "unknown")
    file_count  = len(files)
    chunk_count = len(all_chunks)
    print(f"  [chunker] {repo_name}: {file_count} files → {chunk_count} chunks")

    return all_chunks
