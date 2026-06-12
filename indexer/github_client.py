"""
github_client.py
----------------
All GitHub REST API interactions for the GitHub Brain project.

Responsibilities:
  - Fetch all public, non-fork, non-archived repos for the authenticated user
  - Fetch the recursive file tree for a given repo
  - Fetch raw file content for a given file path
  - Fetch specific known files (README, package.json, etc.) for metadata generation

Rate limit: 5000 requests/hr with a personal access token.
All requests go through _get() which handles errors and rate limit responses uniformly.
"""

import os
import time
import base64
import requests
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"

# Files fetched specifically for metadata generation (not necessarily indexed as chunks).
# Order matters — this matches the tiered fallback priority in metadata_generator.py.
METADATA_CANDIDATE_FILES = [
    "README.md",
    "readme.md",
    "README.MD",
    "package.json",
    "requirements.txt",
    "pyproject.toml",
]

# File extensions excluded regardless of filter mode.
# These are always bad regardless of what the README says.
EXCLUDED_PATH_SEGMENTS = {
    "node_modules",
    "dist",
    "build",
    ".git",
    "__pycache__",
    ".next",
    ".nuxt",
    "coverage",
    "vendor",
}

EXCLUDED_EXTENSIONS = {".lock", ".min.js", ".min.css", ".map", ".pyc", ".log"}

# Files matching these exact names are excluded even if the extension would pass.
EXCLUDED_FILENAMES = {".env", ".env.local", ".env.production", ".DS_Store"}

# Maximum file size in bytes to index (50KB).
MAX_FILE_SIZE_BYTES = 50 * 1024


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------

class GitHubClient:
    """
    Thin wrapper around the GitHub REST API.

    Usage:
        client = GitHubClient(token=os.getenv("GITHUB_TOKEN"))
        repos  = client.get_repos()
        tree   = client.get_file_tree("claimsense", "main")
        content = client.get_file_content("claimsense", "src/parser.py", "main")
    """

    def __init__(self, token: str):
        if not token:
            raise ValueError("GITHUB_TOKEN is required but was not provided.")
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        # Resolved once on first use, cached for all subsequent calls.
        self._username: Optional[str] = None

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get(self, url: str, params: dict = None) -> dict | list:
        """
        Make a GET request. Handles:
          - Rate limit responses (429 / X-RateLimit-Remaining == 0) → waits and retries
          - Non-200 responses → raises RuntimeError with context
        """
        for attempt in range(3):
            response = self.session.get(url, params=params)

            # Rate limit hit — wait until reset then retry.
            if response.status_code == 403:
                remaining = int(response.headers.get("X-RateLimit-Remaining", 1))
                if remaining == 0:
                    reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait = max(reset_time - int(time.time()), 1)
                    print(f"  [github] Rate limit hit. Waiting {wait}s...")
                    time.sleep(wait)
                    continue

            if response.status_code == 404:
                return None  # Caller decides how to handle missing resources.

            if not response.ok:
                raise RuntimeError(
                    f"GitHub API error {response.status_code} for {url}: {response.text[:200]}"
                )

            return response.json()

        raise RuntimeError(f"GitHub API request failed after 3 attempts: {url}")

    def _get_paginated(self, url: str, params: dict = None) -> list:
        """
        Fetch all pages of a paginated GitHub endpoint.
        GitHub paginates with ?page=N&per_page=100.
        """
        params = params or {}
        params["per_page"] = 100
        results = []
        page = 1

        while True:
            params["page"] = page
            data = self._get(url, params=params)
            if not data:
                break
            results.extend(data)
            # GitHub returns fewer than per_page items on the last page.
            if len(data) < 100:
                break
            page += 1

        return results

    def _get_username(self) -> str:
        """Resolve and cache the authenticated user's login name."""
        if self._username is None:
            data = self._get(f"{GITHUB_API_BASE}/user")
            self._username = data["login"]
        return self._username

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_repos(self) -> list[dict]:
        """
        Fetch all public, non-fork, non-archived repos for the authenticated user.

        Returns a list of repo dicts. Each dict is the raw GitHub API response,
        but we only surface the fields downstream modules actually need:
          - name            (str)  exact repo name from GitHub
          - language        (str)  primary language detected by GitHub, may be None
          - topics          (list) user-defined tags
          - description     (str)  one-line description from GitHub settings, may be None
          - default_branch  (str)  usually "main" or "master"
          - html_url        (str)  full GitHub URL for the repo

        Filtering applied here (not downstream):
          - fork: True     → skipped
          - archived: True → skipped
          - private: True  → skipped (only public repos in scope for Phase 1)
        """
        username = self._get_username()
        print(f"[github] Fetching repos for user: {username}")

        all_repos = self._get_paginated(
            f"{GITHUB_API_BASE}/user/repos",
            params={"type": "owner", "sort": "updated"},
        )

        filtered = []
        for repo in all_repos:
            if repo.get("fork"):
                continue
            if repo.get("archived"):
                continue
            if repo.get("private"):
                continue
            filtered.append({
                "name":           repo["name"],
                "language":       repo.get("language"),
                "topics":         repo.get("topics", []),
                "description":    repo.get("description"),
                "default_branch": repo.get("default_branch", "main"),
                "html_url":       repo.get("html_url"),
            })

        print(f"[github] Found {len(filtered)} indexable repos "
              f"(skipped forks, archived, private)")
        return filtered

    def get_file_tree(self, repo_name: str, branch: str) -> list[dict]:
        """
        Fetch the full recursive file tree for a repo branch.

        Returns a flat list of file entries. Each entry has:
          - path  (str)  full path within the repo e.g. "src/auth/login.py"
          - size  (int)  file size in bytes (0 for directories)
          - type  (str)  "blob" for files, "tree" for directories

        Only "blob" (file) entries are returned — directories are excluded.
        Truncation: GitHub truncates trees with > 100,000 entries. This is
        logged as a warning but not treated as a fatal error.
        """
        username = self._get_username()
        url = (f"{GITHUB_API_BASE}/repos/{username}/{repo_name}"
               f"/git/trees/{branch}?recursive=1")
        data = self._get(url)

        if data is None:
            print(f"  [github] Warning: could not fetch tree for {repo_name}@{branch}")
            return []

        if data.get("truncated"):
            print(f"  [github] Warning: file tree for {repo_name} was truncated by GitHub")

        return [
            {
                "path": item["path"],
                "size": item.get("size", 0),
                "type": item["type"],
            }
            for item in data.get("tree", [])
            if item["type"] == "blob"
        ]

    def get_file_content(
        self, repo_name: str, file_path: str, branch: str
    ) -> Optional[str]:
        """
        Fetch the decoded text content of a single file.

        Returns the file content as a UTF-8 string, or None if:
          - The file does not exist (404)
          - The file is binary and cannot be decoded as UTF-8
          - The file exceeds MAX_FILE_SIZE_BYTES

        GitHub returns file content as base64-encoded in the "content" field.
        """
        username = self._get_username()
        url = (f"{GITHUB_API_BASE}/repos/{username}/{repo_name}"
               f"/contents/{file_path}?ref={branch}")
        data = self._get(url)

        if data is None:
            return None

        # Size check before decoding.
        size = data.get("size", 0)
        if size > MAX_FILE_SIZE_BYTES:
            return None

        # Decode base64 content.
        try:
            raw = base64.b64decode(data.get("content", "")).decode("utf-8")
            return raw
        except (UnicodeDecodeError, Exception):
            # Binary file or malformed content — skip it.
            return None

    def get_metadata_file(self, repo_name: str, branch: str) -> tuple[str, str] | tuple[None, None]:
        """
        Attempt to fetch the best available metadata source file for a repo.
        Tries files in METADATA_CANDIDATE_FILES order (README first, then deps).

        Returns:
            (source_type, content) where source_type is one of:
                "readme", "package", "requirements", "pyproject"
            (None, None) if none of the candidate files exist.

        This drives the tiered fallback logic in metadata_generator.py.
        """
        source_type_map = {
            "README.md":       "readme",
            "readme.md":       "readme",
            "README.MD":       "readme",
            "package.json":    "package",
            "requirements.txt":"requirements",
            "pyproject.toml":  "pyproject",
        }

        for filename in METADATA_CANDIDATE_FILES:
            content = self.get_file_content(repo_name, filename, branch)
            if content:
                return source_type_map[filename], content

        return None, None

    def get_indexable_files(
        self, repo_name: str, branch: str, file_tree: list[dict],
        file_filter: dict = None,
    ) -> list[dict]:
        """
        Filter the file tree to files that should be indexed and fetch their content.

        file_filter comes from metadata_generator.generate_repo_metadata() and has
        two possible modes:

        mode = "readme"
            The README explicitly listed important files/folders.
            A file passes if its path starts with any entry in filter["paths"]
            OR its filename exactly matches any entry in filter["paths"].
            Extensions in filter["extensions"] are still checked as a secondary
            gate so binary/generated files with matching paths are excluded.

        mode = "language_fallback" (or file_filter is None)
            No explicit file listing in README. Fall back to extension-based
            filtering using filter["extensions"] (language-appropriate set).
            The hardcoded EXCLUDED_PATH_SEGMENTS still apply to avoid
            node_modules, build artifacts, etc.

        In both modes:
          - Files > MAX_FILE_SIZE_BYTES are skipped (size check is free)
          - Files whose content cannot be decoded as UTF-8 are skipped
          - EXCLUDED_FILENAMES (.env etc.) are always skipped
          - EXCLUDED_EXTENSIONS (.min.js etc.) are always skipped

        Returns a list of dicts:
          {"path": "src/parser.py", "content": "import pdfplumber...", "ext": ".py"}
        """
        if file_filter is None:
            file_filter = {"mode": "language_fallback", "paths": [], "extensions": []}

        mode            = file_filter.get("mode", "language_fallback")
        allowed_paths   = file_filter.get("paths", [])
        allowed_exts    = set(file_filter.get("extensions", []))
        results         = []

        for entry in file_tree:
            path = entry["path"]
            size = entry["size"]

            # Size check — no API call needed.
            if size > MAX_FILE_SIZE_BYTES:
                continue

            filename = path.split("/")[-1]
            ext      = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""

            # Always-excluded filenames and extension patterns.
            if filename in EXCLUDED_FILENAMES:
                continue
            if any(filename.endswith(excl) for excl in EXCLUDED_EXTENSIONS):
                continue

            # --- Mode-specific path/extension matching ---

            if mode == "readme" and allowed_paths:
                # File must match at least one explicitly listed path or prefix.
                # e.g. allowed_paths entry "src/" matches "src/auth/login.py"
                #      allowed_paths entry "app.py" matches "app.py" exactly
                #      allowed_paths entry "server/controllers/" matches any file under it
                matched = False
                for allowed in allowed_paths:
                    allowed_clean = allowed.strip("/")
                    if (
                        path == allowed_clean                        # exact file match
                        or path.startswith(allowed_clean + "/")      # folder prefix match
                        or filename == allowed_clean                  # bare filename match
                    ):
                        matched = True
                        break
                if not matched:
                    continue

                # Secondary extension gate — skip binary/generated even if path matched.
                if allowed_exts and ext not in allowed_exts:
                    continue

            else:
                # language_fallback mode — extension-based filtering.
                if ext not in allowed_exts:
                    continue

                # Excluded directory segments (node_modules, build, dist, etc.)
                path_parts = set(path.split("/"))
                if path_parts & EXCLUDED_PATH_SEGMENTS:
                    continue

            # Fetch content.
            content = self.get_file_content(repo_name, path, branch)
            if content is None:
                continue

            results.append({
                "path":    path,
                "content": content,
                "ext":     ext,
            })

        return results
