"""
indexer/main.py
---------------
Orchestrates the full indexing pipeline for GitHub Brain.

Wires together:
  github_client.py      → fetch repos and files
  metadata_generator.py → generate structured metadata per repo
  chunker.py            → split files into overlapping chunks
  embedder.py           → embed chunks via Gemini
  deeplake_store.py     → store embedded chunks in Deep Lake

Two indexing modes (called from cli.py):
  index_all()           → full re-index of all eligible repos
  index_repo(name)      → re-index a single repo by name
"""

import os
from indexer.github_client      import GitHubClient
from indexer.metadata_generator import generate_repo_metadata
from indexer.chunker             import chunk_repo_files
from indexer.embedder            import embed_chunks
from indexer.deeplake_store      import store_chunks, delete_repo_chunks


# ---------------------------------------------------------------------------
# Internal: index a single repo
# ---------------------------------------------------------------------------

def _index_single_repo(
    client: GitHubClient,
    repo: dict,
    delete_existing: bool = False,
) -> dict:
    """
    Run the full pipeline for one repo.

    Parameters:
        client           Authenticated GitHubClient instance
        repo             Repo dict from GitHubClient.get_repos()
        delete_existing  If True, delete existing chunks before re-indexing.
                         Used for single-repo re-index mode.

    Returns a summary dict:
        {
            "repo_name":   "claimsense",
            "files":       12,
            "chunks":      87,
            "stored":      87,
            "skipped":     0,
            "status":      "ok" | "error",
            "error":       None | "error message"
        }
    """
    repo_name = repo["name"]
    branch    = repo["default_branch"]

    print(f"\n{'='*60}")
    print(f"[indexer] Processing: {repo_name} (branch: {branch})")
    print(f"{'='*60}")

    summary = {
        "repo_name": repo_name,
        "files":     0,
        "chunks":    0,
        "stored":    0,
        "status":    "ok",
        "error":     None,
    }

    try:
        # Step 1: Fetch file tree (used for both metadata and file content).
        print(f"  [indexer] Fetching file tree...")
        file_tree = client.get_file_tree(repo_name, branch)
        if not file_tree:
            print(f"  [indexer] Empty file tree. Skipping.")
            summary["status"] = "skipped"
            return summary

        # Step 2: Metadata generation.
        # Returns (metadata dict, file_filter dict, file_roles dict, file_purposes dict).
        # metadata      → stored on every chunk in Deep Lake.
        # file_filter   → used only here to decide which files to fetch, then discarded.
        # file_roles    → per-file role from README's curated list, used by chunker.py
        #                 to build context headers. Empty dict if README had no list.
        # file_purposes → per-file purpose description from README's curated list,
        #                 used by chunker.py alongside file_roles. Empty dict if
        #                 README had no list.
        print(f"  [indexer] Generating metadata...")
        source_type, source_content = client.get_metadata_file(repo_name, branch)
        repo_metadata, file_filter, file_roles, file_purposes = generate_repo_metadata(
            repo, source_type, source_content
        )

        # Step 3: Fetch indexable file contents.
        # file_filter tells get_indexable_files() which paths/extensions to include.
        print(f"  [indexer] Fetching indexable files "
              f"(filter mode: {file_filter['mode']})...")
        files = client.get_indexable_files(repo_name, branch, file_tree, file_filter)
        summary["files"] = len(files)

        if not files:
            print(f"  [indexer] No indexable files found. Skipping.")
            summary["status"] = "skipped"
            return summary

        # Step 4: Chunk files.
        # file_roles and file_purposes are passed through so chunker.py can build
        # a context header for each file ("File: ... | Role: ... | Purpose: ...").
        # For files not covered by these dicts, chunker.py falls back to
        # filename/folder-based role inference.
        print(f"  [indexer] Chunking {len(files)} files...")
        chunks = chunk_repo_files(files, repo_metadata, file_roles, file_purposes)
        summary["chunks"] = len(chunks)

        if not chunks:
            print(f"  [indexer] No chunks produced. Skipping.")
            summary["status"] = "skipped"
            return summary

        # Step 5: Embed chunks.
        print(f"  [indexer] Embedding {len(chunks)} chunks...")
        embedded_chunks = embed_chunks(chunks)

        # Step 6: Delete existing if re-indexing.
        if delete_existing:
            print(f"  [indexer] Deleting existing chunks for {repo_name}...")
            delete_repo_chunks(repo_name)

        # Step 7: Store in Deep Lake.
        print(f"  [indexer] Storing chunks in Deep Lake...")
        stored = store_chunks(embedded_chunks)
        summary["stored"] = stored

        print(f"  [indexer] ✓ {repo_name}: {len(files)} files, "
              f"{len(chunks)} chunks, {stored} stored")

    except Exception as e:
        print(f"  [indexer] ✗ Error processing {repo_name}: {e}")
        summary["status"] = "error"
        summary["error"]  = str(e)

    return summary


# ---------------------------------------------------------------------------
# Public: index all repos
# ---------------------------------------------------------------------------

def index_all() -> list[dict]:
    """
    Full index of all eligible repos for the authenticated GitHub user.

    Eligible = public, non-fork, non-archived (filtered in github_client.py).

    This does NOT delete existing data first — it appends. If you want a
    completely clean re-index, use the CLI: `python cli.py index --mode full --clean`
    (the --clean flag calls delete_repo_chunks for each repo before indexing).

    Returns a list of per-repo summary dicts for the final report.
    """
    token = os.getenv("GITHUB_TOKEN")
    client = GitHubClient(token=token)

    print("[indexer] Starting full index...")
    repos = client.get_repos()

    if not repos:
        print("[indexer] No eligible repos found.")
        return []

    print(f"[indexer] {len(repos)} repos to index.")

    summaries = []
    for i, repo in enumerate(repos):
        print(f"\n[indexer] Repo {i+1}/{len(repos)}")
        summary = _index_single_repo(client, repo, delete_existing=False)
        summaries.append(summary)

    _print_report(summaries)
    return summaries


# ---------------------------------------------------------------------------
# Public: index one repo
# ---------------------------------------------------------------------------

def index_repo(repo_name: str) -> dict:
    """
    Re-index a single repo by name.

    Deletes all existing chunks for this repo first, then re-indexes from scratch.
    Useful after pushing significant changes to a repo.

    Returns the summary dict for this repo.
    """
    token  = os.getenv("GITHUB_TOKEN")
    client = GitHubClient(token=token)

    print(f"[indexer] Re-indexing repo: {repo_name}")

    # Find the repo in the user's repo list to get branch info.
    all_repos = client.get_repos()
    repo = next((r for r in all_repos if r["name"] == repo_name), None)

    if repo is None:
        print(f"[indexer] Repo '{repo_name}' not found in your public, non-fork repos.")
        return {"repo_name": repo_name, "status": "not_found", "error": "Repo not found"}

    summary = _index_single_repo(client, repo, delete_existing=True)
    _print_report([summary])
    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(summaries: list[dict]):
    """Print a summary table after indexing completes."""
    print(f"\n{'='*60}")
    print(f"INDEXING REPORT")
    print(f"{'='*60}")

    ok      = [s for s in summaries if s["status"] == "ok"]
    skipped = [s for s in summaries if s["status"] == "skipped"]
    errors  = [s for s in summaries if s["status"] == "error"]

    total_chunks = sum(s.get("stored", 0) for s in ok)

    print(f"  Indexed:  {len(ok)} repos  ({total_chunks} chunks stored)")
    print(f"  Skipped:  {len(skipped)} repos  (empty or no indexable files)")
    print(f"  Errors:   {len(errors)} repos")

    if errors:
        print(f"\n  Failed repos:")
        for s in errors:
            print(f"    ✗ {s['repo_name']}: {s['error']}")

    print(f"{'='*60}\n")
