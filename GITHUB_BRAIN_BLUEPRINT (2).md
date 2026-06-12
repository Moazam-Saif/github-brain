# GitHub Brain — Project Blueprint
> This document is the single source of truth for the GitHub Brain project.
> Any developer or AI assistant working on this project should read this first.
> Use it to understand decisions made, why they were made, and where the project stands.

---

## 1. What Is This Project?

A CLI-first tool that lets you chat with your entire GitHub account using natural language.

**Example interactions:**
```
You: Do I have any repo using Redis?
Bot: Yes — found in 2 repos:
     • skillswap → used for session caching
     • beeminor  → used for job queue

You: How does authentication work in claimsense?
Bot: ClaimSense uses token-based auth via Supabase. The verify_token()
     function in src/auth.py checks the JWT against...

You: What repos do I have deployed?
Bot: 2 deployed repos:
     • claimsense → https://claimsense.up.railway.app
     • corplaw-ai → https://corplaw.vercel.app
```

---

## 2. Core Concepts (Read This Before Anything Else)

### What is RAG?
RAG (Retrieval Augmented Generation) solves the problem that LLMs don't know about your private data.
Instead of fine-tuning a model on your code, you:
1. Store your code as embedding vectors in a vector database
2. At query time, retrieve the most relevant chunks using similarity search
3. Feed those chunks to the LLM as context
4. The LLM answers based on your actual code, not its training data

**This entire project is a RAG system.**

### What is an embedding?
A vector representation of text — a list of floats that captures semantic meaning.
Similar meaning = vectors that are close together in space.
This is how "do I use Redis?" can match a chunk that says `redis_client = Redis(host=...)` —
the meanings are similar even if the words differ.

### What is a chunk?
LLMs and embedding models can't process entire files at once efficiently.
Files are split into overlapping segments called chunks.
- Each chunk is ~512 tokens (~400 words / ~30-50 lines of code)
- Chunks overlap by ~50 tokens so context isn't lost at file boundaries
- Each chunk is embedded and stored independently
- Each chunk carries the full repo metadata so you always know where it came from

### Tokens
- 1 token ≈ 4 characters ≈ 0.75 words
- Code tokenizes less efficiently than prose (symbols, indentation)
- A 100-line Python file ≈ 400–800 tokens
- Chunk size of 512 tokens is the empirically established sweet spot:
  - Large enough to contain a complete function with context
  - Small enough that retrieved chunks stay focused on one concept

### Why top-k = 4?
When searching the vector DB, you retrieve the k most similar chunks.
All k chunks go into the LLM's context window.
- k too low → might miss relevant context
- k too high → noisy context, LLM gets confused (known as "lost in the middle" problem)
- k = 4 is the default starting point, tuned later based on answer quality
- Cross-repo queries may use higher k, repo-specific queries lower k

---

## 3. What This Is NOT

The README metadata extraction step (feeding README to Gemini to get JSON) is **not RAG**.
It is plain structured extraction — a one-time LLM call at index time that preprocesses
a document into structured fields. No retrieval, no vector search, no similarity scores.

```
README → [Gemini extraction call] → {technologies: [...], deployment_url: "..."}
```

This runs once per repo during indexing. Its output becomes metadata attached to every
chunk from that repo, making the RAG system smarter.

---

## 4. Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Embeddings | Gemini `text-embedding-004` | Free tier: 100 req/min |
| LLM | Gemini `gemini-1.5-flash` | Free tier: 15 req/min, 1M tokens/day |
| Metadata extraction | Gemini `gemini-1.5-flash` | Same model, one call per repo |
| Vector store | Activeloop Deep Lake | Free tier, cloud-hosted, metadata filtering |
| GitHub data | GitHub REST API | 5000 req/hr with personal token |
| Backend | FastAPI (Python) | Lightweight, async, easy to extend later |
| Interface | CLI (Phase 1), React + TypeScript (Phase 2) | Validate core before building UI |

---

## 5. Metadata Schema

Every chunk stored in Deep Lake carries this metadata:

```json
{
  "repo_name":           "claimsense",
  "repo_language":       "python",
  "repo_topics":         ["ai", "flask", "python"],

  "repo_description":    "AI insurance claim verification agent using adversarial LLM pattern",
  "repo_technologies":   ["python", "flask", "gemini", "supabase", "postgresql"],
  "repo_purpose":        "web app",
  "deployment_url":      "https://claimsense.up.railway.app",

  "has_authentication":    true,
  "has_database":          true,
  "database_type":         "postgresql",
  "has_api":               true,
  "api_style":             "REST",
  "has_frontend":          true,
  "frontend_framework":    null,
  "architecture_pattern":  "agent-based",
  "key_features":          ["claim submission", "multi-agent verification", "admin review queue"],
  "external_services":     ["supabase", "railway", "vertex ai"],
  "has_tests":             false,

  "metadata_source":     "readme",
  "metadata_confidence": "high",
  "file_path":           "src/parser.py",
  "file_type":           ".py",
  "chunk_index":         3,
  "indexed_at":          "2026-06-10T12:00:00+00:00"
}
```

### Field ownership

| Field | Source | Method |
|---|---|---|
| `repo_name` | GitHub API | Always. Never inferred. |
| `repo_language` | GitHub API | Always. |
| `repo_topics` | GitHub API | Always. Stored as-is. |
| `repo_description` | Gemini | From metadata source |
| `repo_technologies` | Gemini | From metadata source |
| `repo_purpose` | Gemini | From metadata source |
| `deployment_url` | Gemini | From README only. null if not found. |
| `has_authentication` | Gemini | true if any auth/login/JWT/OAuth mentioned |
| `has_database` | Gemini | true if any DB or persistence mentioned |
| `database_type` | Gemini | specific DB name or null |
| `has_api` | Gemini | true if HTTP routes/endpoints present |
| `api_style` | Gemini | REST / GraphQL / other / null |
| `has_frontend` | Gemini | true if any UI present |
| `frontend_framework` | Gemini | specific UI framework or null |
| `architecture_pattern` | Gemini | MVC / microservices / agent-based / etc / null |
| `key_features` | Gemini | 3-6 main features from README Features section |
| `external_services` | Gemini | third-party integrations beyond main stack |
| `has_tests` | Gemini | true if tests mentioned |
| `metadata_source` | System | Set by tiered fallback logic |
| `metadata_confidence` | System | high / medium / low |
| `file_path` | GitHub API | File path within repo |
| `file_type` | System | File extension |
| `chunk_index` | System | Position within file |
| `indexed_at` | System | ISO timestamp |

### Why these extra fields reduce vector search

With only basic fields, questions like "which repos have authentication?" would route
to cross_repo_semantic and trigger a full vector search. With `has_authentication: true`
in metadata, the router sends it to cross_repo_metadata — answered from stored metadata
in milliseconds, no chunks, no embeddings.

Questions now answerable purely from metadata:
```
"which repos have a database?"          → has_database
"which repos use PostgreSQL?"           → database_type
"do I have anything with a REST API?"   → has_api + api_style
"which projects have a frontend?"       → has_frontend
"do any repos have tests?"              → has_tests
"which projects use Cloudinary?"        → external_services
"which repos use MVC architecture?"     → architecture_pattern
"what features does claimsense have?"   → key_features
```

### deployment_url rules
- If a deployment URL is found in the README → store the full URL string
- If not found → store null
- Never invent or guess URLs

---

## 6. Metadata Tiered Fallback

Not all repos have READMEs. Metadata source is chosen in this priority order:

```
1. README.md / readme.md        → confidence: high
2. package.json                 → confidence: medium
3. requirements.txt             → confidence: medium
4. pyproject.toml               → confidence: medium
5. GitHub API fields only       → confidence: low
   (name, description, language, topics — no Gemini call)
```

For cases 1–4, one Gemini extraction call is made.
For case 5, no Gemini call is made. Raw GitHub API fields are used directly.

Repos with no README and no dependency file still get indexed —
their code chunks are stored with whatever metadata is available.
`metadata_confidence: "low"` signals this to the query layer.

---

## 7. Gemini Extraction Prompt

The full prompt is in `metadata_generator.py`. It extracts all metadata fields
including the richer structural fields in a single Gemini call per repo.

Fields extracted:
```
repo_description      one clear sentence about what the project does
repo_technologies     languages, frameworks, databases, external services
repo_purpose          web app / cli tool / library / ml project / api / desktop app / other
deployment_url        real URL from README or null

has_authentication    true/false — any login/JWT/OAuth mentioned
has_database          true/false — any DB or persistence mentioned
database_type         postgresql / mysql / mongodb / sqlite / supabase / etc / null
has_api               true/false — any HTTP routes or endpoints
api_style             REST / GraphQL / gRPC / other / null
has_frontend          true/false — any UI framework or interface
frontend_framework    react / vue / javafx / etc / null
architecture_pattern  MVC / microservices / agent-based / monolith / etc / null
key_features          3-6 main features from the Features section
external_services     third-party APIs/platforms beyond main stack
has_tests             true/false — any testing mentioned

files_to_index        found (bool), paths (list), extensions (list)
```

All returned as a single JSON object. Parsed and validated defensively —
wrong types are normalized to safe defaults rather than crashing.

---

## 8. Files to Index Per Repo

### How files are selected

File selection is README-driven, not hardcoded. The system has two modes:

**Mode 1 — README-driven (preferred)**
If the README explicitly lists important files or folders (e.g. a "Project Structure"
section, "Only the files that matter" block, or any code block listing source files
with descriptions), Gemini extracts those paths during metadata generation.
Only files whose paths match the extracted list are fetched and indexed.
Files not mentioned in the README are ignored — even if their extension would normally pass.

**Mode 2 — Language fallback**
If the README has no explicit file listing (or there is no README), the system falls
back to a language-appropriate extension set derived from the GitHub API `language` field:

```
Python     → .py .pyi
JavaScript → .js .mjs .cjs .jsx
TypeScript → .ts .tsx
Java       → .java
C++        → .cpp .cc .cxx .h .hpp .hxx
C          → .c .h
C#         → .cs
Go         → .go
Rust       → .rs
CMake      → .cmake + CMakeLists.txt
...and so on. Falls back to a broad multi-language default if language is unknown.
```

`.md` files are always included regardless of mode (READMEs provide context).

### Always excluded (both modes)
```
node_modules/    dist/           build/          .git/
__pycache__/     .next/          coverage/       vendor/
*.lock           *.min.js        *.min.css       *.map
*.pyc            *.log           .env            .DS_Store
```

### Size limit
Skip any file larger than 50KB — almost always generated or minified.

### The file_filter object
`metadata_generator.py` returns a `file_filter` dict alongside the metadata dict.
It is passed to `github_client.get_indexable_files()` and then discarded.
It is never stored in Deep Lake — it is only used during indexing.

```python
# README-driven example
file_filter = {
    "mode":       "readme",
    "paths":      ["app.py", "src/", "server/controllers/", "templates/"],
    "extensions": [".py", ".html", ".md"],
}

# Language fallback example
file_filter = {
    "mode":       "language_fallback",
    "paths":      [],
    "extensions": [".java", ".md"],
}
```

---

## 9. Indexing Pipeline — Step by Step

```
Step 1: Fetch all repos
  GitHub API: GET /user/repos (paginated, handle all pages)
  Filter out: forks (fork: true), archived repos (archived: true)
  Result: list of repo objects with name, language, topics, description, default_branch

Step 2: Per-repo metadata generation
  Fetch file tree: GET /repos/{owner}/{repo}/git/trees/{branch}?recursive=1
  Run tiered fallback logic to pick metadata source
  Make Gemini extraction call if source is readme/package/requirements/pyproject
  Merge Gemini output with GitHub API fields into final metadata object
  Cache result — skip this step on re-index if nothing changed (future optimization)

Step 3: Fetch and filter files
  Using file tree from Step 2 (already fetched)
  Apply include/exclude rules and 50KB size limit
  Fetch file content for each included file via GitHub API

Step 4: Chunk files
  Split each file into chunks of 512 tokens
  Overlap consecutive chunks by 50 tokens
  Attach full repo metadata + file_path + file_type + chunk_index to each chunk

Step 5: Embed chunks
  Send chunk text to Gemini text-embedding-004
  Batch requests — stay within 100 req/min rate limit
  Add delay between batches if needed

Step 6: Store in Deep Lake
  One dataset for the entire GitHub account: hub://YOUR_ORG/github_brain
  Store: embedding vector + text + metadata JSON
  overwrite=False by default — append new repos without wiping existing data
  For re-indexing a specific repo: delete that repo's chunks first, then re-add
```

---

## 10. Deep Lake Dataset Structure

```
Dataset: hub://YOUR_ORG/github_brain

Tensors:
  embedding   → float32 (768-dim, Gemini text-embedding-004 output size)
  text        → str     (the raw chunk content)
  metadata    → json    (full metadata object per chunk)
```

---

## 11. The Five Query Types

Every question is classified into one of five types before any search happens.

**list_repos**
```
"what repos do I have?" / "show me my projects"
→ list_all_repos() from metadata, no vector search
→ Gemini formats the list conversationally
```

**cross_repo_metadata**
```
"do I have anything using Redis?" / "which repos are deployed?"
→ list_all_repos() — returns every repo's metadata summary
→ Gemini filters and answers from that metadata
→ no embeddings, no vector search — fastest path
```

**cross_repo_semantic**
```
"do any of my repos implement rate limiting?"
→ embed question → similarity_search across ALL chunks (no repo filter)
→ top-k = 6 chunks from whichever repos scored highest
→ Gemini answers: find/not-find, not a ranking
```

**cross_repo_comparative**
```
"which repo has the best authentication?"
→ embed question → similarity_search_aggregated (candidate_k=50)
→ group chunks by repo, score each repo = avg of its top-3 chunk scores
→ rank repos, take top-3 repos × top-3 chunks each
→ Gemini compares fairly — each repo gets equal representation
```

**repo_specific**
```
"how does auth work in claimsense?" / "walk me through the parser" (active session)
→ embed enriched query (question + last 2 turns) → search within repo only
→ top-k = 4 chunks, deduplicated against already-seen chunks
→ Gemini answers with full session history as context
→ session persists across turns
```

### Why cross_repo_semantic and cross_repo_comparative are different

cross_repo_semantic returns the global top-k chunks. For "do any repos use WebSockets?"
this is fine — you just need to find relevant chunks somewhere.

cross_repo_comparative cannot use global top-k. For "which repo has the best auth?",
global top-k would return 5-6 chunks all from the one repo with the most auth code,
completely ignoring other repos. The aggregated approach scores each repo independently
so the comparison is fair regardless of repo size or chunk count.

---

## 12. Query Pipeline — Step by Step

```
Step 1: Receive user question

Step 2: Route the question (one Gemini call → router.py)
  Returns one of:
    {"type": "list_repos"}
    {"type": "cross_repo_metadata"}
    {"type": "cross_repo_semantic"}
    {"type": "cross_repo_comparative"}
    {"type": "repo_specific", "repo": "claimsense"}

Step 3a: list_repos
  list_all_repos() → all repo metadata summaries
  _build_metadata_prompt() → Gemini answers
  No session. Done.

Step 3b: cross_repo_metadata
  list_all_repos() → all repo metadata summaries
  _build_metadata_prompt() → Gemini filters and answers
  No vector search. No embeddings. No session. Done.

Step 3c: cross_repo_semantic
  embed_query(question) → 768-dim vector
  similarity_search(repo_name=None, top_k=6) → 6 highest-scoring chunks globally
  _build_semantic_prompt() → Gemini answers
  No session. Done.

Step 3d: cross_repo_comparative
  embed_query(question) → 768-dim vector
  similarity_search_aggregated(candidate_k=50, top_repos=3, chunks_per_repo=3)
    → fetch top-50 chunks globally
    → group by repo_name
    → score each repo = avg of its top-3 chunk scores
    → rank repos, return top-3 with best 3 chunks each
  _build_comparative_prompt() → Gemini compares repos fairly
  No session. Done.

Step 3e: repo_specific
  Check session: start new or continue existing
  _manage_context_window() if history is long
  build enriched query = question + last 2 conversation turns
  embed_query(enriched) → 768-dim vector
  similarity_search(repo_name=X, top_k=8) → filter to repo, fetch 8
  deduplicate against seen_chunk_ids → return top 4 fresh chunks
  _build_repo_specific_prompt() → Gemini answers with history + code
  Append turn to session. Return updated session.
```

---

## 13. Session Management (Repo Deep Dives)

### Why sessions are needed

A deep dive is a conversation, not a single query. Each follow-up gives
more signal about what the user is looking for. Without session state,
every question is answered in isolation and the system loses all context
from previous turns.

### Session object

Lives in memory for the duration of the CLI conversation. No database needed.

```python
session = {
    "active_repo":         "claimsense",   # which repo is in focus
    "conversation_history": [              # running list of turns
        {"role": "user",      "content": "how does the PDF parser work?"},
        {"role": "assistant", "content": "The parser uses pdfplumber to..."}
    ],
    "seen_chunk_ids":      set(),          # deduplication — chunk hashes already used
    "repo_metadata":       {...}           # loaded once at session start, reused every turn
}
```

### Conversation-aware retrieval

The query sent to Deep Lake is NOT the raw user message — it is enriched
with prior context so the vector search finds better matches:

```
# Naive (bad)
search_query = "how does that feed into the verification step?"

# Enriched (good)
search_query = """
Previous context: PDF parsing using pdfplumber, extracts fields into dict
Current question: how does that feed into the verification step?
"""
```

The enriched query produces far better vector matches because it carries
semantic context from the conversation. Use the last 2 turns as context
for enrichment — going further back adds noise.

### Deduplication

Without deduplication, the same highly-similar chunks are retrieved
repeatedly across turns, wasting context space and confusing the LLM.

Every retrieved chunk gets hashed and added to seen_chunk_ids.
On subsequent turns, chunks already in seen_chunk_ids are filtered out
before being added to the context window.

### Context window structure per turn

```
[system prompt]                    ← fixed, describes the assistant's role
[repo metadata summary]            ← one paragraph, loaded once at session start
[summarized older turns]           ← older history compressed into one paragraph
[last 4 conversation turns]        ← recent history kept verbatim
[newly retrieved chunks]           ← fresh code context for this question
[current user question]            ← the actual question
```

### Sliding window with summarization

Conversation history cannot grow indefinitely. Management strategy:

```
Keep always:
  - System prompt
  - Repo metadata summary
  - Last 4 turns verbatim

When history exceeds 4 turns:
  - Older turns are summarized into one paragraph by Gemini
  - That summary replaces the raw older turns
  - Prevents context window bloat without losing conversational continuity
```

### Session lifecycle

```
Session starts    → user asks about a specific repo (router returns repo_specific)
Session continues → every follow-up that stays within the same repo
Session resets    → user asks cross_repo_metadata, cross_repo_semantic, or list_repos
                  → user explicitly names a different repo
                  → user types "reset" in the CLI
```

When a session resets, conversation_history and seen_chunk_ids are cleared.
repo_metadata is re-loaded if the new repo is different.

---

## 14. Query Router Prompt

See `query/router.py` for the full prompt. Summary of classification rules:

```
list_repos          → user wants to list, count, or browse repos
                      "what repos do I have?" / "show me my projects"

cross_repo_metadata → question about technology, language, deployment, or
                      project category answerable from stored metadata
                      "do I have anything using Redis?"
                      "which repos are deployed?"
                      "do I have any Java projects?"

cross_repo_semantic → question about concept or pattern requiring actual code
                      "which repo implements rate limiting?"
                      "do any repos handle file uploads?"

repo_specific       → targets one repo by name or active session
                      "how does auth work in claimsense?"
                      "walk me through the parser" (active session)

Fallback: cross_repo_semantic (safe — searches broadly rather than wrong repo)
```

---

## 15. File Structure

```
github-brain/
│
├── indexer/
│   ├── main.py                  ← orchestrates full pipeline, CLI entry point
│   ├── github_client.py         ← all GitHub API calls (fetch repos, files, trees)
│   ├── metadata_generator.py    ← tiered fallback logic + Gemini extraction call
│   ├── chunker.py               ← splits file text into overlapping token chunks
│   ├── embedder.py              ← sends chunks to Gemini embedding API
│   └── deeplake_store.py        ← all Deep Lake read/write operations
│
├── query/
│   ├── engine.py                ← main query function (router + search + answer)
│   ├── router.py                ← classifies question type via Gemini
│   └── retriever.py             ← Deep Lake similarity search with optional filtering
│
├── cli.py                       ← interactive CLI loop (Phase 1 interface)
│
├── .env                         ← API keys (never committed)
├── .env.example                 ← template showing required keys
├── requirements.txt
└── BLUEPRINT.md                 ← this file
```

---

## 16. Environment Variables

```
GITHUB_TOKEN=         # classic personal token, public_repo scope is enough
ACTIVELOOP_TOKEN=     # from app.activeloop.ai
GEMINI_API_KEY=       # from aistudio.google.com
ACTIVELOOP_ORG=       # your Activeloop username/org
```

---

## 17. Indexing Modes

```
python cli.py index --mode full
  → Indexes all non-fork, non-archived public repos from scratch

python cli.py index --mode repo --name claimsense
  → Re-indexes one specific repo only

python cli.py chat
  → Starts the interactive query CLI (indexing must have run first)
```

---

## 18. Known Constraints and Limits

| Constraint | Limit | Mitigation |
|---|---|---|
| Gemini embedding | 100 req/min free | Batch with delays |
| Gemini Flash | 15 req/min free | Queue calls, don't parallelize |
| GitHub API | 5000 req/hr with token | Paginate carefully, reuse file tree |
| Deep Lake free tier | Storage limits apply | One dataset, efficient chunking |
| File size | Skip > 50KB | Avoids generated/minified files |

---

## 19. What Is Not In Scope (Phase 1)

- Private repos
- Forked repos
- Archived repos
- Automatic re-indexing / webhooks / cron jobs
- React frontend (Phase 2)
- Authentication / multi-user support
- Docker metadata fields (is_dockerized, has_ci_cd)

---

## 20. Build Order

```
1. github_client.py          ← fetch repos and files
2. metadata_generator.py     ← tiered fallback + Gemini extraction
3. chunker.py                ← split files into chunks
4. embedder.py               ← embed chunks via Gemini
5. deeplake_store.py         ← store and retrieve from Deep Lake
6. indexer/main.py           ← wire 1-5 together into full pipeline
7. query/router.py           ← classify question type
8. query/retriever.py        ← similarity search with metadata filtering
9. query/engine.py           ← full query flow
10. cli.py                   ← interactive interface
```

Each module is tested independently before wiring together.

---

## 21. File Purposes and How Each File Works

A complete reference for every file in the project. Read this to understand
what a file owns, what it does not own, and how it connects to other files.

---

### `indexer/github_client.py`
**Purpose:** The only file that talks to the GitHub REST API.
**Owns:** Fetching repo lists, file trees, file content, metadata candidate files.
**Does not own:** Any business logic about what to do with the data.

How it works:
- Uses a persistent `requests.Session` so auth headers are set once, not per call
- `get_repos()` paginates through all repos and filters out forks/archived/private
- `get_file_tree()` fetches the full recursive tree in one call (`?recursive=1`)
  and caches it — the same tree is used for metadata detection AND file fetching
- `get_metadata_file()` tries README → package.json → requirements.txt → pyproject.toml
  in order, returns on the first hit
- `get_indexable_files()` accepts a `file_filter` dict from metadata_generator.
  In "readme" mode: matches paths by prefix or exact filename.
  In "language_fallback" mode: matches by extension, skips excluded directories.
- Rate limit handling: reads `X-RateLimit-Reset` header and sleeps until reset

---

### `indexer/metadata_generator.py`
**Purpose:** Generate structured metadata and file filter config for each repo.
**Owns:** One Gemini extraction call per repo. Tiered fallback logic.
**Does not own:** File fetching, embedding, or storage.

How it works:
- Receives the metadata source file (README or deps file) from github_client
- Makes ONE Gemini call that returns both repo metadata AND files_to_index in the same response
- Merges Gemini output with GitHub API fields (repo_name, language, topics always from GitHub)
- Returns a tuple: `(metadata_dict, file_filter_dict)`
  - `metadata_dict` is stored on every chunk in Deep Lake permanently
  - `file_filter_dict` is used only during indexing to decide which files to fetch, then discarded
- If README has explicit file listing → file_filter mode = "readme"
- If not → file_filter mode = "language_fallback" using language from GitHub API
- `.md` always included via ALWAYS_INCLUDE_EXTENSIONS regardless of mode

**This is NOT RAG.** It is plain structured extraction — one LLM call, document in, JSON out.

---

### `indexer/chunker.py`
**Purpose:** Split file content into overlapping chunks and attach metadata.
**Owns:** The chunking logic and the structure of each chunk dict.
**Does not own:** File fetching or embedding.

How it works:
- Uses character-based approximation: 1 token ≈ 4 chars, so 512 tokens = 2048 chars
- Sliding window: step = chunk_size - overlap = 2048 - 200 = 1848 chars per step
- Each chunk carries the full repo metadata from metadata_generator plus
  file_path, file_type, chunk_index, and indexed_at
- chunk_index is per-file (starts at 0 for each file), not global
- Empty or whitespace-only content produces zero chunks — silently skipped

Why overlap: prevents function/class definitions from being split across chunks
without any surrounding context in either half.

---

### `indexer/embedder.py`
**Purpose:** Convert text into 768-dimensional semantic vectors.
**Owns:** All calls to Gemini's text-embedding-004 model.
**Does not own:** What to do with the vectors.

How it works:
- `embed_chunks()` — called at index time, uses `task_type="RETRIEVAL_DOCUMENT"`
- `embed_query()` — called at query time, uses `task_type="RETRIEVAL_QUERY"`
- The asymmetry matters: Gemini's embedding model produces slightly different
  vector spaces for documents vs queries. Mismatching them degrades search quality.
- Rate limiting: 0.65s sleep between calls stays under 100 req/min free tier
- Failed embeddings return None and are filtered out before storage

---

### `indexer/deeplake_store.py`
**Purpose:** All read and write operations against the Deep Lake vector database.
**Owns:** Dataset creation, chunk storage, chunk deletion, similarity search,
          aggregated repo-level search, repo listing.
**Does not own:** Embedding generation or metadata generation.

How it works:
- One dataset (`hub://ORG/github_brain`) holds all repos — no per-repo datasets
- Each row stores: embedding (float32[768]), text (str), metadata (JSON str)

**`similarity_search()` — batch matrix multiplication:**
  1. `ds.embedding.numpy()` loads ALL vectors as matrix M shape (N, 768) — one network call
  2. Normalize M and query vector to unit length
  3. `M @ query_vector` — one matrix multiply gives N cosine scores simultaneously
  4. `np.argsort` to find top-k indices
  Replaces old per-row loop (N network calls, ~60s) with 1 call + 1 multiply (~1s).

**`similarity_search_aggregated()` — repo-level scoring for comparative questions:**
  1. Call similarity_search(candidate_k=50) — broad search across all chunks
  2. Group chunks by repo_name
  3. For each repo: repo_score = average of its top-3 chunk scores
     (top-3 average is fair across repo sizes — not biased by chunk count)
  4. Rank repos by repo_score descending
  5. Return top_repos repos, each with their best chunks_per_repo chunks
  Used by retrieve_cross_repo_comparative() in retriever.py.

- `delete_repo_chunks()` deletes in reverse index order — forward deletion shifts
  indices causing wrong rows to be deleted
- `list_all_repos()` reuses `_load_all()` but ignores embeddings — only reads metadata
  to build one summary dict per unique repo_name

---

### `indexer/main.py`
**Purpose:** Orchestrate the full indexing pipeline for one or all repos.
**Owns:** The order of pipeline steps and the indexing summary report.
**Does not own:** Any individual step — it only calls other modules.

How it works:
- `index_all()` → fetches all repos, calls `_index_single_repo()` for each
- `index_repo(name)` → finds the named repo, calls `_index_single_repo()` with delete_existing=True
- `_index_single_repo()` runs: file_tree → metadata → files → chunks → embeddings → store
- Produces a summary report at the end showing ok/skipped/error counts

---

### `query/router.py`
**Purpose:** Classify each user question into one of five query types.
**Owns:** One Gemini call per question. The classification logic and prompt.
**Does not own:** Any search or answer generation.

How it works:
- Sends the question + active_repo hint to Gemini Flash with a detailed prompt
- Gemini returns JSON: `{"type": "cross_repo_metadata"}` etc.
- Falls back to `cross_repo_semantic` on failure

Five types and when each is used:
- `list_repos` → user wants to see/count repos
- `cross_repo_metadata` → technology/language/deployment (answerable from metadata)
- `cross_repo_semantic` → concept exists somewhere? (requires code, answer is find/not-find)
- `cross_repo_comparative` → which repo is best at X? (requires fair ranking)
- `repo_specific` → targets one named repo or active session repo

The key distinction the router must make:
- metadata vs semantic: can the answer come from stored metadata fields or does it need code?
- semantic vs comparative: is the answer find/not-find or a ranking/comparison?

---

### `query/retriever.py`
**Purpose:** Translate a classified question into actual search results.
**Owns:** The four retrieval strategies. Query enrichment. Chunk deduplication.
**Does not own:** Classification (router) or answer generation (engine).

How it works:

`retrieve_cross_repo_metadata()`:
  Calls list_all_repos() — no vector search, no embeddings.
  Returns all repo metadata summaries. Engine + Gemini do the filtering.

`retrieve_cross_repo_semantic(question)`:
  Embeds question → `similarity_search_per_repo()` — returns the single
  best-scoring chunk per repo. Every repo gets exactly one representative
  chunk so Gemini can assess all repos, not just those that dominated top-k.

`retrieve_cross_repo_comparative(question)`:
  Embeds question → `similarity_search_aggregated(candidate_k=50)`.
  Groups top-50 chunks by repo, scores each repo as avg of its top-3
  chunk scores, ranks repos, returns top-3 repos × top-3 chunks each.
  Fair comparison regardless of repo size or chunk count.

`retrieve_repo_specific(question, repo_name, history, seen_ids)`:
  Enriches query with last 2 turns, embeds, filters to repo,
  deduplicates, returns top-4 fresh chunks.

---

### `query/engine.py`
**Purpose:** The main orchestrator. The only module cli.py calls.
**Owns:** Session lifecycle, context window management, prompt building, answer generation.
**Does not own:** Classification (delegates to router), search (delegates to retriever).

How it works per turn:
  1. Call router to classify the question
  2. Call the appropriate retriever function
  3. Build a prompt from retrieved context + session history
  4. Call Gemini to generate the answer
  5. Update session history (repo_specific only)
  6. Return (answer, updated_session)

Session management:
- Session is a plain Python dict, lives in memory in cli.py
- Starts on first repo_specific question, resets on any cross-repo or list question
- Context window: after 4 turns (8 history entries), older turns are summarized
  into one paragraph by Gemini — prevents unbounded prompt growth
- Summarization failure is non-fatal: placeholder used so session continues

Five prompt builders — one per query type:
- `_build_metadata_prompt()` — list_repos + cross_repo_metadata (full metadata per repo)
- `_build_semantic_prompt()` — cross_repo_semantic (flat chunk list)
- `_build_comparative_prompt()` — cross_repo_comparative (structured by repo with rank/score)
- `_build_repo_specific_prompt()` — repo_specific (history + repo metadata + chunks)

---

### `cli.py`
**Purpose:** The user-facing interface. The entry point for running the tool.
**Owns:** Command parsing, the chat loop, user output formatting.
**Does not own:** Any query logic — delegates everything to engine.query().

How it works:
- Two subcommands: `index` (runs indexing pipeline) and `chat` (starts conversation)
- Chat loop: read input → call engine.query(question, session) → print answer → repeat
- Session is held in the `session` variable in the loop and passed back each turn
- Built-in commands: `reset` (clears session), `scope` (shows active repo), `exit`/`quit`
- `load_dotenv()` called before any imports that read env vars — import order matters

---

## 22. Implementation Notes

Key decisions made during implementation that any contributor should know:

### deeplake_store.py
- REPLACED: old per-row loop (N network calls, O(N) time)
- `_load_all()` loads all embeddings as numpy matrix in one call,
  then `M @ query_vector` computes all N cosine scores in one matrix multiply
- `similarity_search_per_repo()`: scores all chunks, keeps only the best-scoring
  chunk per repo — guarantees every repo is represented in cross_repo_semantic results
- `similarity_search_aggregated()`: scores all chunks, groups by repo, scores each
  repo as avg of its top-3 chunk scores — fair ranking for cross_repo_comparative
- `list_all_repos()` now surfaces all new metadata fields so the metadata prompt
  gives Gemini the full picture
- delete_repo_chunks() deletes in reverse index order to avoid index shifting bugs

### query/router.py
- Five types: added cross_repo_comparative alongside existing four
- cross_repo_semantic = find/not-find ("do any repos have X?")
- cross_repo_comparative = ranking ("which repo has the best X?")
- Fallback is cross_repo_semantic on any classification failure

### query/retriever.py
- retrieve_cross_repo_semantic() now uses similarity_search_per_repo() instead of
  global top-k — returns one chunk per repo, every repo covered, none crowded out
- retrieve_cross_repo_comparative() uses similarity_search_aggregated() — fair scoring
- TOP_K_CROSS_REPO_SEMANTIC constant removed — per-repo search doesn't use fixed k
- retrieve_cross_repo_metadata() unchanged — no embedding, no vector search

### query/engine.py
- Five branches in query() — cross_repo_comparative added
- _build_metadata_prompt() now formats all new metadata fields per repo line
  (has_authentication, database_type, key_features, external_services, etc.)
- _build_comparative_prompt() new — structures output by repo with rank and score
- Session resets on all cross-repo and list questions

### github_client.py
- get_indexable_files() accepts file_filter dict from metadata_generator
- "readme" mode: path prefix/exact match + extension gate
- "language_fallback" mode: extension-based with EXCLUDED_PATH_SEGMENTS
- INDEXABLE_EXTENSIONS constant removed

### metadata_generator.py
- Returns (metadata, file_filter) tuple
- Single Gemini call now extracts all fields including new structural ones
- New fields: has_authentication, has_database, database_type, has_api, api_style,
  has_frontend, frontend_framework, architecture_pattern, key_features,
  external_services, has_tests
- All new fields normalized defensively — wrong types → safe defaults, never crash
- Language fallback covers 17 languages + broad default
- .md always included via ALWAYS_INCLUDE_EXTENSIONS

### Gemini SDK
- All files use `google-genai` (new SDK: `from google import genai`)
  NOT `google-generativeai` (deprecated as of mid-2025)
- Pattern: `client = genai.Client(api_key=...)`
  `client.models.generate_content(model="gemini-1.5-flash", contents=prompt)`
  `client.models.embed_content(model=..., contents=text, config={"task_type": ...})`

### chunker.py
- 1 token ≈ 4 chars approximation — no tokenizer library needed
- chunk_index is per-file not global

### embedder.py
- RETRIEVAL_DOCUMENT at index time, RETRIEVAL_QUERY at query time — mismatching degrades quality
- 0.65s sleep per call stays under 100 req/min free tier

### cli.py
- load_dotenv() before any imports that read env vars
- reset/scope commands make no API calls

---

## 23. Progress Tracker

```
[x] github_client.py           — README-driven file filter, language fallback
[x] metadata_generator.py      — rich metadata extraction: 11 structural fields + file_filter
[x] chunker.py
[x] embedder.py
[x] deeplake_store.py          — batch search + per-repo search + aggregated scoring
[x] indexer/main.py            — passes file_filter from metadata into get_indexable_files
[x] query/router.py            — five types: list/metadata/semantic/comparative/repo_specific
[x] query/retriever.py         — four strategies: per-repo semantic, aggregated comparative
[x] query/engine.py            — five branches, rich metadata prompt, fair repo ranking
[x] cli.py
[ ] End-to-end test with real repos
```

---

*Last updated: 2026-06-12*
*Status: All modules implemented. Five query types. Rich metadata. Per-repo semantic search. Aggregated comparative scoring.*
