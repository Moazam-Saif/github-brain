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
| Embeddings | Gemini `text-embedding-004` (Vertex AI) | Free tier: 100 req/min |
| LLM | Gemini `gemini-2.5-flash` (Vertex AI) | Free tier: 15 req/min |
| Metadata extraction | Same Gemini model | Same model, one call per repo |
| Vector store | Activeloop Deep Lake (`hub://ORG/github_brain_v5`) | Free tier, cloud-hosted, metadata filtering |
| Retrieval | **Hybrid Search: BM25 + cosine via RRF** | Fixes pure-cosine failure mode (see Section 22) |
| GitHub data | GitHub REST API | 5000 req/hr with personal token |
| Backend | FastAPI (Python) | Lightweight, async, easy to extend later |
| Interface | CLI (Phase 1), React + TypeScript (Phase 2) | Validate core before building UI |

### Authentication: Vertex AI service account, not API key

`gemini_client.py` authenticates via a **GCP service account JSON**, not a Gemini API key.
The full service account JSON is stored as a single-line string in the
`GCP_SERVICE_ACCOUNT_JSON` environment variable. `get_client()` parses it, builds
OAuth2 credentials, and returns `genai.Client(vertexai=True, project=..., location=...,
credentials=...)`.

`GEMINI_MODEL` is read from env (`GEMINI_MODEL`, default `gemini-2.5-flash`).
`EMBEDDING_MODEL` is `text-embedding-004` (Vertex AI path — no `models/` prefix).

### Dataset naming history

The dataset went through several names due to an Activeloop constraint:
**once a dataset is deleted from Deep Lake Cloud, that path can never be reused.**
`github_brain` and `github_brain_v2`/`v3`/`v4` were burned this way during setup.
The current and stable path is **`hub://ORG/github_brain_v5`**, hardcoded in
`deeplake_store.py`'s `_get_dataset_path()`. Do not delete this dataset — see
Section 10 for how re-indexing works without ever deleting the dataset.

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
  "file_role":           "api",
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
| `file_role` | Gemini (README) or inferred from filename/folder | Controlled vocabulary — see Section 8a |
| `chunk_index` | System | Position within file (per-file, starts at 0) |
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
including the richer structural fields, plus per-file role/purpose mappings,
in a single Gemini call per repo.

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
file_roles            {file_path: role}    — only when files_to_index.found
file_purposes         {file_path: purpose} — only when files_to_index.found
```

All returned as a single JSON object. Parsed and validated defensively —
wrong types are normalized to safe defaults rather than crashing. Roles not
in the controlled vocabulary (`VALID_FILE_ROLES`) are coerced to `"other"`.

---

## 8. Files to Index Per Repo

### How files are selected

File selection is README-driven, not hardcoded. The system has two modes:

**Mode 1 — README-driven (preferred)**
`files_to_index.found` is set to `true` ONLY when the README has a dedicated
section whose heading signals an intentionally curated list of important files —
trigger phrases: "Essential Files", "Key Files", "Read these files", "Only the
files that matter", "Start here", "Important files", or similar.

A generic "Project Structure" section or full directory tree listing ALL files
is **NOT** a valid trigger on its own.

**Priority rule:** if the README contains BOTH a full project structure tree AND
a separately named curated section, only the curated section is used — the full
tree is ignored entirely. This was added after CorpLaw-AI's README (which has both)
produced inconsistent results (9 files one run, 31 the next) before this rule existed.

Exclusion blocks ("Not worth looking at", "Ignore these", "Skip these") are honored —
files mentioned there are never included even if they appear elsewhere.

Only files whose paths match the extracted curated list are fetched and indexed.

**Mode 2 — Language fallback**
If the README has no curated file list (or there is no README), the system falls
back to a language-appropriate extension set derived from the GitHub API `language` field.

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

---

## 8a. File Roles, Purposes, and Context Headers

### The problem this solves

Cosine similarity ranks chunks by semantic closeness to the question, but the
most important chunks aren't always the closest in embedding space. A dense
configuration file (e.g. `lib/auth.ts` containing NextAuth provider setup) doesn't
*sound like* "how does authentication work?" even though it directly answers it.
Files that merely *talk about* a topic in passing can outscore the file that
actually *implements* it.

### The fix: embed each chunk's purpose, not just its code

Before chunking, `chunker.py` prepends a one-line context header to every file's
content:

```
File: lib/auth.ts | Role: authentication | Purpose: Google OAuth setup and NextAuth adapter configuration
```

This header becomes part of chunk 0's embedded text, pulling that chunk's vector
toward "authentication", "OAuth", "NextAuth" in semantic space.

### Why context headers alone are not sufficient

In practice (confirmed with CorpLaw-AI), the context header improves embedding
quality but does not guarantee the auth file wins the cosine similarity ranking.
Files with high volumes of semantically adjacent content (e.g. session management
code that co-occurs with auth in the embedding model's training data) can still
outscore the actual auth file. This is why hybrid search (Section 22) was added —
the two techniques complement each other.

### Role/purpose resolution order

1. **README-extracted** (`file_roles` / `file_purposes` from `metadata_generator.py`) —
   only available for files in the README's curated "important files" list (Section 8,
   Mode 1). Most accurate — comes from the repo author's own description.
2. **Filename/folder inference** (`_infer_role_from_path` in `chunker.py`) — fallback
   for every file not covered by README extraction, including ALL files in repos with
   no curated list (Mode 2 / language fallback).
3. **No match** — header is just `File: <path>`, no Role/Purpose.

Purpose is only ever included if it came from the README — inferred roles never
carry a purpose string, since there's nothing reliable to say beyond the role itself.

### Controlled vocabulary (`VALID_FILE_ROLES`)

```
authentication, database, routing, api, middleware, state management,
configuration, testing, queue/background jobs, file storage, search/retrieval,
realtime, notifications, caching, payments, logging, utilities,
ui component, ui page, styling, ai/llm, api client, other
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

Step 3: Fetch and filter files
  Apply include/exclude rules and 50KB size limit
  Fetch file content for each included file via GitHub API

Step 4: Chunk files
  Prepend a one-line context header to each file's content first:
    "File: <path> | Role: <role> | Purpose: <purpose>" (purpose only if README-sourced)
  Split header+content into chunks of 512 tokens
  Overlap consecutive chunks by 50 tokens
  Attach full repo metadata + file_path + file_type + file_role + chunk_index to each chunk

Step 5: Embed chunks
  Send chunk text to Gemini text-embedding-004
  Batch requests — stay within 100 req/min rate limit
  Add delay between batches if needed

Step 6: Store in Deep Lake
  One dataset for the entire GitHub account: hub://ORG/github_brain_v5
  Store: embedding vector + text + metadata JSON
  Always appends — never overwrites existing rows directly
  For re-indexing a specific repo: delete_repo_chunks() rewrites the dataset
  in place (load all → filter out target repo → reset dataset content via
  deeplake.empty(overwrite=True) → re-append kept rows), then new chunks
  for the repo are appended.
```

---

## 10. Deep Lake Dataset Structure

```
Dataset: hub://ORG/github_brain_v5

Tensors:
  embedding   → float32 (768-dim, Gemini text-embedding-004 output size)
  text        → str     (the raw chunk content, chunk 0 prefixed with context header)
  metadata    → json    (full metadata object per chunk, includes file_role)
```

### Dataset creation (`get_or_create_dataset()`)

```python
ds = deeplake.dataset(path, token=token)   # loads if exists, creates empty if not
if "embedding" not in ds.tensors:
    ds.create_tensor("embedding", htype="embedding", dtype="float32", sample_compression=None)
    ds.create_tensor("text", htype="text")
    ds.create_tensor("metadata", htype="json")
```

### Deletion strategy: rewrite-in-place, never `ds.pop()` or `deeplake.delete()`

**`ds.pop()` is broken in deeplake 3.x** — raises `OverflowError: can't convert
negative int to unsigned` inside Deep Lake's own commit-diff tracking.

**`deeplake.delete()` + recreate is also unworkable** — path can never be reused.

**The actual approach** (`delete_repo_chunks()` in `deeplake_store.py`):
load all → filter out target repo → reset via `deeplake.empty(overwrite=True)` →
recreate tensors → re-append kept rows.

---

## 11. The Five Query Types

Every question is classified into one of five types before any search happens.

**list_repos** → list_all_repos() from metadata, no vector search
**cross_repo_metadata** → list_all_repos(), Gemini filters, no vector search
**cross_repo_semantic** → similarity_search_per_repo() — one chunk per repo
**cross_repo_comparative** → similarity_search_aggregated() — fair repo ranking
**repo_specific** → hybrid_search() within one repo + session history

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

Step 3a/3b: list_repos / cross_repo_metadata
  list_all_repos() → all repo metadata summaries → Gemini answers
  No vector search. No session.

Step 3c: cross_repo_semantic
  embed_query(question) → similarity_search_per_repo() → one chunk per repo
  Gemini answers find/not-find. No session.

Step 3d: cross_repo_comparative
  embed_query(question) → similarity_search_aggregated(candidate_k=50)
  → group by repo → top-3 avg score → rank → Gemini compares. No session.

Step 3e: repo_specific
  Normalize repo name case-insensitively
  Start or continue session
  _manage_context_window() if history is long
  Build enriched query (question + last 2 turns)
  embed_query(enriched) + hybrid_search(enriched, repo_name=X, top_k=20)
  → deduplicate → return top 5 fresh chunks
  Gemini answers with history + code context
  Append turn to session.
```

---

## 13. Session Management (Repo Deep Dives)

### Why sessions are needed

A deep dive is a conversation, not a single query. Without session state,
every question is answered in isolation.

### Session object

```python
session = {
    "active_repo":         "claimsense",
    "conversation_history": [
        {"role": "user",      "content": "how does the PDF parser work?"},
        {"role": "assistant", "content": "The parser uses pdfplumber to..."}
    ],
    "seen_chunk_ids":      set(),
    "repo_metadata":       {...}
}
```

### Conversation-aware retrieval

The enriched query is used for BOTH the cosine embedding and BM25 tokenization
in hybrid_search — enrichment adds context words that help BM25 find relevant
follow-up chunks too, not just the cosine component.

### Session lifecycle

```
Session starts    → user asks about a specific repo (router returns repo_specific)
Session continues → every follow-up that stays within the same repo
Session resets    → user asks cross_repo_metadata, cross_repo_semantic, or list_repos
                  → user explicitly names a different repo
                  → user types "reset" in the CLI
```

---

## 14. Query Router Prompt

See `query/router.py` for the full prompt. Five types:

```
list_repos          → user wants to list, count, or browse repos
cross_repo_metadata → technology, language, deployment answerable from metadata
cross_repo_semantic → concept/pattern requiring actual code (find/not-find)
cross_repo_comparative → ranks or compares repos on some quality (needs code)
repo_specific       → targets one repo by name or active session

Fallback: cross_repo_semantic
```

---

## 15. File Structure

```
github-brain/
│
├── indexer/
│   ├── main.py
│   ├── github_client.py
│   ├── metadata_generator.py
│   ├── chunker.py
│   ├── embedder.py
│   └── deeplake_store.py        ← hybrid_search() added
│
├── query/
│   ├── engine.py
│   ├── router.py
│   └── retriever.py             ← uses hybrid_search for repo_specific
│
├── cli.py
├── .env
├── .env.example
├── requirements.txt             ← rank-bm25 added
└── BLUEPRINT.md
```

---

## 16. Environment Variables

```
GITHUB_TOKEN=
ACTIVELOOP_TOKEN=
ACTIVELOOP_ORG=
GCP_SERVICE_ACCOUNT_JSON=
GOOGLE_CLOUD_LOCATION=     # defaults to us-central1
GEMINI_MODEL=              # optional, defaults to gemini-2.5-flash
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
| BM25 index | Built in-memory per query | Negligible cost for <10k chunks |

---

## 19. What Is Not In Scope (Phase 1)

- Private repos
- Forked repos
- Archived repos
- Automatic re-indexing / webhooks / cron jobs
- React frontend (Phase 2)
- Authentication / multi-user support
- Docker metadata fields (is_dockerized, has_ci_cd)
- Qodo-Embed model (code-specific embeddings — better long-term, requires full re-index)

---

## 20. Build Order

```
1. github_client.py
2. metadata_generator.py
3. chunker.py
4. embedder.py
5. deeplake_store.py          ← hybrid_search() is the primary search function
6. indexer/main.py
7. query/router.py
8. query/retriever.py         ← retrieve_repo_specific uses hybrid_search
9. query/engine.py
10. cli.py
```

---

## 21. File Purposes and How Each File Works

### `indexer/deeplake_store.py`
**Primary search function: `hybrid_search()`** — BM25 + cosine similarity fused
via Reciprocal Rank Fusion (RRF). Used by `retrieve_repo_specific()`.

`similarity_search()` is kept as a cosine-only fallback used internally by
`similarity_search_aggregated()` and `similarity_search_per_repo()`. These
cross-repo functions don't need hybrid search — their fairness problems are
solved at the aggregation/per-repo level, not at the term-matching level.

`_build_bm25_index(texts)` builds a BM25Okapi index from a list of chunk texts
using simple whitespace tokenization. Called fresh on every hybrid_search call —
fast enough for <10k chunks, avoids stale index issues.

### `query/retriever.py`
`retrieve_repo_specific()` now calls `hybrid_search()` instead of `similarity_search()`.
Fetches `TOP_K_HYBRID_FETCH = 20` raw candidates (up from 10) before deduplication,
because hybrid scoring changes rank ordering significantly versus pure cosine —
more headroom ensures the deduplication pass always has 5 fresh chunks to return.

`retrieve_cross_repo_semantic()` and `retrieve_cross_repo_comparative()` are
unchanged — they use `similarity_search_per_repo()` and `similarity_search_aggregated()`
respectively, which remain cosine-only.

The enriched query (question + last 2 turns) is passed as `query_text` to
`hybrid_search()` — this means BM25 also benefits from conversation context,
not just the cosine embedding component.

---

## 22. Implementation Notes

### Why Hybrid Search was added (the lib/auth.ts problem)

**The failure mode:** "how does authentication work in CorpLaw-AI" returned
`app/chat/[sessionId]/page.tsx` and `app/api/chat/route.ts` instead of `lib/auth.ts`.

**Root cause:** Pure cosine similarity has a ~44-63% benchmark accuracy ceiling.
`lib/auth.ts` had the correct context header (`Role: authentication`) and the
correct embedding, but session/chat files contained enough semantically adjacent
vocabulary (`session`, `sessionId`, `fetchSession`) to outscore it in cosine space.
The embedding model learned that session management and authentication co-occur,
so those files ranked higher despite not being the auth source.

**Why context headers alone weren't enough:** The header improves the embedding
signal but cannot overcome a volume imbalance — CorpLaw-AI had 18 chunks of
session/chat code vs 2 chunks of auth code. The session files dominated the
cosine ranking by sheer presence in the vector neighborhood.

**Why the old keyword-to-file layer was not revived:** An earlier iteration
reserved 2 of 5 chunk slots for files whose *path* matched keyword hint strings.
This was removed because path substring matching produced false positives
(e.g. "session" in the auth keyword group matched `app/chat/[sessionId]/page.tsx`
instead of `lib/auth.ts`) and it only handled black-and-white questions cleanly.

**The hybrid search fix:** BM25 scores `lib/auth.ts` highly because "auth" appears
literally in the filename and code — it's a lexical exact match. Cosine scores
the session files highly semantically. RRF fuses both rankings:

```
RRF_score(chunk) = 1/(60 + cosine_rank) + 1/(60 + bm25_rank)
```

A chunk that ranks well on BOTH methods (like `lib/auth.ts` — good BM25 on "auth",
decent cosine on authentication concept) scores higher than a chunk that only
wins on one dimension (the session files — great cosine, poor BM25 on "auth").

**Why RRF over weighted score combination:** Raw BM25 and cosine scores are on
incompatible scales (BM25 is unbounded, cosine is -1 to 1). RRF normalizes by
rank rather than raw score, making the fusion stable without needing to tune a
λ weighting parameter.

**Scope:** Hybrid search is applied only to `retrieve_repo_specific()` — the path
where this failure mode occurs. Cross-repo functions use their own fairness
mechanisms (per-repo best, aggregated scoring) that already address their
respective failure modes.

### deeplake_store.py
- `hybrid_search()` is the new primary search function for repo-specific queries
- `similarity_search()` retained for internal use by aggregated/per-repo functions
- `_build_bm25_index()` uses `BM25Okapi` from `rank-bm25` package
- BM25 index is built in-memory per query call — no persistence needed at this scale
- All other store functions (store_chunks, delete_repo_chunks, list_all_repos) unchanged

### query/retriever.py
- `retrieve_repo_specific()` now imports and calls `hybrid_search()`
- `TOP_K_HYBRID_FETCH = 20` (was `TOP_K_REPO_SPECIFIC * 2 = 10`) — wider fetch
  because hybrid reranking changes the order significantly
- Enriched query string passed as both embedding input and BM25 query text
- Cross-repo retrieval functions unchanged

### Future embedding upgrade path
The current Gemini `text-embedding-004` is a general-purpose model not optimized
for code retrieval. Qodo-Embed-1-1.5B (CoIR score 68.53 vs OpenAI 65.17) is
purpose-built for natural language → code retrieval and would improve the cosine
component of hybrid search. Migration requires:
1. New dataset path (v6) — embedding dimensions change from 768 to 1536
2. Full re-index of all repos with Qodo embedder
3. Update `embedder.py` to use SentenceTransformers instead of Gemini API
4. BM25 component carries over unchanged — no re-indexing needed for that

---

## 23. Progress Tracker

```
[x] gemini_client.py
[x] github_client.py
[x] metadata_generator.py
[x] chunker.py                    — context headers (Role/Purpose)
[x] embedder.py
[x] deeplake_store.py             — hybrid_search() (BM25 + cosine + RRF)
[x] indexer/main.py
[x] query/router.py               — five query types
[x] query/retriever.py            — retrieve_repo_specific uses hybrid_search
[x] query/engine.py
[x] cli.py
[x] End-to-end test (CorpLaw-AI)  — indexing + chat working
[x] Hybrid search implemented     — fixes lib/auth.ts retrieval failure
[ ] Full index of all 21 repos
[ ] Verify hybrid search on CorpLaw-AI auth question
[ ] Future: Qodo-Embed-1-1.5B migration for better code-specific cosine
```

---

*Last updated: 2026-06-15*
*Status: Hybrid search (BM25 + cosine + RRF) implemented in deeplake_store.py and
retriever.py. Fixes the pure-cosine failure mode where lib/auth.ts lost to
session/chat files on "how does authentication work". Next: install rank-bm25,
test the fix on CorpLaw-AI, then full index of remaining 21 repos.*
