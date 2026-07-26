"""
api/server.py
-------------
FastAPI HTTP server wrapping query/engine.py.

Start with: uvicorn api.server:app --reload --port 8000

One in-memory session store (sessions dict). Single-user for Phase 2.
For multi-user: replace sessions dict with Redis — interface unchanged.

CORS is open (*) for Phase 2 local development.
Tighten in production.

Endpoints (INTEGRATION_PLAN.md Section 9):
  POST /ask     — main endpoint, calls query() and returns its result dict
  GET  /repos   — list_all_repos(), passed through as-is
  GET  /health  — status + indexed repo count
  POST /reset   — clear a session by id
"""

from dotenv import load_dotenv

# Load .env the same way cli.py does, before any import that reads env vars
# (query.engine constructs a GitHubClient(token=os.getenv("GITHUB_TOKEN"))
# at call time inside query(), so this must run before that first call —
# doing it here at import time, same as cli.py, covers that).
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

from query.engine import query
from indexer.deeplake_store import list_all_repos

app = FastAPI(title="GitHub Brain API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# In-memory session store.
# Key: session_id (str UUID)
# Value: session dict from engine.py (active_repo, conversation_history, etc.)
sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None

class ResetRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    repos = list_all_repos()
    return {"status": "ok", "indexed_repos": len(repos)}


@app.get("/repos")
def get_repos():
    # list_all_repos() already returns the exact shape the frontend needs
    # (repo_name, repo_description, repo_language, repo_technologies,
    # deployment_url, has_authentication, has_database, ...) — no
    # transformation needed, per plan Section 9.
    repos = list_all_repos()
    return {"repos": repos}


@app.post("/ask")
def ask(body: AskRequest):
    sid     = body.session_id or str(uuid4())
    session = sessions.get(sid)

    answer, updated_session, result = query(body.question, session=session)

    # Persist updated session.
    if updated_session is not None:
        sessions[sid] = updated_session
    elif sid in sessions:
        # Session was reset by the engine (cross-repo question after repo deep-dive,
        # or an error path in engine.py that intentionally returns session=None —
        # see e.g. the cross_repo_metadata "not indexed yet" branch).
        del sessions[sid]

    result["session_id"] = sid
    return result


@app.post("/reset")
def reset(body: ResetRequest):
    sid = body.session_id
    if sid in sessions:
        del sessions[sid]
    return {"session_id": sid, "cleared": True}
