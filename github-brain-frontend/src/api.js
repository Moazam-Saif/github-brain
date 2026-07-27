/**
 * api.js
 * ------
 * Thin fetch wrappers around api/server.py's endpoints.
 * See INTEGRATION_PLAN.md Section 9 for the full request/response contract.
 */

const API_BASE = 'http://localhost:8000';

export async function fetchRepos() {
  const res = await fetch(`${API_BASE}/repos`);
  if (!res.ok) throw new Error(`GET /repos failed: HTTP ${res.status}`);
  const data = await res.json();
  return data.repos;
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`GET /health failed: HTTP ${res.status}`);
  return res.json();
}

/**
 * Ask a question. Returns the full result dict from api/server.py's /ask
 * (query_type, repo, answer, summary, chunks, files, repo_metadata,
 * session_id, and optionally error).
 */
export async function askQuestion(question, sessionId) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      session_id: sessionId || undefined,
    }),
  });
  if (!res.ok) throw new Error(`POST /ask failed: HTTP ${res.status}`);
  return res.json();
}

export async function resetSession(sessionId) {
  if (!sessionId) return null;
  const res = await fetch(`${API_BASE}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(`POST /reset failed: HTTP ${res.status}`);
  return res.json();
}
