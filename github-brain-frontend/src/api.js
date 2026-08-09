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

/**
 * Streaming variant of askQuestion, consuming api/server.py's /ask/stream
 * SSE endpoint. Can't use EventSource (it can't send a POST body), so this
 * reads the response body as a stream and parses SSE frames by hand.
 *
 * onEvent(eventName, data) is called for each event as it arrives:
 *   'summary'       { summary: str }
 *   'sections'      { sections: [{heading}, ...] }
 *   'answer'        { answer: str }
 *   'chunk_blocks'  { chunk_blocks: {num: {heading, text}} }
 *   'done'          the full result dict (same shape askQuestion() returns)
 *   'error'         { message: str }
 *
 * Returns a promise that resolves once the stream ends (after 'done' or
 * 'error'). Callers generally only need the onEvent callback, not the
 * return value, but it resolves to undefined either way.
 */
export async function askQuestionStream(question, sessionId, onEvent) {
  const res = await fetch(`${API_BASE}/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      session_id: sessionId || undefined,
    }),
  });
  if (!res.ok || !res.body) {
    throw new Error(`POST /ask/stream failed: HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line ("\n\n").
    let frameEnd;
    while ((frameEnd = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, frameEnd);
      buffer = buffer.slice(frameEnd + 2);

      let eventName = 'message';
      let dataLine = '';
      for (const line of frame.split('\n')) {
        if (line.startsWith('event: ')) eventName = line.slice(7);
        else if (line.startsWith('data: ')) dataLine = line.slice(6);
      }
      if (!dataLine) continue;

      let data;
      try {
        data = JSON.parse(dataLine);
      } catch {
        continue; // malformed frame — skip rather than crash the whole stream
      }
      onEvent(eventName, data);
    }
  }
}
