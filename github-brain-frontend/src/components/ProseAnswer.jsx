// Renders one section's prose body with real formatting:
//   **bold**            -> <strong>
//   single newlines      -> <br /> (soft line breaks within a paragraph)
//   blank-line breaks     -> separate <p> blocks
//   "- item" / "* item"    -> a real <ul><li> list
//   [N]                     -> clickable marker jumping to that chunk
//   {RN}                     -> clickable, jumps to the EXACT file+line the
//     backend resolved and validated for this reference (see engine.py's
//     RESPONSE_STRUCTURE_INSTRUCTION ---REFERENCES--- section and
//     _build_result's hallucination-guard). Replaces an earlier frontend-
//     only approach that regex-guessed at "code-looking" substrings and
//     searched fetched file content for a literal match — that produced
//     poor, arbitrary-looking highlights (see the reference screenshot in
//     INTEGRATION_PROGRESS.md) because it had no idea what the model
//     actually meant to cite. Now the model itself states exactly which
//     chunk and line it's citing, and the backend validates that claim
//     against the chunk's real line range before ever sending it to the
//     frontend — the frontend only renders links it's been explicitly
//     handed, never a guess.

const MARKER_RE = /\[(\d+)\]/g;
const BOLD_RE    = /\*\*(.+?)\*\*/g;
const REF_RE     = /\{R(\d+)\}/g;

function renderInline(text, chunks, references, onJump, onJumpToLine, keyPrefix) {
  // First split out **bold** spans, then run marker-detection over each
  // resulting plain-text piece (bold spans can't contain their own [N]
  // marker parsing without more bookkeeping than this simple case needs —
  // in practice Gemini doesn't bold a chunk reference).
  const boldParts = [];
  let lastIndex = 0;
  let m;
  let key = 0;
  BOLD_RE.lastIndex = 0;
  while ((m = BOLD_RE.exec(text)) !== null) {
    if (m.index > lastIndex) boldParts.push({ bold: false, text: text.slice(lastIndex, m.index) });
    boldParts.push({ bold: true, text: m[1] });
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < text.length) boldParts.push({ bold: false, text: text.slice(lastIndex) });
  if (boldParts.length === 0) boldParts.push({ bold: false, text });

  const chunkByIndex = new Map((chunks || []).map((c) => [c.index, c]));

  // A {RN} marker is only ever rendered as a link if `references[N]`
  // actually exists — the backend already dropped anything it couldn't
  // validate (wrong chunk, hallucinated line), so a missing entry here
  // just means "the model tried to cite something that didn't check out."
  // Rendered as plain text in that case, same as an unresolved [N].
  function renderRefsAndText(str, keyBase) {
    const out = [];
    let pos = 0;
    let rm;
    let rk = 0;
    REF_RE.lastIndex = 0;
    while ((rm = REF_RE.exec(str)) !== null) {
      if (rm.index > pos) out.push(str.slice(pos, rm.index));
      const refNum = parseInt(rm[1], 10);
      const ref = references ? references[refNum] : null;
      if (ref) {
        out.push(
          <button
            key={`${keyBase}-ref-${rk++}`}
            onClick={() => onJumpToLine(ref.file_path, ref.line)}
            className="font-mono text-[0.85em] bg-purple/10 text-purple px-1 py-0.5 rounded hover:bg-purple/20 transition-colors"
            title={`Jump to ${ref.file_path}:${ref.line}`}
          >
            {ref.expression}
          </button>
        );
      }
      // No fallback text when a {RN} doesn't resolve — the marker itself
      // is scaffolding syntax, not something a reader should ever see; an
      // unresolved reference just silently contributes nothing rather than
      // leaking "{R3}" into the rendered prose.
      pos = rm.index + rm[0].length;
    }
    if (pos < str.length) out.push(str.slice(pos));
    return out;
  }

  return boldParts.map((part, pi) => {
    const segments = [];
    let pos = 0;
    let mm;
    MARKER_RE.lastIndex = 0;
    while ((mm = MARKER_RE.exec(part.text)) !== null) {
      if (mm.index > pos) segments.push(...renderRefsAndText(part.text.slice(pos, mm.index), `${keyPrefix}-p${pi}-${pos}`));
      const num = parseInt(mm[1], 10);
      const chunk = chunkByIndex.get(num);
      if (chunk) {
        segments.push(
          <button
            key={`${keyPrefix}-marker-${key++}`}
            onClick={() => onJump(chunk)}
            className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-purple/15 text-purple text-[0.6rem] font-bold mx-0.5 align-middle hover:bg-purple/25"
          >
            {num}
          </button>
        );
      } else {
        segments.push(mm[0]);
      }
      pos = mm.index + mm[0].length;
    }
    if (pos < part.text.length) segments.push(...renderRefsAndText(part.text.slice(pos), `${keyPrefix}-p${pi}-tail`));

    return part.bold ? (
      <strong key={`${keyPrefix}-bold-${pi}`} className="font-bold">
        {segments}
      </strong>
    ) : (
      <span key={`${keyPrefix}-plain-${pi}`}>{segments}</span>
    );
  });
}

function renderBody(body, chunks, references, onJump, onJumpToLine) {
  // Split into paragraph-level blocks on blank lines, then within each
  // block detect a bullet list ("- " / "* " prefixed lines). A block can
  // be a lead-in sentence followed by bullets (e.g. "Key features:\n- a\n- b")
  // — only the bullet-prefixed lines become <li>s; a non-bullet line before
  // them renders as its own paragraph first.
  const blocks = body.split(/\n\s*\n/).filter((b) => b.trim());
  const BULLET_RE = /^[-*]\s+/;

  return blocks.flatMap((block, bi) => {
    const lines = block.split('\n').map((l) => l.trim()).filter(Boolean);

    // Split the block into runs of consecutive bullet / non-bullet lines,
    // preserving order, so "lead-in sentence" + "- item" + "- item" becomes
    // [paragraph, list] instead of forcing the whole block one way or the
    // other.
    const runs = [];
    for (const line of lines) {
      const isBullet = BULLET_RE.test(line);
      const last = runs[runs.length - 1];
      if (last && last.isBullet === isBullet) {
        last.lines.push(line);
      } else {
        runs.push({ isBullet, lines: [line] });
      }
    }

    return runs.map((run, ri) => {
      const key = `b${bi}-r${ri}`;
      if (run.isBullet) {
        return (
          <ul key={key} className="list-disc pl-5 font-reading text-[0.98rem] leading-[1.75] text-[#241f3d] px-2 space-y-1">
            {run.lines.map((line, li) => (
              <li key={li}>
                {renderInline(line.replace(BULLET_RE, ''), chunks, references, onJump, onJumpToLine, `${key}-l${li}`)}
              </li>
            ))}
          </ul>
        );
      }
      return (
        <p key={key} className="font-reading text-[0.98rem] leading-[1.75] text-[#241f3d] px-2">
          {run.lines.map((line, li) => (
            <span key={li}>
              {renderInline(line, chunks, references, onJump, onJumpToLine, `${key}-l${li}`)}
              {li < run.lines.length - 1 && <br />}
            </span>
          ))}
        </p>
      );
    });
  });
}

export default function ProseAnswer({ body, chunks, references, onJumpToChunk, onJumpToLine }) {
  return (
    <div className="flex-1 overflow-y-auto px-2.5 py-3 flex flex-col gap-2">
      {renderBody(
        body || '',
        chunks,
        references,
        onJumpToChunk || (() => {}),
        onJumpToLine || (() => {})
      )}
    </div>
  );
}
