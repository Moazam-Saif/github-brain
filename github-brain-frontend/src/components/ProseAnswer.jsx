// Renders one section's prose body with real formatting:
//   **bold**            -> <strong>
//   single newlines      -> <br /> (soft line breaks within a paragraph)
//   blank-line breaks     -> separate <p> blocks
//   "- item" / "* item"    -> a real <ul><li> list
//   [N]                     -> clickable marker jumping to that chunk
//   `code` or bare.dotted.identifiers -> clickable, jumps to the exact
//     line in the source file where that literal text is found (see
//     findCodeReference below)
// Gemini's responses were never told to avoid markdown, and up to now the
// frontend only handled [N] and "## " headings — everything else (bold,
// bullets, line breaks) was showing up as literal asterisks/dashes. This
// replaces that with real formatting throughout.

const MARKER_RE    = /\[(\d+)\]/g;
const BOLD_RE       = /\*\*(.+?)\*\*/g;
// Two ways a code reference can show up in prose:
//   1. Explicitly backtick-wrapped: `request.form.get`
//   2. A bare dotted-identifier chain with 2+ segments and at least one
//      lowercase/underscore segment (so it doesn't accidentally match
//      ordinary prose like "U.S." or a version number "3.11") — e.g.
//      request.form.get, db.session.commit, os.path.join
const BACKTICK_RE   = /`([^`]+)`/g;
const DOTTED_ID_RE  = /\b([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*){1,})\b/g;

/**
 * Search the given files for the first line containing `needle` as a
 * literal substring. Searches `preferredFile` first (the file the
 * currently-active/most-relevant chunk belongs to, if known), then falls
 * back to every other fetched file — so a reference to code in a
 * different file than the one currently open still resolves.
 *
 * Returns { filePath, lineNumber } (1-indexed) or null if not found in
 * any fetched file's content.
 */
function findCodeReference(needle, files, preferredFile) {
  if (!needle || needle.length < 3 || !files) return null;
  const fileOrder = preferredFile && files[preferredFile]
    ? [preferredFile, ...Object.keys(files).filter((k) => k !== preferredFile)]
    : Object.keys(files);

  for (const filePath of fileOrder) {
    const lines = files[filePath]?.lines || [];
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes(needle)) {
        return { filePath, lineNumber: i + 1 };
      }
    }
  }
  return null;
}

function renderInline(text, chunks, files, preferredFile, onJump, onJumpToLine, keyPrefix) {
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

  // Within a plain-text run (after [N] markers are pulled out), find
  // backtick-wrapped or bare dotted-identifier code references and turn
  // any that actually match somewhere in the fetched file content into a
  // clickable span. Non-matching candidates render as plain text (or, for
  // backticks, as inline code styling without the click behavior) rather
  // than silently guessing — a reference that doesn't resolve to a real
  // line is not clickable, per findCodeReference's null-return contract.
  function renderCodeRefs(str, keyBase) {
    const candidates = [];
    let bm;
    BACKTICK_RE.lastIndex = 0;
    while ((bm = BACKTICK_RE.exec(str)) !== null) {
      candidates.push({ s: bm.index, e: bm.index + bm[0].length, text: bm[1], backticked: true });
    }
    DOTTED_ID_RE.lastIndex = 0;
    let dm;
    while ((dm = DOTTED_ID_RE.exec(str)) !== null) {
      // Skip if this overlaps a backtick match already found.
      if (candidates.some((c) => dm.index < c.e && dm.index + dm[0].length > c.s)) continue;
      candidates.push({ s: dm.index, e: dm.index + dm[0].length, text: dm[1], backticked: false });
    }
    candidates.sort((a, b) => a.s - b.s);

    if (candidates.length === 0) return [str];

    const out = [];
    let pos = 0;
    let ck = 0;
    for (const c of candidates) {
      if (c.s > pos) out.push(str.slice(pos, c.s));
      const target = files ? findCodeReference(c.text, files, preferredFile) : null;
      if (target) {
        out.push(
          <button
            key={`${keyBase}-code-${ck++}`}
            onClick={() => onJumpToLine(target.filePath, target.lineNumber)}
            className="font-mono text-[0.85em] bg-purple/10 text-purple px-1 py-0.5 rounded hover:bg-purple/20 transition-colors"
            title={`Jump to ${target.filePath}:${target.lineNumber}`}
          >
            {c.text}
          </button>
        );
      } else if (c.backticked) {
        // Backticked but no match found in fetched files — still render as
        // inline code styling (that's what the backticks signaled), just
        // not clickable.
        out.push(
          <code key={`${keyBase}-code-${ck++}`} className="font-mono text-[0.85em] bg-black/5 px-1 py-0.5 rounded">
            {c.text}
          </code>
        );
      } else {
        out.push(c.text);
      }
      pos = c.e;
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
      if (mm.index > pos) segments.push(...renderCodeRefs(part.text.slice(pos, mm.index), `${keyPrefix}-p${pi}-${pos}`));
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
    if (pos < part.text.length) segments.push(...renderCodeRefs(part.text.slice(pos), `${keyPrefix}-p${pi}-tail`));

    return part.bold ? (
      <strong key={`${keyPrefix}-bold-${pi}`} className="font-bold">
        {segments}
      </strong>
    ) : (
      <span key={`${keyPrefix}-plain-${pi}`}>{segments}</span>
    );
  });
}

function renderBody(body, chunks, files, preferredFile, onJump, onJumpToLine) {
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
                {renderInline(line.replace(BULLET_RE, ''), chunks, files, preferredFile, onJump, onJumpToLine, `${key}-l${li}`)}
              </li>
            ))}
          </ul>
        );
      }
      return (
        <p key={key} className="font-reading text-[0.98rem] leading-[1.75] text-[#241f3d] px-2">
          {run.lines.map((line, li) => (
            <span key={li}>
              {renderInline(line, chunks, files, preferredFile, onJump, onJumpToLine, `${key}-l${li}`)}
              {li < run.lines.length - 1 && <br />}
            </span>
          ))}
        </p>
      );
    });
  });
}

export default function ProseAnswer({ body, chunks, files, preferredFile, onJumpToChunk, onJumpToLine }) {
  return (
    <div className="flex-1 overflow-y-auto px-2.5 py-3 flex flex-col gap-2">
      {renderBody(
        body || '',
        chunks,
        files,
        preferredFile,
        onJumpToChunk || (() => {}),
        onJumpToLine || (() => {})
      )}
    </div>
  );
}
