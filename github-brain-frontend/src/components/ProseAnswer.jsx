// Renders one section's prose body with real formatting:
//   **bold**            -> <strong>
//   single newlines      -> <br /> (soft line breaks within a paragraph)
//   blank-line breaks     -> separate <p> blocks
//   "- item" / "* item"    -> a real <ul><li> list
//   [N]                     -> clickable marker jumping to that chunk
// Gemini's responses were never told to avoid markdown, and up to now the
// frontend only handled [N] and "## " headings — everything else (bold,
// bullets, line breaks) was showing up as literal asterisks/dashes. This
// replaces that with real formatting throughout.

const MARKER_RE = /\[(\d+)\]/g;
const BOLD_RE   = /\*\*(.+?)\*\*/g;

function renderInline(text, chunks, onJump, keyPrefix) {
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

  return boldParts.map((part, pi) => {
    const segments = [];
    let pos = 0;
    let mm;
    MARKER_RE.lastIndex = 0;
    while ((mm = MARKER_RE.exec(part.text)) !== null) {
      if (mm.index > pos) segments.push(part.text.slice(pos, mm.index));
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
    if (pos < part.text.length) segments.push(part.text.slice(pos));

    return part.bold ? (
      <strong key={`${keyPrefix}-bold-${pi}`} className="font-bold">
        {segments}
      </strong>
    ) : (
      <span key={`${keyPrefix}-plain-${pi}`}>{segments}</span>
    );
  });
}

function renderBody(body, chunks, onJump) {
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
                {renderInline(line.replace(BULLET_RE, ''), chunks, onJump, `${key}-l${li}`)}
              </li>
            ))}
          </ul>
        );
      }
      return (
        <p key={key} className="font-reading text-[0.98rem] leading-[1.75] text-[#241f3d] px-2">
          {run.lines.map((line, li) => (
            <span key={li}>
              {renderInline(line, chunks, onJump, `${key}-l${li}`)}
              {li < run.lines.length - 1 && <br />}
            </span>
          ))}
        </p>
      );
    });
  });
}

export default function ProseAnswer({ body, chunks, onJumpToChunk }) {
  return (
    <div className="flex-1 overflow-y-auto px-2.5 py-3 flex flex-col gap-2">
      {renderBody(body || '', chunks, onJumpToChunk || (() => {}))}
    </div>
  );
}
