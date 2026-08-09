import { slugify } from './SectionNav';

// Renders prose, converting inline [N] chunk references (added by
// engine.py's RESPONSE_STRUCTURE_INSTRUCTION) into clickable markers that
// jump to that chunk in the ChunkList/SourceViewer, and rendering "## "
// lines as real headings with an id matching SectionNav's slugify() so
// clicking a section nav pill scrolls to the right spot.
const MARKER_RE = /\[(\d+)\]/g;

function renderWithMarkers(text, chunks, onJump) {
  const chunkByIndex = new Map((chunks || []).map((c) => [c.index, c]));
  const parts = [];
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = MARKER_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const num = parseInt(match[1], 10);
    const chunk = chunkByIndex.get(num);
    if (chunk) {
      parts.push(
        <button
          key={`marker-${key++}`}
          onClick={() => onJump(chunk)}
          className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-purple/15 text-purple text-[0.6rem] font-bold mx-0.5 align-middle hover:bg-purple/25"
        >
          {num}
        </button>
      );
    } else {
      parts.push(match[0]);
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts;
}

export default function ProseAnswer({ answer, chunks, onJumpToChunk }) {
  const blocks = (answer || '').split('\n\n').filter((p) => p.trim());

  return (
    <div className="flex-1 overflow-y-auto px-2.5 py-3 flex flex-col gap-1">
      {blocks.map((block, i) => {
        const headingMatch = block.match(/^##\s+(.+)$/m);
        if (headingMatch) {
          const heading = headingMatch[1].trim();
          const rest = block.replace(/^##\s+.+$/m, '').trim();
          return (
            <div key={i}>
              <h4
                id={`section-${slugify(heading)}`}
                className="text-[0.78rem] font-bold text-charcoal px-2 pt-3 pb-1 scroll-mt-4"
              >
                {heading}
              </h4>
              {rest && (
                <p className="text-[0.85rem] leading-[1.7] text-charcoal px-2">
                  {chunks && chunks.length > 0
                    ? renderWithMarkers(rest, chunks, onJumpToChunk || (() => {}))
                    : rest}
                </p>
              )}
            </div>
          );
        }
        return (
          <p key={i} className="text-[0.85rem] leading-[1.7] text-charcoal p-2">
            {chunks && chunks.length > 0
              ? renderWithMarkers(block, chunks, onJumpToChunk || (() => {}))
              : block}
          </p>
        );
      })}
    </div>
  );
}
