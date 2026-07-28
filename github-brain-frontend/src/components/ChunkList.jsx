const COL_STYLES = {
  a: { num: 'bg-purple/15 text-purple', hl: 'bg-hl-a-bg border-hl-a-bd' },
  b: { num: 'bg-muted-teal/25 text-[#3a7a6d]', hl: 'bg-hl-b-bg border-hl-b-bd' },
  c: { num: 'bg-bright-green/25 text-[#2e7a38]', hl: 'bg-hl-c-bg border-hl-c-bd' },
};

// Chunk text is raw source code (see engine.py's _build_result — the "File: ... |
// Role: ... | Purpose: ..." embedding header is already stripped server-side, but
// the body is still real code, not prose). Show a short preview, not the whole thing.
const PREVIEW_LINE_COUNT = 4;

function previewLines(text) {
  const lines = (text || '').split('\n').filter((l) => l.trim() !== '');
  const preview = lines.slice(0, PREVIEW_LINE_COUNT);
  return { preview, truncated: lines.length > PREVIEW_LINE_COUNT };
}

export default function ChunkList({ chunks, selectedIndex, onSelect }) {
  if (!chunks || chunks.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto px-2.5 py-3">
        <p className="text-xs text-muted-teal italic text-center py-4 px-2">
          Select a question to see the answer broken into source chunks.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-2.5 py-3 flex flex-col gap-2">
      {chunks.map((chunk) => {
        const style = COL_STYLES[chunk.col] || COL_STYLES.c;
        const isSelected = selectedIndex === chunk.index;
        const { preview, truncated } = previewLines(chunk.text);
        return (
          <div
            key={chunk.index}
            onClick={() => onSelect(chunk)}
            className={`flex gap-2.5 px-3 py-2.5 rounded cursor-pointer border transition-colors ${
              isSelected
                ? `${style.hl} border`
                : 'border-transparent hover:bg-black/[0.04] hover:border-muted-teal/30'
            }`}
          >
            <div
              className={`w-5 h-5 rounded-full flex items-center justify-center font-mono text-[0.62rem] font-bold flex-shrink-0 mt-0.5 ${style.num}`}
            >
              {chunk.index}
            </div>
            <div className="min-w-0 flex-1">
              <pre className="font-mono text-[0.72rem] leading-[1.55] text-charcoal whitespace-pre-wrap break-words bg-black/[0.03] rounded px-2 py-1.5 overflow-hidden">
                {preview.join('\n')}
                {truncated && (
                  <span className="text-muted-teal">{'\n…'}</span>
                )}
              </pre>
              <div className="font-mono text-[0.63rem] text-muted-teal mt-1 whitespace-nowrap overflow-hidden text-ellipsis">
                {chunk.repo_name ? `${chunk.repo_name} · ` : ''}
                {chunk.file_path} · chunk {chunk.chunk_index}
                {truncated && ' · click to view full file →'}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
