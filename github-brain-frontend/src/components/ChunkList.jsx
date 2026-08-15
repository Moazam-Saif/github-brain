const COL_STYLES = {
  a: { num: 'bg-purple/15 text-purple', hl: 'bg-hl-a-bg border-hl-a-bd' },
  b: { num: 'bg-muted-teal/25 text-[#3a7a6d]', hl: 'bg-hl-b-bg border-hl-b-bd' },
  c: { num: 'bg-bright-green/25 text-[#2e7a38]', hl: 'bg-hl-c-bg border-hl-c-bd' },
};

// Chunk heading/explanation text now lives ONLY in the Source file panel,
// as a hover popup anchored to each chunk's highlighted region (see
// SourceViewer.jsx) — not duplicated here. This list is just a compact,
// clickable index of which chunks were used and where they came from.

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
    <div className="flex-1 overflow-y-auto px-2.5 py-3 flex flex-col gap-1.5">
      {chunks.map((chunk) => {
        const style = COL_STYLES[chunk.col] || COL_STYLES.c;
        const isSelected = selectedIndex === chunk.index;
        return (
          <button
            key={chunk.index}
            onClick={() => onSelect(chunk)}
            className={`flex items-center gap-2.5 px-3 py-2 rounded cursor-pointer border transition-colors text-left ${
              isSelected
                ? `${style.hl} border`
                : 'border-transparent hover:bg-black/[0.04] hover:border-muted-teal/30'
            }`}
          >
            <div
              className={`w-5 h-5 rounded-full flex items-center justify-center font-mono text-[0.62rem] font-bold flex-shrink-0 ${style.num}`}
            >
              {chunk.index}
            </div>
            <div className="min-w-0 font-sans text-[0.8rem] text-charcoal truncate">
              {chunk.repo_name ? `${chunk.repo_name} · ` : ''}
              <span className="font-medium">{chunk.file_path}</span>
              <span className="text-muted-teal"> · chunk {chunk.chunk_index}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
