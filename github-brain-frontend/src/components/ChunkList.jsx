const COL_STYLES = {
  a: { num: 'bg-purple/15 text-purple', hl: 'bg-hl-a-bg border-hl-a-bd' },
  b: { num: 'bg-muted-teal/25 text-[#3a7a6d]', hl: 'bg-hl-b-bg border-hl-b-bd' },
  c: { num: 'bg-bright-green/25 text-[#2e7a38]', hl: 'bg-hl-c-bg border-hl-c-bd' },
};

// chunk.heading/chunk.text are parsed server-side from the SAME Gemini call
// that produced the main answer (see engine.py's CHUNK_STRUCTURE_INSTRUCTION /
// _split_answer_and_chunk_blocks) — real prose explanation with a short
// heading, not raw code. The actual code lives in the Source file panel via
// `files`. If parsing didn't find a block for a chunk (heading/text both
// empty — partial-degrade case), fall back to just the file reference.

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
        const hasExplanation = chunk.heading || chunk.text;
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
            <div className="min-w-0">
              {hasExplanation ? (
                <>
                  {chunk.heading && (
                    <div className="text-[0.78rem] font-semibold text-charcoal mb-0.5">
                      {chunk.heading}
                    </div>
                  )}
                  <div className="text-[0.82rem] leading-[1.62] text-charcoal">
                    {chunk.text}
                  </div>
                </>
              ) : (
                <div className="text-[0.82rem] leading-[1.62] text-muted-teal italic">
                  Related code — see file reference below.
                </div>
              )}
              <div className="font-mono text-[0.63rem] text-muted-teal mt-1 whitespace-nowrap overflow-hidden text-ellipsis">
                {chunk.repo_name ? `${chunk.repo_name} · ` : ''}
                {chunk.file_path} · chunk {chunk.chunk_index}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
