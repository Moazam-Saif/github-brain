import { useState, useEffect } from 'react';
import { tokenizeLine } from '../utils/highlight';

const HL_ROW_CLASS = { a: 'hl-row-a', b: 'hl-row-b', c: 'hl-row-c' };
const HL_BORDER_CLASS = { a: 'border-purple', b: 'border-muted-teal', c: 'border-bright-green' };

/**
 * Strategy A line-range approximation (INTEGRATION_PLAN.md Section 6).
 * chunk_index is the chunk's position within the file, NOT a line number —
 * this estimates a line range assuming chunks are evenly distributed.
 * TODO: replace with chunk.start_line / chunk.end_line once chunker.py
 * is updated to store exact line ranges (plan Section 6, Strategy B).
 */
function estimateLineRange(chunk, allChunks, totalLines) {
  const chunksForFile = allChunks.filter((c) => c.file_path === chunk.file_path);
  const totalChunks = Math.max(chunksForFile.length, 1);
  const linesPerChunk = Math.ceil(totalLines / totalChunks) || 1;
  const startLine = chunk.chunk_index * linesPerChunk + 1;
  const endLine = Math.min(startLine + linesPerChunk - 1, totalLines);
  return { startLine, endLine };
}

export default function SourceViewer({ files, chunks, selectedChunk, onFileSelect, onSelectChunk }) {
  const fileKeys = Object.keys(files || {});
  const [activeKey, setActiveKey] = useState(fileKeys[0] || null);
  // Which chunk's hover popup is currently open, by chunk index — not tied
  // to selectedChunk, since EVERY chunk in the file gets its own hoverable
  // icon now, not just the selected one.
  const [openInfoIndex, setOpenInfoIndex] = useState(null);

  function selectFile(key) {
    setActiveKey(key);
    if (onFileSelect) onFileSelect(key);
  }

  useEffect(() => {
    if (selectedChunk?.file_path && files[selectedChunk.file_path]) {
      selectFile(selectedChunk.file_path);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedChunk, files]);

  useEffect(() => {
    if (!activeKey && fileKeys.length > 0) selectFile(fileKeys[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileKeys, activeKey]);

  useEffect(() => {
    setOpenInfoIndex(null);
  }, [activeKey]);

  if (fileKeys.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center flex-1 p-8 gap-2.5">
        <p className="text-[0.8rem] text-white/30 text-center leading-relaxed">
          Click a chunk on the left to jump to its source lines.
        </p>
      </div>
    );
  }

  const activeFile = files[activeKey];

  // Every chunk that belongs to the currently active file gets its own
  // highlighted region + dotted border + hover icon — not just whichever
  // one is "selected". selectedChunk still gets the stronger solid
  // background treatment so there's a visible difference between "this is
  // one of the chunks referenced in this file" and "this is the one you
  // just clicked".
  const chunksInFile = (chunks || []).filter((c) => c.file_path === activeKey);
  const ranges = chunksInFile.map((c) => ({
    chunk: c,
    ...estimateLineRange(c, chunks, activeFile.lines.length),
  }));

  function rangeForLine(lineNum) {
    // A line can only sensibly belong to one chunk's range under the
    // current even-split estimation (Strategy A), so first match wins.
    return ranges.find((r) => lineNum >= r.startLine && lineNum <= r.endLine) || null;
  }

  return (
    <>
      <div className="flex overflow-x-auto flex-shrink-0 border-b border-bright-green/15">
        {fileKeys.map((key) => (
          <button
            key={key}
            onClick={() => selectFile(key)}
            className={`px-3.5 py-1.5 font-mono text-[0.63rem] whitespace-nowrap border-b-2 transition-colors ${
              key === activeKey
                ? 'text-bright-green border-bright-green'
                : 'text-white/38 border-transparent hover:text-white/70'
            }`}
          >
            {files[key].name}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto relative">
        <div className="px-3.5 py-[0.32rem] font-mono text-[0.6rem] text-white/28 border-b border-white/[0.07]">
          {activeFile.name}
        </div>

        {activeFile.lines.length === 0 ? (
          <p className="p-4 text-[0.8rem] text-white/30">
            File not available.
          </p>
        ) : (
          <table className="w-full border-collapse font-mono text-[0.69rem]">
            <tbody>
              {activeFile.lines.map((line, i) => {
                const lineNum = i + 1;
                const range = rangeForLine(lineNum);
                const chunk = range?.chunk;
                const isSelected = chunk && selectedChunk?.index === chunk.index;
                const isFirstInRange = range && lineNum === range.startLine;
                const isLastInRange  = range && lineNum === range.endLine;
                const borderClass = chunk ? HL_BORDER_CLASS[chunk.col] || 'border-white/40' : '';
                // Solid highlight background only for the actively-selected
                // chunk; other chunks in this file still get the dotted
                // border + icon but a plain (unhighlighted) row background,
                // so the selected one visually stands out.
                const rowClass = isSelected ? HL_ROW_CLASS[chunk.col] || '' : '';
                const topBorder    = isFirstInRange ? `border-t-2 border-dotted ${borderClass}` : '';
                const bottomBorder = isLastInRange ? `border-b-2 border-dotted ${borderClass}` : '';
                const sideBorder   = range ? `border-dotted ${borderClass}` : '';
                const hasInfo = chunk && (chunk.heading || chunk.text);
                const infoOpen = hasInfo && openInfoIndex === chunk.index;

                return (
                  <tr key={lineNum} className={rowClass}>
                    <td
                      className={`ln text-white/18 text-right select-none min-w-[28px] pr-3.5 border-r border-white/[0.08] py-px px-2.5 relative ${topBorder} ${bottomBorder} ${
                        range ? `border-l-2 ${sideBorder}` : ''
                      }`}
                    >
                      {isFirstInRange && hasInfo && (
                        <div
                          className="absolute -top-1.5 -left-1.5 z-10"
                          onMouseEnter={() => setOpenInfoIndex(chunk.index)}
                          onMouseLeave={() => setOpenInfoIndex(null)}
                        >
                          <button
                            onClick={() => onSelectChunk && onSelectChunk(chunk)}
                            className={`w-4 h-4 rounded-full border-2 border-dotted ${borderClass} bg-cream flex items-center justify-center cursor-help hover:bg-white transition-colors`}
                            aria-label={`Summary for chunk ${chunk.index}`}
                          >
                            <span className="text-[0.52rem] font-bold text-purple leading-none select-none">
                              i
                            </span>
                          </button>
                          {infoOpen && (
                            <div className="absolute top-4 left-0 w-72 bg-white border border-purple/20 rounded-md shadow-lg p-3.5 z-20">
                              {chunk.heading && (
                                <div className="font-sans text-[0.8rem] font-bold text-purple mb-1.5 leading-snug">
                                  {chunk.heading}
                                </div>
                              )}
                              {chunk.text && (
                                <div className="font-reading text-[0.86rem] leading-[1.6] text-purple">
                                  {chunk.text}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                      {lineNum}
                    </td>
                    <td
                      className={`text-white/55 whitespace-pre py-px px-2.5 ${topBorder} ${bottomBorder} ${
                        range ? `border-r-2 ${sideBorder}` : ''
                      }`}
                    >
                      {tokenizeLine(line).map((seg, si) =>
                        seg.cls ? (
                          <span key={si} className={seg.cls}>
                            {seg.text}
                          </span>
                        ) : (
                          <span key={si}>{seg.text}</span>
                        )
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
