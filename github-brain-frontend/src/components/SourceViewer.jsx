import { useState, useEffect } from 'react';
import { tokenizeLine } from '../utils/highlight';

const HL_ROW_CLASS = { a: 'hl-row-a', b: 'hl-row-b', c: 'hl-row-c' };

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

export default function SourceViewer({ files, chunks, selectedChunk }) {
  const fileKeys = Object.keys(files || {});
  const [activeKey, setActiveKey] = useState(fileKeys[0] || null);

  // Switch to the file tab for whichever chunk was just selected.
  useEffect(() => {
    if (selectedChunk?.file_path && files[selectedChunk.file_path]) {
      setActiveKey(selectedChunk.file_path);
    }
  }, [selectedChunk, files]);

  useEffect(() => {
    if (!activeKey && fileKeys.length > 0) setActiveKey(fileKeys[0]);
  }, [fileKeys, activeKey]);

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
  let highlightRange = null;
  if (selectedChunk && activeFile && selectedChunk.file_path === activeKey) {
    highlightRange = estimateLineRange(selectedChunk, chunks, activeFile.lines.length);
  }

  return (
    <>
      <div className="flex overflow-x-auto flex-shrink-0 border-b border-bright-green/15">
        {fileKeys.map((key) => (
          <button
            key={key}
            onClick={() => setActiveKey(key)}
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

      <div className="flex-1 overflow-y-auto">
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
                const inRange =
                  highlightRange &&
                  lineNum >= highlightRange.startLine &&
                  lineNum <= highlightRange.endLine;
                const rowClass = inRange ? HL_ROW_CLASS[selectedChunk.col] || '' : '';
                return (
                  <tr key={lineNum} className={rowClass}>
                    <td className="ln text-white/18 text-right select-none min-w-[28px] pr-3.5 border-r border-white/[0.08] py-px px-2.5">
                      {lineNum}
                    </td>
                    <td className="text-white/55 whitespace-pre py-px px-2.5">
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
