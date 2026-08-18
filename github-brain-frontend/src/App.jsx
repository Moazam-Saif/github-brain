import { useState, useEffect } from 'react';
import Header from './components/Header';
import AskBar from './components/AskBar';
import SummaryCard from './components/SummaryCard';
import SourceViewer from './components/SourceViewer';
import ProseAnswer from './components/ProseAnswer';
import SectionNav from './components/SectionNav';
import { fetchRepos, askQuestionStream, resetSession } from './api';

export default function App() {
  const [repos, setRepos] = useState([]);
  const [sessionId, setSessionId] = useState(null);

  // Progressive answer state — populated incrementally as SSE events arrive,
  // summary first (see askQuestionStream in api.js / query()'s on_event in
  // engine.py). `result` becomes the full merged object once 'done' fires;
  // until then these are the individual pieces so the UI can render each as
  // soon as it lands rather than waiting for the whole response.
  const [summary, setSummary] = useState(null);
  const [sections, setSections] = useState([]);
  const [answer, setAnswer] = useState(null);
  const [result, setResult] = useState(null); // full response, set on 'done'

  const [selectedChunk, setSelectedChunk] = useState(null);
  // Set by clicking an inline code reference (e.g. `request.form.get`) in
  // the answer prose — jumps the source viewer to an exact file+line that
  // isn't necessarily inside any chunk's estimated highlight range, so it's
  // tracked separately from selectedChunk rather than overloading it.
  const [jumpTarget, setJumpTarget] = useState(null);
  const [activeSectionIndex, setActiveSectionIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeRepo, setActiveRepo] = useState(null);

  useEffect(() => {
    fetchRepos()
      .then(setRepos)
      .catch((err) => console.error('Failed to load repos:', err));
  }, []);

  async function handleAsk(question) {
    setLoading(true);
    setError(null);
    setSummary(null);
    setSections([]);
    setAnswer(null);
    setResult(null);
    setSelectedChunk(null);
    setJumpTarget(null);
    setActiveSectionIndex(0);

    try {
      await askQuestionStream(question, sessionId, (eventName, data) => {
        switch (eventName) {
          case 'summary':
            setSummary(data.summary);
            break;
          case 'sections':
            setSections(data.sections);
            break;
          case 'answer':
            setAnswer(data.answer);
            break;
          case 'chunk_blocks':
            // Chunk headings/text arrive here but we don't have file/score/
            // etc. for them yet — those only exist once 'done' ships the
            // full chunks array (files need the GitHub fetch, which is why
            // 'done' waits — see Option C in INTEGRATION_PROGRESS.md). No
            // separate state needed; 'done' below is what actually
            // populates the chunk list.
            break;
          case 'done':
            setSessionId(data.session_id);
            setActiveRepo(data.repo);
            setResult(data);
            setLoading(false);
            break;
          case 'error':
            setError(data.message);
            setLoading(false);
            break;
          default:
            break;
        }
      });
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  }

  // Reset the session when the user switches to asking about a different
  // repo than the one currently active (plan Section 14i) — only relevant
  // for repo_specific sessions; cross-repo questions naturally reset
  // server-side already.
  async function handleAskWithReset(question) {
    if (
      result?.query_type === 'repo_specific' &&
      activeRepo &&
      !question.toLowerCase().includes(activeRepo.toLowerCase())
    ) {
      await resetSession(sessionId).catch(() => {});
      setSessionId(null);
    }
    handleAsk(question);
  }

  const hasChunks = result?.chunks && result.chunks.length > 0;
  // Prefer the fully-merged result's answer once available (it's re-parsed
  // from the complete text server-side, so it's authoritative); fall back to
  // the streamed-in answer piece while still waiting on 'done'.
  const displayAnswer = result?.answer ?? answer;
  const displaySummary = result?.summary ?? summary;
  const displaySections = result?.sections?.length ? result.sections : sections;
  // section_content (per-topic body + which chunk indices that topic
  // references) only exists once 'done' ships the full result — it's
  // computed server-side from the complete answer text, see engine.py's
  // _split_answer_into_sections. Before that, there's nothing to slice by
  // topic yet, so the tab bar/body just isn't shown until 'done' (the
  // un-sliced streamed-in `answer` isn't usable for per-topic display).
  const sectionContent = result?.section_content || [];
  // Chunks are only real, indexed data once 'done' fires (see note in
  // 'chunk_blocks' case above) — using result?.chunks here (not just
  // displayAnswer's chunk markers) ensures marker clicks work as soon as
  // they're clickable, and are correctly NON-clickable (render as plain
  // text) before that, rather than silently doing nothing.
  const displayChunks = result?.chunks || [];

  function handleJumpToSource(chunk) {
    // Used by inline [N] markers. Per the user's ask: this moves the source
    // file panel ONLY — it must NOT change which topic/section is currently
    // showing in the answer panel (previously this also called
    // setActiveSectionIndex, which visibly changed the answer view out from
    // under the reader).
    setSelectedChunk(chunk);
    setJumpTarget(null);
  }

  function handleJumpToLine(filePath, lineNumber) {
    // Used by inline code-reference clicks (e.g. `request.form.get`) — an
    // exact file+line match found by searching fetched file content client-
    // side (see ProseAnswer's findCodeReference), independent of any
    // chunk's estimated highlight range. Also does not touch the topic tab.
    setJumpTarget({ filePath, lineNumber });
  }

  function handleFileSelect(filePath) {
    // Triggered by clicking a source file tab directly. This is a
    // deliberate "I want to look at this file" action rather than an
    // inline reference click mid-read, so switching the topic tab to match
    // still makes sense here (unlike handleJumpToSource above).
    const chunk = displayChunks.find((c) => c.file_path === filePath);
    if (chunk) {
      setSelectedChunk(chunk);
      setJumpTarget(null);
      const idx = sectionContent.findIndex((s) => s.chunk_indices.includes(chunk.index));
      if (idx !== -1) setActiveSectionIndex(idx);
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      {/* .page-body: flex:1 so it (and .cards inside it) fill all remaining
          vertical space below the header/ticker — NOT a centered max-width
          column. Padding driven entirely by --gap so it shrinks in step
          with the header/summary bleed at the same breakpoints. */}
      <main className="flex-1 flex flex-col gap-5 min-h-0 px-(--gap) pt-(--gap) pb-8">
        <AskBar repos={repos} onAsk={handleAskWithReset} activeRepo={activeRepo} />

        <SummaryCard summary={displaySummary} />

        {error && (
          <p className="text-sm text-purple font-mono">Error: {error}</p>
        )}

        {/* .cards: grid, 1fr 1fr, flex:1 + min-height:0 so it fills the rest
            of the page and its children can scroll internally instead of
            growing the page. At 768px it narrows but stays side-by-side; at
            560px it stacks to one column and stops stretching (each card
            gets its own min/max-height instead), exactly matching the mock. */}
        <div className="grid grid-cols-2 gap-5 flex-1 min-h-0 max-[768px]:min-h-[320px] max-[560px]:grid-cols-1 max-[560px]:flex-none">
          {/* LEFT: answer chunks or prose */}
          <div className="rounded-md flex flex-col overflow-hidden min-h-0 bg-cream-dark border-[1.5px] border-muted-teal max-[560px]:min-h-[280px] max-[560px]:max-h-[60vh]">
            <div className="px-[1.1rem] py-3 flex items-center justify-between flex-shrink-0 border-b border-muted-teal">
              <h3 className="font-mono text-[0.65rem] tracking-[0.2em] uppercase text-charcoal">
                Answer chunks
              </h3>
              <span className="font-mono text-[0.6rem] text-muted-teal">
                {loading && !hasChunks
                  ? 'Searching...'
                  : hasChunks
                  ? `${result.chunks.length} chunks`
                  : ''}
              </span>
            </div>

            {!displaySummary && !displaySections.length && !displayAnswer && loading ? (
              <p className="text-xs text-muted-teal italic text-center py-4 px-2">
                Searching your repos...
              </p>
            ) : !displaySections.length && !displayAnswer && !result ? (
              <div className="flex-1 overflow-y-auto px-2.5 py-3">
                <p className="text-xs text-muted-teal italic text-center py-4 px-2">
                  Select a question to see the answer broken into source chunks.
                </p>
              </div>
            ) : sectionContent.length > 0 ? (
              // 'done' has landed and the answer was split into per-topic
              // slices — show the tab bar + only the active topic's body,
              // per the user's ask (each topic hidden behind its own tab,
              // like the source file tabs on the right).
              <div className="flex-1 overflow-y-auto flex flex-col">
                <SectionNav
                  sections={sectionContent}
                  activeIndex={activeSectionIndex}
                  onSelect={setActiveSectionIndex}
                />
                <ProseAnswer
                  body={sectionContent[activeSectionIndex]?.body}
                  chunks={displayChunks}
                  references={result?.references}
                  onJumpToChunk={handleJumpToSource}
                  onJumpToLine={handleJumpToLine}
                />
              </div>
            ) : (
              // Mid-stream: summary/sections/answer text has arrived but
              // 'done' hasn't landed yet, so section_content (the per-topic
              // split) doesn't exist yet. Show the raw streamed-in answer
              // as one block for now — it'll be replaced by the tabbed view
              // above the moment 'done' fires. Chunks aren't real/indexed
              // yet either at this point, so markers render as plain text
              // (not clickable) until then — see displayChunks above.
              <div className="flex-1 overflow-y-auto flex flex-col">
                <ProseAnswer body={displayAnswer} chunks={displayChunks} onJumpToChunk={handleJumpToSource} />
              </div>
            )}
          </div>

          {/* RIGHT: source file viewer */}
          <div className="rounded-md flex flex-col overflow-hidden min-h-0 bg-purple border-[1.5px] border-bright-green max-[560px]:min-h-[280px] max-[560px]:max-h-[60vh]">
            <div className="px-[1.1rem] py-3 flex items-center justify-between flex-shrink-0 border-b border-bright-green/20">
              <h3 className="font-mono text-[0.65rem] tracking-[0.2em] uppercase text-bright-green">
                Source file
              </h3>
            </div>
            {hasChunks ? (
              <SourceViewer
                files={result.files}
                chunks={result.chunks}
                selectedChunk={selectedChunk}
                jumpTarget={jumpTarget}
                onFileSelect={handleFileSelect}
                onSelectChunk={handleJumpToSource}
              />
            ) : (
              <div className="flex flex-col items-center justify-center flex-1 p-8 gap-2.5">
                <p className="text-[0.8rem] text-white/30 text-center leading-relaxed">
                  {result
                    ? 'No source files for this query type.'
                    : 'Click a chunk on the left to jump to its source lines.'}
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
