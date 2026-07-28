import { useState, useEffect } from 'react';
import Header from './components/Header';
import AskBar from './components/AskBar';
import SummaryCard from './components/SummaryCard';
import ChunkList from './components/ChunkList';
import SourceViewer from './components/SourceViewer';
import ProseAnswer from './components/ProseAnswer';
import { fetchRepos, askQuestion, resetSession } from './api';

export default function App() {
  const [repos, setRepos] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [result, setResult] = useState(null); // full /ask response
  const [selectedChunk, setSelectedChunk] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRepos()
      .then(setRepos)
      .catch((err) => console.error('Failed to load repos:', err));
  }, []);

  async function handleAsk(question) {
    setLoading(true);
    setError(null);
    try {
      const data = await askQuestion(question, sessionId);
      setSessionId(data.session_id);
      setResult(data);
      setSelectedChunk(null);
    } catch (err) {
      setError(err.message);
    } finally {
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
      result?.repo &&
      !question.toLowerCase().includes(result.repo.toLowerCase())
    ) {
      await resetSession(sessionId).catch(() => {});
      setSessionId(null);
    }
    handleAsk(question);
  }

  const hasChunks = result?.chunks && result.chunks.length > 0;

  return (
    <div className="min-h-screen bg-cream text-charcoal">
      <Header />

      <main className="max-w-[860px] mx-auto px-4 sm:px-6 pt-6 pb-12 flex flex-col gap-6">
        <AskBar repos={repos} onAsk={handleAskWithReset} activeRepo={result?.repo} />

        <SummaryCard summary={result?.summary} />

        {error && (
          <p className="text-sm text-purple font-mono">Error: {error}</p>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          {/* LEFT: answer chunks or prose */}
          <div className="rounded-md min-h-[340px] flex flex-col overflow-hidden bg-cream-dark border-[1.5px] border-muted-teal">
            <div className="px-4.5 py-3.5 flex items-center justify-between flex-shrink-0 border-b border-muted-teal">
              <h3 className="font-mono text-[0.68rem] tracking-[0.2em] uppercase text-charcoal">
                Answer chunks
              </h3>
              <span className="font-mono text-[0.62rem] text-muted-teal">
                {loading
                  ? 'Searching...'
                  : hasChunks
                  ? `${result.chunks.length} chunks`
                  : ''}
              </span>
            </div>

            {loading ? (
              <p className="text-xs text-muted-teal italic text-center py-4 px-2">
                Searching your repos...
              </p>
            ) : !result ? (
              <div className="flex-1 overflow-y-auto px-2.5 py-3">
                <p className="text-xs text-muted-teal italic text-center py-4 px-2">
                  Select a question to see the answer broken into source chunks.
                </p>
              </div>
            ) : hasChunks ? (
              <div className="flex-1 overflow-y-auto flex flex-col">
                <ProseAnswer
                  answer={result.answer}
                  chunks={result.chunks}
                  onJumpToChunk={setSelectedChunk}
                />
                <div className="border-t border-muted-teal/40 mx-2.5" />
                <ChunkList
                  chunks={result.chunks}
                  selectedIndex={selectedChunk?.index}
                  onSelect={setSelectedChunk}
                />
              </div>
            ) : (
              <ProseAnswer answer={result.answer} />
            )}
          </div>

          {/* RIGHT: source file viewer */}
          <div className="rounded-md min-h-[340px] flex flex-col overflow-hidden bg-purple border-[1.5px] border-bright-green">
            <div className="px-4.5 py-3.5 flex items-center justify-between flex-shrink-0 border-b border-bright-green/20">
              <h3 className="font-mono text-[0.68rem] tracking-[0.2em] uppercase text-bright-green">
                Source file
              </h3>
            </div>
            {hasChunks ? (
              <SourceViewer
                files={result.files}
                chunks={result.chunks}
                selectedChunk={selectedChunk}
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
