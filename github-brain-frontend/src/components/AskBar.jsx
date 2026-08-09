import { useState } from 'react';

export default function AskBar({ repos, onAsk, activeRepo }) {
  const [modalOpen, setModalOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');

  function submit() {
    const val = inputValue.trim();
    setModalOpen(false);
    setInputValue('');
    if (val) onAsk(val);
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') submit();
  }

  return (
    <>
      {/* Single row, wraps naturally; only stacks to a column at the 560px
          mobile breakpoint, matching the mock's .ask-row exactly (the 768px
          tablet tier doesn't touch this layout at all). */}
      <div className="flex items-center gap-3 flex-wrap max-[560px]:flex-col max-[560px]:items-start max-[560px]:gap-2.5">
        <button
          onClick={() => setModalOpen(true)}
          className="flex-shrink-0 bg-bright-green text-dark-green font-mono text-[0.85rem] font-bold tracking-[0.12em] uppercase px-8 py-3 rounded whitespace-nowrap transition-[background,transform] duration-150 hover:bg-muted-teal hover:-translate-y-px max-[768px]:text-[0.78rem] max-[768px]:px-[1.4rem] max-[768px]:py-[0.65rem] max-[560px]:w-full max-[560px]:text-center"
        >
          Ask Question
        </button>

        <div className="flex gap-[0.45rem] flex-wrap items-center max-[560px]:w-full">
          {repos.map((repo) => (
            <button
              key={repo.repo_name}
              onClick={() => onAsk(`${repo.repo_name} — tell me about this repo`)}
              className={`text-[0.73rem] rounded-full px-3 py-[0.28rem] border font-sans whitespace-nowrap transition-colors ${
                activeRepo === repo.repo_name
                  ? 'bg-muted-teal text-dark-green border-muted-teal'
                  : 'border-muted-teal text-charcoal hover:bg-muted-teal hover:text-dark-green'
              }`}
            >
              {repo.repo_name}
            </button>
          ))}
        </div>
      </div>

      {modalOpen && (
        <div
          className="fixed inset-0 bg-dark-green/65 z-[100] flex items-center justify-center"
          onClick={(e) => e.target === e.currentTarget && setModalOpen(false)}
        >
          <div className="bg-cream rounded-lg p-6 w-[min(540px,92vw)] flex flex-col gap-3.5">
            <h2 className="font-mono text-sm tracking-[0.1em] uppercase text-dark-green">
              Ask a question
            </h2>
            <input
              autoFocus
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your question…"
              className="px-4 py-3 border border-muted-teal rounded font-sans text-sm bg-cream text-charcoal outline-none focus:border-purple w-full"
            />
            <div className="flex gap-2.5 justify-end">
              <button
                onClick={() => setModalOpen(false)}
                className="px-5 py-2.5 rounded border border-muted-teal text-charcoal font-mono text-xs"
              >
                Cancel
              </button>
              <button
                onClick={submit}
                className="px-5 py-2.5 rounded bg-bright-green text-dark-green font-mono text-xs font-bold"
              >
                Ask →
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
