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
      <div className="flex justify-start">
        <button
          onClick={() => setModalOpen(true)}
          className="w-full sm:w-auto bg-bright-green text-dark-green font-mono text-sm font-bold tracking-[0.12em] uppercase px-10 py-3.5 rounded sm:min-w-[280px] text-center transition-colors hover:bg-muted-teal hover:-translate-y-px"
        >
          Ask Question
        </button>
      </div>

      <div className="flex gap-2 flex-wrap">
        {repos.map((repo) => (
          <button
            key={repo.repo_name}
            onClick={() => onAsk(`${repo.repo_name} — tell me about this repo`)}
            className={`text-xs rounded-full px-3 py-1.5 border font-sans transition-colors ${
              activeRepo === repo.repo_name
                ? 'bg-muted-teal text-dark-green border-muted-teal'
                : 'border-muted-teal text-charcoal hover:bg-muted-teal hover:text-dark-green'
            }`}
          >
            {repo.repo_name}
          </button>
        ))}
      </div>

      {modalOpen && (
        <div
          className="fixed inset-0 bg-dark-green/65 z-[100] flex items-center justify-center"
          onClick={(e) => e.target === e.currentTarget && setModalOpen(false)}
        >
          <div className="bg-cream rounded-lg p-6 w-[min(540px,90vw)] flex flex-col gap-3.5">
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
              className="px-4 py-3 border border-muted-teal rounded font-sans text-sm bg-cream text-charcoal outline-none focus:border-purple"
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
