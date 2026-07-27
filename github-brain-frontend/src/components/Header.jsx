export default function Header() {
  const bits =
    '1 0 0 1 0 0 0 1 0 0 0 1 0 1 1 1 1 0 0 1 1 1 1 1 0 0 0 0 1 0 1 0 1 0 1 0 1 1 1 0 1 0 0 0 1 0 1 0 0 1 1 0 0 1 0 1 0 1 0 1 1 0 1 0 0 0 1 0 1 1 1 0 0 1 0 0 1 0 0 0 1 0 1 1 1 0 0 1';

  return (
    <>
      <header className="bg-dark-green px-4 sm:px-8 py-5 text-center">
        <h1 className="font-mono text-[clamp(1.5rem,4vw,2.5rem)] tracking-[0.18em] uppercase text-purple">
          Moazam's Repo
        </h1>
      </header>

      <div className="flex items-center mt-6 px-2 sm:px-0 gap-2">
        <div className="flex-1 overflow-hidden py-2 min-w-0">
          <div className="flex whitespace-nowrap ticker-track">
            <span className="font-mono text-xs text-charcoal tracking-[0.1em] pr-12 opacity-35 inline-block">
              {bits}
            </span>
            <span className="font-mono text-xs text-charcoal tracking-[0.1em] pr-12 opacity-35 inline-block">
              {bits}
            </span>
          </div>
        </div>
        <button className="flex-shrink-0 bg-cream-dark text-charcoal font-sans text-xs sm:text-sm font-semibold px-3 sm:px-5 py-2 rounded-l-md border border-muted-teal border-r-0 flex items-center gap-1.5 whitespace-nowrap">
          <span className="hidden sm:inline">Moazam's repo</span>
          <span className="sm:hidden">Repo</span>
          <span className="text-[0.7rem]">▾</span>
        </button>
      </div>
    </>
  );
}
