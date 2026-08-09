export default function Header() {
  const bits =
    '1 0 0 1 0 0 0 1 0 0 0 1 0 1 1 1 1 0 0 1 1 1 1 1 0 0 0 0 1 0 1 0 1 0 1 0 1 1 1 0 1 0 0 0 1 0 1 0 0 1 1 0 0 1 0 1 0 1 0 1 1 0 1 0 0 0 1 0 1 1 1 0 0 1 0 0 1 0 0 0 1 0 1 1 1 0 0 1';

  return (
    <>
      {/* Default padding 1.25rem 2rem; only changes at the 560px mobile
          breakpoint (1rem var(--gap)) — the 768px tablet tier leaves the
          header untouched, matching the mock exactly. */}
      <header className="bg-dark-green text-center flex-shrink-0 px-8 py-5 max-[560px]:px-(--gap) max-[560px]:py-4">
        <h1 className="font-mono text-[clamp(1.3rem,3vw,2.2rem)] tracking-[0.18em] uppercase text-purple">
          Moazam's Repo
        </h1>
      </header>

      <div className="flex items-center mt-6 flex-shrink-0">
        <div className="flex-1 overflow-hidden py-2 min-w-0 pl-(--gap)">
          <div className="flex whitespace-nowrap ticker-track">
            <span className="font-mono text-xs text-charcoal tracking-[0.1em] pr-12 opacity-35 inline-block">
              {bits}
            </span>
            <span className="font-mono text-xs text-charcoal tracking-[0.1em] pr-12 opacity-35 inline-block">
              {bits}
            </span>
          </div>
        </div>
        <button className="flex-shrink-0 bg-cream-dark text-charcoal font-sans text-[0.8rem] font-semibold px-[1.2rem] py-2 rounded-l-md border border-muted-teal border-r-0 flex items-center gap-1.5 whitespace-nowrap after:content-['▾'] after:text-[0.7rem]">
          <span className="max-[400px]:hidden">Moazam's repo</span>
          <span className="hidden max-[400px]:inline">Repo</span>
        </button>
      </div>
    </>
  );
}
