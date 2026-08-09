export default function SummaryCard({ summary }) {
  return (
    // Bleeds to the true left edge of the viewport: negative margin equal to
    // the page's own --gap padding, then the inner padding is compensated
    // (gap + 0.5rem) so the text stays visually aligned with the rest of the
    // content — ported exactly from the mock's .summary-wrap/.summary calc().
    <div className="ml-[calc(-1*var(--gap))]">
      <div className="bg-muted-teal rounded-r-md py-5 pr-8 pl-[calc(var(--gap)+0.5rem)]">
        <h2 className="font-mono text-[0.68rem] tracking-[0.2em] uppercase text-dark-green mb-2">
          Summary
        </h2>
        <p className="text-[0.9rem] leading-[1.7] text-dark-green">
          {summary || 'Pick a question above to see a summary.'}
        </p>
      </div>
    </div>
  );
}
