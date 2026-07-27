export default function SummaryCard({ summary }) {
  return (
    <div className="-ml-6">
      <div className="bg-muted-teal rounded-r-md py-6 pr-8 pl-12">
        <h2 className="font-mono text-xs tracking-[0.2em] uppercase text-dark-green mb-2.5">
          Summary
        </h2>
        <p className="text-[0.95rem] leading-[1.7] text-dark-green">
          {summary || 'Pick a question above to see a summary.'}
        </p>
      </div>
    </div>
  );
}
