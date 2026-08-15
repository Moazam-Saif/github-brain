// Topic tab bar — mirrors the Source file panel's file-tab style. One topic
// is active at a time (index into `sections`/`sectionContent`); its body is
// the only one shown below (see App.jsx). Auto-selects to match whichever
// chunk/source file was just clicked (see onSelectChunk's chunk_indices
// lookup in App.jsx) rather than scrolling, since other sections' text is
// now hidden, not just off-screen.
export default function SectionNav({ sections, activeIndex, onSelect }) {
  if (!sections || sections.length === 0) return null;

  return (
    <div className="flex overflow-x-auto flex-shrink-0 border-b border-muted-teal/40 px-1">
      {sections.map((s, i) => (
        <button
          key={i}
          onClick={() => onSelect(i)}
          className={`px-3 py-2 font-mono text-[0.66rem] whitespace-nowrap border-b-2 transition-colors ${
            i === activeIndex
              ? 'text-purple border-purple font-bold'
              : 'text-muted-teal border-transparent hover:text-charcoal'
          }`}
        >
          {s.heading}
        </button>
      ))}
    </div>
  );
}
