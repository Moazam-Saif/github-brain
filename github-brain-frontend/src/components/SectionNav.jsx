// Renders the answer's section headings as clickable jump links. Scrolls
// to the matching "## Heading" in the rendered ProseAnswer below it via
// element id (see ProseAnswer's heading id generation — must match).
function slugify(heading) {
  return heading
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

export default function SectionNav({ sections }) {
  if (!sections || sections.length === 0) return null;

  function handleClick(heading) {
    const el = document.getElementById(`section-${slugify(heading)}`);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  return (
    <div className="flex flex-wrap gap-1.5 px-2.5 pt-2">
      {sections.map((s, i) => (
        <button
          key={i}
          onClick={() => handleClick(s.heading)}
          className="text-[0.68rem] font-mono px-2 py-1 rounded border border-muted-teal/50 text-muted-teal hover:bg-muted-teal hover:text-dark-green transition-colors"
        >
          {s.heading}
        </button>
      ))}
    </div>
  );
}

export { slugify };
