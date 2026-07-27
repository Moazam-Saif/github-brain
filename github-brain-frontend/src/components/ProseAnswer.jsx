export default function ProseAnswer({ answer }) {
  const paragraphs = (answer || '').split('\n\n').filter((p) => p.trim());

  return (
    <div className="flex-1 overflow-y-auto px-2.5 py-3 flex flex-col gap-2">
      {paragraphs.map((para, i) => (
        <p key={i} className="text-[0.85rem] leading-[1.7] text-charcoal p-2">
          {para}
        </p>
      ))}
    </div>
  );
}
