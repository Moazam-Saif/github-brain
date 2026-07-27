/**
 * Lightweight regex-based syntax highlighter.
 * Ported from the original index.html mock's highlight()/esc() functions.
 * Returns an array of {text, cls} tokens instead of an HTML string, so
 * React can render them as <span> elements without dangerouslySetInnerHTML.
 */

const KW   = /\b(const|let|var|function|return|if|else|try|catch|throw|new|class|extends|super|import|require|module|exports|async|await|for|of|in|while|do|typeof|instanceof|true|false|null|undefined)\b/g;
const STR  = /(["'`])(?:\\.|(?!\1)[^\\])*\1/g;
const CMT  = /(\/\/.*)/g;
const NUM  = /\b(\d+)\b/g;
const FN   = /\b([a-zA-Z_$][\w$]*)(?=\s*\()/g;
const PROP = /\.([a-zA-Z_$][\w$]*)/g;

function collect(re, cls, raw, tokens) {
  let m;
  re.lastIndex = 0;
  while ((m = re.exec(raw)) !== null) {
    tokens.push({ s: m.index, e: m.index + m[0].length, cls, txt: m[0] });
    // Guard against zero-length matches looping forever.
    if (m[0].length === 0) re.lastIndex++;
  }
}

/**
 * Tokenize a single line of code into an array of
 * { text: string, cls: string | null } segments, in left-to-right order,
 * with no overlaps (first-come wins — same precedence as the original:
 * comments, then strings, then keywords, then function names, then
 * properties, then numbers).
 */
export function tokenizeLine(raw) {
  if (!raw || !raw.trim()) return [{ text: raw ?? '', cls: null }];

  let tokens = [];
  collect(CMT,  'tok-cmt',  raw, tokens);
  collect(STR,  'tok-str',  raw, tokens);
  collect(KW,   'tok-kw',   raw, tokens);
  collect(FN,   'tok-fn',   raw, tokens);
  collect(PROP, 'tok-prop', raw, tokens);
  collect(NUM,  'tok-num',  raw, tokens);

  tokens.sort((a, b) => a.s - b.s);
  const clean = [];
  let cursor = 0;
  for (const t of tokens) {
    if (t.s < cursor) continue;
    clean.push(t);
    cursor = t.e;
  }

  const segments = [];
  let pos = 0;
  for (const t of clean) {
    if (t.s > pos) segments.push({ text: raw.slice(pos, t.s), cls: null });
    segments.push({ text: t.txt, cls: t.cls });
    pos = t.e;
  }
  if (pos < raw.length) segments.push({ text: raw.slice(pos), cls: null });

  return segments.length ? segments : [{ text: raw, cls: null }];
}
