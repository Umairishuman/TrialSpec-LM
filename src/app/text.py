"""Turning user text into numbered sentences and criterion lists (D-028).

A note already in the dataset format ("<id>. sentence" per line) keeps its original IDs, so cited IDs stay
comparable with the source record (rubric C3). Free text is split into sentences and numbered from 0, as in
the dataset.
"""
import re

_NUMBERED = re.compile(r"^\s*(\d+)\.\s+(\S.*?)\s*$")
# Bullets and list markers: "-", "*", "•", "1.", "1)", "(1)", "a)", "(a)". Not "a." (would eat "E. coli").
_BULLET = re.compile(r"^\s*(?:[-*•●▪]|\(?[0-9]{1,3}[.)]|\(?[a-zA-Z]\))\s+")
# Candidate sentence end: terminal punctuation (plus closing quotes/brackets), whitespace, then a plausible start.
_END = re.compile(r"([.!?]+[\"')\]]*)\s+(?=[\"'(\[]?[A-Z0-9])")
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr", "vs", "etc", "approx", "no", "fig", "al", "dept",
    "hx", "pt", "pts", "yr", "yrs", "wk", "wks", "mo", "mos", "ca", "cf", "inc", "ltd", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_TERMINAL = (".", "!", "?", ":", ";")


def _is_abbreviation(text_before: str) -> bool:
    """True if the period ending `text_before` belongs to an abbreviation, not a sentence end."""
    m = re.search(r"([A-Za-z.]+)\.$", text_before)
    if not m:
        return False
    token = m.group(1)
    if "." in token:                     # e.g. / i.e. / y.o. / b.i.d.
        return True
    if len(token) == 1:                  # initials such as "J. Smith"
        return True
    return token.lower() in ABBREVIATIONS


def _split_block(block: str) -> list[str]:
    out, start = [], 0
    for m in _END.finditer(block):
        end = m.end(1)
        piece = block[start:end]
        if piece.endswith(".") and _is_abbreviation(piece):
            continue
        out.append(piece.strip())
        start = m.end()
    out.append(block[start:].strip())
    return [s for s in out if s]


def _blocks(note: str) -> list[str]:
    """Join hard-wrapped lines; break at blank lines, bullets, and lines that end with terminal punctuation."""
    blocks, cur = [], []
    for raw in note.splitlines():
        line = raw.strip()
        if not line:
            if cur:
                blocks.append(" ".join(cur)); cur = []
            continue
        is_bullet = bool(_BULLET.match(line))
        line = _BULLET.sub("", line) if is_bullet else line
        if is_bullet and cur:
            blocks.append(" ".join(cur)); cur = []
        cur.append(line)
        if line.endswith(_TERMINAL) or is_bullet:
            blocks.append(" ".join(cur)); cur = []
    if cur:
        blocks.append(" ".join(cur))
    return blocks


def split_note(note: str) -> tuple[list[tuple[int, str]], str]:
    """Return ([(id, sentence)], mode) with mode "numbered" (original IDs kept) or "split" (numbered from 0)."""
    lines = [l for l in (note or "").splitlines() if l.strip()]
    if not lines:
        return [], "empty"
    matches = [_NUMBERED.match(l) for l in lines]
    if all(matches):
        ids = [int(m.group(1)) for m in matches]
        if len(set(ids)) == len(ids):
            return [(i, re.sub(r"\s+", " ", m.group(2))) for i, m in zip(ids, matches)], "numbered"
    sentences = []
    for block in _blocks(note):
        sentences.extend(_split_block(re.sub(r"\s+", " ", block)))
    return list(enumerate(sentences)), "split"


def parse_criteria(text: str) -> tuple[list[str], list[str]]:
    """One criterion per line; strips bullets/numbering, drops blanks and duplicates. Returns (criteria, notices)."""
    out, notices, seen = [], [], set()
    for raw in (text or "").splitlines():
        line = re.sub(r"\s+", " ", _BULLET.sub("", raw.strip())).strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            notices.append(f"Duplicate criterion ignored: “{line[:80]}”")
            continue
        seen.add(key)
        out.append(line)
    return out, notices
