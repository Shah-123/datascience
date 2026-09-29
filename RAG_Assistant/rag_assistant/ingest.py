"""Load documents, split them into sections, and chunk them for retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .text import slugify, split_sentences, strip_markdown


@dataclass(frozen=True)
class Section:
    doc_id: str
    doc_title: str
    heading: str
    text: str  # cleaned text; paragraphs / list items / table rows separated by "\n"

    @property
    def section_id(self) -> str:
        return f"{self.doc_id}#{slugify(self.heading)}"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str
    section_id: str
    text: str
    ordinal: int

    def index_text(self, add_heading: bool = True) -> str:
        """Text that gets indexed. Prepending the heading gives each chunk its context."""
        return f"{self.doc_title}. {self.heading}. {self.text}" if add_heading else self.text


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str = "structure"  # "structure": sentence-packed inside sections; "fixed": word windows
    max_words: int = 90
    overlap: int = 1  # structure: overlapping sentences; fixed: overlapping words
    add_heading: bool = True

    def label(self) -> str:
        h = "+h" if self.add_heading else ""
        return f"{self.strategy}-{self.max_words}w-o{self.overlap}{h}"


# --- loading ------------------------------------------------------------------------------


def _split_front_matter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---"):
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, flags=re.S)
        if m:
            return yaml.safe_load(m.group(1)) or {}, raw[m.end():]
    return {}, raw


def _linearize_table(rows: list[str]) -> list[str]:
    """Turn a markdown table into one sentence per row so it survives sentence-based chunking.

    ``| B- | 2.7 | 80-82 |`` under headers ``Letter grade | Grade points | Percentage`` becomes
    ``Letter grade B-: Grade points 2.7, Percentage 80-82.``
    """
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    if len(cells) < 3:
        return [" ".join(r) for r in rows]
    header, body = cells[0], cells[2:]  # cells[1] is the |---| separator
    lines = []
    for row in body:
        rest = ", ".join(f"{h} {c}" for h, c in zip(header[1:], row[1:]))
        lines.append(f"{header[0]} {row[0]}: {rest}." if rest else f"{header[0]} {row[0]}.")
    return lines


def _clean_block(lines: list[str]) -> str:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            out.extend(_linearize_table(lines[i:j]))
            i = j
            continue
        line = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", line)  # list markers
        line = re.sub(r"^#{3,}\s*", "", line)  # sub-headings become plain lines
        line = strip_markdown(line)
        if line:
            out.append(line)
        i += 1
    return "\n".join(out)


def parse_markdown(path: Path) -> list[Section]:
    meta, body = _split_front_matter(path.read_text(encoding="utf-8"))
    doc_id = path.stem
    title = meta.get("title") or doc_id.replace("_", " ").title()
    sections: list[Section] = []
    heading, buf = "Overview", []

    def flush() -> None:
        text = _clean_block(buf)
        if text.strip():
            sections.append(Section(doc_id, title, heading, text))

    for line in body.splitlines():
        if re.match(r"^#\s+", line):  # document title line
            title = meta.get("title") or line.lstrip("# ").strip()
        elif re.match(r"^##\s+", line):
            flush()
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    flush()
    return sections


def parse_text(path: Path) -> list[Section]:
    text = _clean_block(path.read_text(encoding="utf-8").splitlines())
    title = path.stem.replace("_", " ").title()
    return [Section(path.stem, title, "Full text", text)] if text else []


def parse_pdf(path: Path) -> list[Section]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Reading PDFs needs `pip install pypdf`.") from exc
    title = path.stem.replace("_", " ").title()
    sections = []
    for n, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = _clean_block((page.extract_text() or "").splitlines())
        if text:
            sections.append(Section(path.stem, title, f"Page {n}", text))
    return sections


def load_corpus(path: str | Path) -> list[Section]:
    """Load every .md / .txt / .pdf file under ``path`` (a directory or a single file)."""
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    parsers = {".md": parse_markdown, ".txt": parse_text, ".pdf": parse_pdf}
    sections: list[Section] = []
    for f in files:
        parser = parsers.get(f.suffix.lower())
        if parser:
            sections.extend(parser(f))
    if not sections:
        raise FileNotFoundError(f"No .md/.txt/.pdf documents found under {root}")
    return sections


# --- chunking -----------------------------------------------------------------------------


def _words(s: str) -> int:
    return len(s.split())


def _chunk_structure(sections: list[Section], cfg: ChunkConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in sections:
        sents = split_sentences(sec.text)
        start = 0
        ordinal = 0
        while start < len(sents):
            end, words = start, 0
            while end < len(sents) and (end == start or words + _words(sents[end]) <= cfg.max_words):
                words += _words(sents[end])
                end += 1
            chunks.append(_make_chunk(sec, " ".join(sents[start:end]), ordinal))
            ordinal += 1
            if end >= len(sents):
                break
            start = max(end - cfg.overlap, start + 1)
    return chunks


def _chunk_fixed(sections: list[Section], cfg: ChunkConfig) -> list[Chunk]:
    """Naive baseline: slide a word window over each document, ignoring section boundaries."""
    chunks: list[Chunk] = []
    by_doc: dict[str, list[Section]] = {}
    for sec in sections:
        by_doc.setdefault(sec.doc_id, []).append(sec)
    stride = max(cfg.max_words - cfg.overlap, 1)
    for secs in by_doc.values():
        words: list[str] = []
        owner: list[int] = []  # index of the section each word came from
        for i, sec in enumerate(secs):
            w = sec.text.split()
            words.extend(w)
            owner.extend([i] * len(w))
        for ordinal, start in enumerate(range(0, len(words), stride)):
            window = words[start:start + cfg.max_words]
            chunks.append(_make_chunk(secs[owner[start]], " ".join(window), ordinal))
            if start + cfg.max_words >= len(words):
                break
    return chunks


def _make_chunk(sec: Section, text: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=f"{sec.section_id}:{ordinal}",
        doc_id=sec.doc_id,
        doc_title=sec.doc_title,
        heading=sec.heading,
        section_id=sec.section_id,
        text=text,
        ordinal=ordinal,
    )


def chunk_sections(sections: list[Section], cfg: ChunkConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    if cfg.strategy == "structure":
        return _chunk_structure(sections, cfg)
    if cfg.strategy == "fixed":
        return _chunk_fixed(sections, cfg)
    raise ValueError(f"Unknown chunking strategy: {cfg.strategy!r}")
