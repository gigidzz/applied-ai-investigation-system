"""Two strategies for cutting a document into retrievable pieces.

Why cut at all. A record has to be small enough that its embedding means one
thing — a 1,378-word paragraph embeds to a vector that is near the centre of
the space and therefore close to everything and useful for nothing — and small
enough that fifteen of them fit in a prompt.

Why two strategies, and why they differ in *where* they cut rather than in how
much they take: the corpus was measured before this module was written. The
median record is 78 words and the 95th percentile is 177, so a 300-word chunk
would split 0.5% of records and a comparison of two chunk *sizes* would be a
comparison of two no-ops. See docs/CORPUS_PROFILE.md.

    structural  Respect the document's own boundaries, falling back through a
                hierarchy: blank lines first, then sentence ends, then words
                only as a last resort. Never cuts mid-sentence, and carries no
                overlap. Chunks vary in length, but each is a whole number of
                sentences, and each knows which markdown section it came from.

                The hierarchy matters because formats differ in what structure
                survives extraction. Markdown keeps its blank lines; text
                pulled out of a PDF has none at all, because the extractor
                flattens the layout. Splitting only on blank lines would make
                this strategy identical to `fixed` on every PDF in the
                corpus — which it was, until the sentence level was added.

    fixed       Ignore boundaries. Uniform windows of `size` words advancing
                by `size - overlap`, so neighbours share text. Chunks are
                uniform and cheap to reason about, but a sentence — or a fact
                — can be cut in half. The overlap is what prevents a fact on a
                boundary from being lost entirely.

Both return Chunks, so nothing downstream knows or cares which was used. The
strategy is recorded on every chunk, which is what lets the evaluation harness
compare them over one corpus without ingesting it twice.
"""

from __future__ import annotations

import re

from ragsys.config import settings
from ragsys.ingestion.models import Chunk, RawDocument, make_chunk

STRATEGIES = ("structural", "fixed")

# A blank line, possibly with trailing spaces, separates blocks.
_BLOCK_BREAK = re.compile(r"\n[ \t]*\n+")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class UnknownStrategyError(ValueError):
    """Raised for a chunking strategy that does not exist."""


def _words(text: str) -> list[str]:
    return text.split()


def fixed_windows(text: str, size: int, overlap: int) -> list[str]:
    """Uniform overlapping windows of `size` words.

    The caller guarantees overlap < size; the configuration layer validates it,
    because a step of zero or less would never advance and loop forever.
    """
    words = _words(text)
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]

    step = size - overlap
    windows: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + size]
        windows.append(" ".join(window))
        if start + size >= len(words):
            break
        start += step
    return windows


def _blocks_with_sections(text: str) -> list[tuple[str, str | None]]:
    """Split into (block, enclosing section heading) pairs.

    A heading is never dropped. Headings are content: "Connection pool sizing"
    is exactly the phrase a user would search for, and discarding it loses both
    those words and the chunk's ability to say what it is about.

    A heading sitting alone between two blank lines is therefore held and
    prepended to the block that follows it, rather than emitted as a block of
    its own. That matters because several sections can be packed into one
    chunk, and attaching the heading only as metadata would silently lose every
    heading except the first.
    """
    pairs: list[tuple[str, str | None]] = []
    section: str | None = None
    pending: str | None = None

    for raw_block in _BLOCK_BREAK.split(text):
        block = raw_block.strip()
        if not block:
            continue

        lines = block.splitlines()
        heading_match = _HEADING.match(lines[0])

        if heading_match:
            section = heading_match.group(2)
            remainder = "\n".join(lines[1:]).strip()
            if remainder:
                pairs.append((f"{section}\n\n{remainder}", section))
                pending = None
            else:
                # Hold it for the next block, so its words are not lost.
                pending = f"{pending}\n\n{section}" if pending else section
            continue

        if pending:
            block = f"{pending}\n\n{block}"
            pending = None
        pairs.append((block, section))

    if pending:
        # A trailing heading with nothing beneath it.
        pairs.append((pending, section))

    return pairs


# Split after sentence-ending punctuation followed by whitespace. Version
# numbers like 'pay-gw-4.2.1' are unaffected, having no space after the dot.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _pack(units: list[str], size: int, joiner: str) -> list[str]:
    """Greedily group units into pieces of at most `size` words."""
    pieces: list[str] = []
    buffer: list[str] = []
    buffer_words = 0

    for unit in units:
        unit_words = len(_words(unit))
        if buffer and buffer_words + unit_words > size:
            pieces.append(joiner.join(buffer))
            buffer, buffer_words = [], 0
        buffer.append(unit)
        buffer_words += unit_words

    if buffer:
        pieces.append(joiner.join(buffer))
    return pieces


def split_oversized_block(block: str, size: int, overlap: int) -> list[str]:
    """Break a block that is too large, at the largest boundary available.

    Sentence ends first. Only a single sentence longer than the limit falls
    through to word windows, which is the one case where this strategy cuts
    mid-sentence.
    """
    sentences = [s.strip() for s in _SENTENCE_END.split(block) if s.strip()]

    if len(sentences) <= 1:
        return fixed_windows(block, size, overlap)

    pieces: list[str] = []
    for piece in _pack(sentences, size, " "):
        if len(_words(piece)) > size:
            pieces.extend(fixed_windows(piece, size, overlap))
        else:
            pieces.append(piece)
    return pieces


def structural_blocks(text: str, size: int, overlap: int) -> list[tuple[str, str | None]]:
    """Pack whole blocks up to `size` words, splitting only oversized blocks."""
    packed: list[tuple[str, str | None]] = []
    buffer: list[str] = []
    buffer_words = 0
    buffer_section: str | None = None

    def flush() -> None:
        """Emit the buffered blocks as one chunk, led by their heading.

        The heading is prepended rather than discarded. It is content, not
        markup: "Connection pool sizing" is exactly the phrase a user would
        search for, and dropping it both loses those words and leaves the
        chunk unable to say what it is about. A section spanning several
        chunks repeats its heading on each, which is deliberate — every chunk
        has to stand alone once retrieved.
        """
        nonlocal buffer, buffer_words
        if not buffer:
            return
        body = "\n\n".join(buffer)
        if buffer_section and not body.startswith(buffer_section):
            body = f"{buffer_section}\n\n{body}"
        packed.append((body, buffer_section))
        buffer = []
        buffer_words = 0

    for block, section in _blocks_with_sections(text):
        block_words = len(_words(block))

        if block_words > size:
            # Too big to pack with anything. Emit what we have, then split the
            # block at the largest boundary inside it.
            flush()
            for piece in split_oversized_block(block, size, overlap):
                packed.append((piece, section))
            buffer_section = section
            continue

        if buffer and buffer_words + block_words > size:
            flush()

        if not buffer:
            buffer_section = section
        buffer.append(block)
        buffer_words += block_words

    flush()
    return packed


def chunk_document(
    doc: RawDocument,
    strategy: str | None = None,
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Cut one document into Chunks using the named strategy.

    Defaults come from configuration, so a caller that does not care gets the
    values chosen from the corpus measurements.
    """
    strategy = strategy or settings.chunk_strategy
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap

    if strategy not in STRATEGIES:
        raise UnknownStrategyError(
            f"unknown chunking strategy {strategy!r}; expected one of {STRATEGIES}"
        )
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    text = doc.text.strip()
    if not text:
        return []

    if strategy == "fixed":
        pieces: list[tuple[str, str | None]] = [
            (window, None) for window in fixed_windows(text, size, overlap)
        ]
    else:
        pieces = structural_blocks(text, size, overlap)

    chunks: list[Chunk] = []
    for index, (piece, section) in enumerate(pieces):
        extra: dict[str, object] = {"chunk_count_hint": len(pieces)}
        if section:
            extra["section"] = section
        chunks.append(make_chunk(doc, piece, index, strategy, extra))
    return chunks
