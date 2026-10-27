"""Profile the corpora before designing anything that processes them.

Answers the questions that chunking, deduplication and metadata extraction
decisions depend on:

  * How long is a typical record, and how long is the tail?
  * At a given chunk size, how many records would actually be split? A chunk
    size larger than almost every record makes the chunker a no-op, which
    would render a comparison of chunking strategies meaningless.
  * How much exact and near duplication already exists?
  * Which metadata fields can be extracted reliably, and from how much of the
    corpus? Metadata that is mostly missing cannot be load-bearing in search.

Text extraction here is deliberately minimal — just enough to measure. The
real loaders, with metadata extraction and error handling, come next.

Usage:
    uv run python scripts/profile_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import docx
from pypdf import PdfReader

from ragsys.config import PROJECT_ROOT, settings
from ragsys.logging import get_logger, setup_logging

log = get_logger("scripts.profile_corpus")

CHUNK_SIZES_TO_TEST = (128, 256, 300, 512, 1024)
# A trailing \b would fail on "2026-03-14T14:20:00Z", because there is no
# word boundary between "4" and "T".
DATE_PATTERN = re.compile(r"(?<!\d)20\d{2}-\d{2}-\d{2}(?!\d)")


@dataclass
class Record:
    """One independent text record, whatever file it came from."""

    record_id: str
    source: str
    fmt: str
    path: str
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# minimal extraction
# --------------------------------------------------------------------------- #


def extract_pdf(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def extract_docx(path: Path) -> str:
    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs if p.text.strip())


def extract_json(path: Path) -> str:
    """Flatten a JSON object to readable text rather than dumping syntax."""
    data = json.loads(path.read_text(encoding="utf-8"))

    def walk(value: object, prefix: str = "") -> list[str]:
        if isinstance(value, dict):
            out: list[str] = []
            for key, item in value.items():
                out += walk(item, f"{prefix}{key}: " if not prefix else f"{prefix}{key}: ")
            return out
        if isinstance(value, list):
            return [f"{prefix}{', '.join(str(v) for v in value)}"] if value else []
        return [f"{prefix}{value}"]

    return "\n".join(walk(data))


def load_ops_records(root: Path) -> list[Record]:
    """One record per file, except the change log, which is one record per row."""
    records: list[Record] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"DESIGN.md", "MANIFEST.json"}:
            continue
        suffix = path.suffix.lstrip(".")
        rel = str(path.relative_to(root))
        try:
            if suffix == "pdf":
                text = extract_pdf(path)
            elif suffix == "docx":
                text = extract_docx(path)
            elif suffix == "json":
                text = extract_json(path)
            elif suffix == "csv":
                # Each deploy is an independent record, so the change log
                # contributes ~300 records rather than one.
                import csv as csv_module

                with path.open(encoding="utf-8") as handle:
                    for index, row in enumerate(csv_module.DictReader(handle)):
                        sentence = (
                            f"Release {row['release']} was deployed to {row['service']} "
                            f"at {row['deployed_at']} by {row['author']}. "
                            f"Risk {row['risk']}. "
                            f"Rolled back: {row['rolled_back']}. {row['summary']}"
                        )
                        records.append(Record(f"ops:{rel}#{index}", "ops", "csv", rel, sentence))
                continue
            else:
                text = path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - profiling must not abort on one file
            log.warning("could not read %s: %s", rel, exc)
            continue
        records.append(Record(f"ops:{rel}", "ops", suffix, rel, text.strip()))
    return records


def load_hotpot_records(path: Path, limit: int | None = None) -> list[Record]:
    records: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            row = json.loads(line)
            records.append(Record(row["doc_id"], "hotpot", "jsonl", path.name, row["text"]))
    return records


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def percentiles(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]

    return {
        "min": ordered[0],
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "p95": at(0.95),
        "max": ordered[-1],
        "mean": round(statistics.mean(ordered), 1),
    }


def shingles(text: str, size: int = 5) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i : i + size]) for i in range(max(0, len(words) - size + 1))}


def near_duplicate_pairs(records: list[Record], threshold: float = 0.5) -> list[tuple]:
    """Jaccard similarity over 5-word shingles. Quadratic, so ops corpus only."""
    prepared = [(r, shingles(r.text)) for r in records if r.words > 40]
    pairs = []
    for i in range(len(prepared)):
        a_rec, a_sh = prepared[i]
        if not a_sh:
            continue
        for j in range(i + 1, len(prepared)):
            b_rec, b_sh = prepared[j]
            if not b_sh:
                continue
            overlap = len(a_sh & b_sh)
            if not overlap:
                continue
            score = overlap / len(a_sh | b_sh)
            if score >= threshold:
                pairs.append((round(score, 3), a_rec.path, b_rec.path))
    return sorted(pairs, reverse=True)


def main() -> int:
    setup_logging()

    ops_root = settings.corpus_dir / "ops"
    hotpot_path = settings.corpus_dir / "hotpot" / "paragraphs.jsonl"

    log.info("loading ops corpus from %s", ops_root)
    ops = load_ops_records(ops_root)
    log.info("loading hotpot corpus from %s", hotpot_path)
    hotpot = load_hotpot_records(hotpot_path)
    everything = ops + hotpot

    lines: list[str] = ["# Corpus profile", ""]
    lines += [
        "Generated by `scripts/profile_corpus.py`. Chunking, deduplication and",
        "metadata decisions are made from these numbers.",
        "",
        "## Record counts",
        "",
        "A *record* is one independent piece of text. Most files contribute one",
        "record; the change log contributes one per row, because each deploy is",
        "independently meaningful.",
        "",
        "| Source | Format | Records | Words (total) |",
        "|---|---|---|---|",
    ]
    grouped: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in everything:
        grouped[(record.source, record.fmt)].append(record)
    for (source, fmt), group in sorted(grouped.items()):
        lines.append(f"| {source} | `.{fmt}` | {len(group):,} | {sum(r.words for r in group):,} |")
    lines += [
        f"| **total** | | **{len(everything):,}** | **{sum(r.words for r in everything):,}** |",
        "",
    ]

    lines += [
        "## Record length in words",
        "",
        "| Set | " + " | ".join(["min", "p25", "median", "p75", "p95", "max", "mean"]) + " |",
        "|---|" + "---|" * 7,
    ]
    for label, group in (("ops", ops), ("hotpot", hotpot), ("combined", everything)):
        stats = percentiles([r.words for r in group])
        lines.append(
            f"| {label} | "
            + " | ".join(
                str(stats[k]) for k in ("min", "p25", "median", "p75", "p95", "max", "mean")
            )
            + " |"
        )
    lines += [""]

    lines += [
        "## What a chunk size would actually do",
        "",
        "The share of records long enough to be split at each candidate size.",
        "A size that splits almost nothing makes the chunker a no-op.",
        "",
        "| Chunk size (words) | ops split | hotpot split | combined split |",
        "|---|---|---|---|",
    ]
    for size in CHUNK_SIZES_TO_TEST:
        row = [f"| {size} "]
        for group in (ops, hotpot, everything):
            longer = sum(1 for r in group if r.words > size)
            pct = 100 * longer / len(group) if group else 0
            row.append(f"| {longer:,} ({pct:.1f}%) ")
        lines.append("".join(row) + "|")
    lines += [""]

    exact = Counter(r.digest for r in everything)
    duplicated = {d: c for d, c in exact.items() if c > 1}
    lines += [
        "## Duplication",
        "",
        f"- Exact duplicate texts: **{len(duplicated)}** distinct texts appearing "
        f"more than once, covering {sum(duplicated.values())} records.",
    ]
    pairs = near_duplicate_pairs(ops)
    lines += [
        f"- Near-duplicate pairs in the ops corpus (Jaccard >= 0.5 over 5-word "
        f"shingles): **{len(pairs)}**.",
        "",
    ]
    if pairs:
        lines += ["| Similarity | Document A | Document B |", "|---|---|---|"]
        lines += [f"| {score} | `{a}` | `{b}` |" for score, a, b in pairs[:10]]
        lines += [""]

    with_date = sum(1 for r in everything if DATE_PATTERN.search(r.text))
    ops_with_date = sum(1 for r in ops if DATE_PATTERN.search(r.text))
    lines += [
        "## Metadata availability",
        "",
        f"- Records containing an ISO date: {with_date:,} of {len(everything):,} "
        f"({100 * with_date / len(everything):.1f}%).",
        f"- Within the ops corpus: {ops_with_date:,} of {len(ops):,} "
        f"({100 * ops_with_date / len(ops):.1f}%).",
        "- HotpotQA paragraphs carry a title and nothing else, so date filtering "
        "can only apply to the ops corpus and must never be a hard filter.",
        "",
    ]

    empty = [r.record_id for r in everything if r.words == 0]
    lines += [
        "## Integrity",
        "",
        f"- Records with no extractable text: {len(empty)}.",
        f"- Shortest record: {min(r.words for r in everything)} words.",
        f"- Longest record: {max(r.words for r in everything)} words.",
        "",
    ]

    # The point of profiling is the conclusions, not the tables.
    ops_median = percentiles([r.words for r in ops]).get("median", 0)
    hot_median = percentiles([r.words for r in hotpot]).get("median", 0)
    split_at_300 = sum(1 for r in everything if r.words > 300)
    split_at_160 = sum(1 for r in everything if r.words > 160)
    lines += [
        "## Design implications",
        "",
        "**Chunk size must come down, or chunking does nothing.** At 300 words "
        f"only {split_at_300:,} of {len(everything):,} records "
        f"({100 * split_at_300 / len(everything):.1f}%) are long enough to "
        "split. Comparing two chunking strategies at that size would compare "
        "two no-ops. At 160 words "
        f"{split_at_160:,} records ({100 * split_at_160 / len(everything):.1f}%) "
        "split, which is enough for the comparison to mean something. The "
        "defaults in `src/ragsys/config.py` are set from this measurement.",
        "",
        "**The strategy comparison should be about boundaries, not size.** The "
        f"median HotpotQA paragraph is {hot_median} words and the median ops "
        f"record is {ops_median}; most records are already a single coherent "
        "unit. The informative comparison is therefore *structural* chunking "
        "(respect existing paragraph and heading boundaries, split only the "
        "long tail) against *fixed* chunking (uniform windows with overlap, "
        "ignoring boundaries). Those differ in retrieval granularity even "
        "where record lengths are similar.",
        "",
        "**Exact-duplicate detection is required, and title identity is not "
        f"enough.** {len(duplicated)} distinct texts appear more than once "
        f"across {sum(duplicated.values())} records. The HotpotQA conversion "
        "deduplicated by title because the gold labels are title-based, which "
        "leaves identical bodies published under different titles. Ingestion "
        "must hash content, not just identifiers.",
        "",
        "**Date filtering cannot be a hard filter.** ISO dates appear in "
        f"{100 * ops_with_date / len(ops):.1f}% of ops records but only "
        f"{100 * with_date / len(everything):.1f}% of the corpus overall, "
        "because HotpotQA paragraphs carry a title and nothing else. A query "
        'like "in March" must therefore boost dated records rather than '
        "exclude undated ones, or it would discard the entire Wikipedia "
        "corpus.",
        "",
        "**The long tail is real but small.** The longest record is "
        f"{max(r.words for r in everything):,} words against a median of "
        f"{percentiles([r.words for r in everything])['median']}. Chunking has "
        "to handle it correctly, but it will not dominate throughput.",
        "",
    ]

    report_path = PROJECT_ROOT / "docs" / "CORPUS_PROFILE.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    log.info("wrote %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
