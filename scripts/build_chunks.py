"""Run the ingestion pipeline over the whole corpus.

Reads every supported file under data/corpus/, extracts text and metadata,
cuts each document with both chunking strategies, and streams the result to
data/processed/chunks.jsonl — the input Stage 3 will embed.

Both strategies are stored side by side, tagged on every chunk, so the
evaluation harness can compare them without ingesting the corpus twice.

Usage:
    uv run python scripts/build_chunks.py
    uv run python scripts/build_chunks.py --one data/corpus/ops/incidents/INC-2026-03-14.pdf
    uv run python scripts/build_chunks.py --limit-files 50
    uv run python scripts/build_chunks.py --strategies structural
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from ragsys.config import PROJECT_ROOT, settings
from ragsys.ingestion.chunking import STRATEGIES, chunk_document
from ragsys.ingestion.dedupe import ContentIndex
from ragsys.ingestion.loaders import load_any
from ragsys.ingestion.metadata import enrich
from ragsys.ingestion.pipeline import IngestSummary, ingest_file, iter_corpus_files
from ragsys.ingestion.stores import JsonlChunkStore
from ragsys.logging import get_logger, setup_logging

log = get_logger("scripts.build_chunks")

# Documentation and manifests sit inside the corpus tree but are not corpus.
SKIP_NAMES = ("DESIGN.md", "MANIFEST.json", "PROFILE.md", "CORPUS_PROFILE.md")


def inspect_one(path: Path, strategies: list[str]) -> int:
    """Print what the pipeline makes of a single file, without storing anything."""
    documents = [enrich(doc) for doc in load_any(path)]
    print(f"\n{path}")
    print(f"  {len(documents)} document(s)\n")

    for doc in documents[:3]:
        print(f"  doc_id    {doc.doc_id}")
        print(f"  format    {doc.fmt}")
        print(f"  words     {doc.word_count}")
        shown = {k: v for k, v in doc.metadata.items() if k != "source_path"}
        print(f"  metadata  {json.dumps(shown, default=str)[:200]}")
        for strategy in strategies:
            chunks = chunk_document(doc, strategy=strategy)
            total = sum(c.word_count for c in chunks)
            print(f"\n  strategy={strategy} -> {len(chunks)} chunks, {total} words")
            for chunk in chunks:
                section = chunk.metadata.get("section")
                label = f"  section={section!r}" if section else ""
                print(f"    [{chunk.index}] {chunk.word_count:>4}w{label}")
                print(f"         {chunk.text[:92].replace(chr(10), ' / ')!r}")
        print()

    if len(documents) > 3:
        print(f"  ... and {len(documents) - 3} more documents in this file")
    return 0


def build(strategies: list[str], limit_files: int | None, out_path: Path) -> int:
    summary = IngestSummary()
    index = ContentIndex()

    paths = list(iter_corpus_files(settings.corpus_dir, skip=SKIP_NAMES))
    if limit_files is not None:
        paths = paths[:limit_files]
    log.info("%d files to ingest from %s", len(paths), settings.corpus_dir)

    with JsonlChunkStore(out_path) as store:
        for path in paths:
            summary.files_seen += 1
            try:
                results = ingest_file(path, store, index, strategies, summary)
            except Exception as exc:  # noqa: BLE001 - never abort on one file
                summary.files_failed += 1
                summary.failures.append((str(path), f"{type(exc).__name__}: {exc}"))
                log.warning("failed on %s: %s", path, exc)
                continue
            for result in results:
                summary.record(result)
            log.info(
                "%-58s %6d docs -> %7d chunks",
                path.name[:58],
                len(results),
                summary.chunks_written,
            )

    report = analyse(out_path, strategies)
    manifest = {
        "corpus_dir": str(settings.corpus_dir.relative_to(PROJECT_ROOT)),
        "strategies": strategies,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "output": str(out_path.relative_to(PROJECT_ROOT)),
        "output_bytes": out_path.stat().st_size,
        **summary.as_dict(),
        "per_strategy": report,
    }
    manifest_path = out_path.parent / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    log.info("wrote %s", out_path)
    log.info("wrote %s", manifest_path)
    print("\n" + json.dumps(manifest, indent=2))
    return 0 if summary.chunks_written else 1


def analyse(path: Path, strategies: list[str]) -> dict[str, object]:
    """Compare the strategies over the file we just wrote."""
    lengths: dict[str, list[int]] = defaultdict(list)
    per_doc: dict[str, Counter[str]] = defaultdict(Counter)

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            strategy = record["strategy"]
            lengths[strategy].append(record["word_count"])
            per_doc[strategy][record["doc_id"]] += 1

    out: dict[str, object] = {}
    for strategy in strategies:
        values = sorted(lengths[strategy])
        if not values:
            continue
        counts = list(per_doc[strategy].values())
        out[strategy] = {
            "chunks": len(values),
            "documents": len(counts),
            "chunks_per_document": round(len(values) / len(counts), 3),
            "documents_split": sum(1 for c in counts if c > 1),
            "words_total": sum(values),
            "words": {
                "min": values[0],
                "median": int(statistics.median(values)),
                "p95": values[int(len(values) * 0.95)],
                "max": values[-1],
                "mean": round(statistics.mean(values), 1),
            },
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", type=Path, default=None, help="inspect a single file")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument(
        "--out", type=Path, default=settings.data_dir / "processed" / "chunks.jsonl"
    )
    args = parser.parse_args()

    setup_logging()
    if args.one is not None:
        return inspect_one(args.one, args.strategies)
    return build(args.strategies, args.limit_files, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
