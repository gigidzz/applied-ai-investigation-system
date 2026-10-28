"""Embed the chunked corpus and load it into the vector store.

Two passes over the chunk file, for a reason that is not optional. BM25 needs
corpus-wide document frequencies before any single chunk can be weighted, and
those statistics must be identical when indexing and when querying. So the
first pass only counts, writes the statistics to disk, and the second pass
encodes against them.

The expensive pass is resumable. Embedding the full corpus is roughly 25
minutes of CPU, and losing that to a dropped connection at chunk 60,000 is not
an acceptable failure mode: progress is recorded per batch and `--resume`
continues from the last completed one.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ragsys.config import PROJECT_ROOT, settings
from ragsys.indexing.embedder import Embedder
from ragsys.indexing.sparse import SparseEncoder, SparseStats, StatsBuilder
from ragsys.indexing.vector_store import QdrantChunkStore
from ragsys.ingestion.models import Chunk
from ragsys.logging import get_logger, setup_logging

log = get_logger(__name__)

DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
PROCESSED = PROJECT_ROOT / "data" / "processed"


def stats_path(strategy: str) -> Path:
    return PROCESSED / f"sparse_stats_{strategy}.json"


def progress_path(collection: str) -> Path:
    return PROCESSED / f"index_progress_{collection}.json"


def record_to_chunk(record: dict[str, Any]) -> Chunk:
    """Rebuild a Chunk from its flattened JSONL form."""
    metadata = {k[5:]: v for k, v in record.items() if k.startswith("meta_")}
    return Chunk(
        chunk_id=record["chunk_id"],
        doc_id=record["doc_id"],
        text=record["text"],
        index=int(record["index"]),
        strategy=record["strategy"],
        metadata=metadata,
    )


def iter_records(path: Path, strategy: str, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream matching records. The file is 131 MB; it is never held in memory."""
    emitted = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("strategy") != strategy:
                continue
            yield record
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def build_stats(path: Path, strategy: str, limit: int | None) -> SparseStats:
    builder = StatsBuilder()
    started = time.perf_counter()
    for count, record in enumerate(iter_records(path, strategy, limit), start=1):
        builder.add(record["text"])
        if count % 20_000 == 0:
            log.info("counted %d chunks", count)
    stats = builder.build()
    log.info(
        "statistics over %d chunks: %d distinct terms, mean length %.1f tokens (%.1fs)",
        stats.total_documents,
        len(stats.document_frequency),
        stats.average_length,
        time.perf_counter() - started,
    )
    return stats


def batches(records: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_progress(collection: str) -> int:
    path = progress_path(collection)
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8")).get("chunks_done", 0))


def save_progress(collection: str, done: int, strategy: str) -> None:
    progress_path(collection).write_text(
        json.dumps({"chunks_done": done, "strategy": strategy}), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--strategy", default=settings.chunk_strategy)
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--limit", type=int, default=None, help="index at most this many chunks")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.embedding_batch_size,
        help="texts per forward pass through the model",
    )
    parser.add_argument(
        "--upload-batch",
        type=int,
        default=256,
        help="chunks per upsert; amortises one network round trip",
    )
    parser.add_argument("--recreate", action="store_true", help="drop the collection first")
    parser.add_argument("--resume", action="store_true", help="continue from recorded progress")
    parser.add_argument("--stats-only", action="store_true", help="first pass only")
    parser.add_argument("--url", default=None, help="override QDRANT_URL")
    args = parser.parse_args()

    setup_logging()
    if not args.chunks.exists():
        raise SystemExit(f"{args.chunks} not found; run scripts/build_chunks.py first")

    target = stats_path(args.strategy)
    if args.resume and target.exists():
        stats = SparseStats.load(target)
        log.info("reusing statistics from %s", target.name)
    else:
        stats = build_stats(args.chunks, args.strategy, args.limit)
        stats.save(target)
        log.info("wrote %s (%.1f MB)", target.name, target.stat().st_size / 1e6)

    if args.stats_only:
        return

    store = QdrantChunkStore(
        embedder=Embedder(batch_size=args.batch_size),
        sparse_encoder=SparseEncoder(stats),
        collection=args.collection,
        url=args.url,
    )
    created = store.ensure_collection(recreate=args.recreate)
    log.info("collection %s %s", args.collection, "created" if created else "already present")

    skip = load_progress(args.collection) if args.resume else 0
    if skip:
        log.info("resuming after %d chunks", skip)

    total_expected = stats.total_documents if args.limit is None else args.limit
    done = skip
    written = 0
    started = time.perf_counter()
    stream = iter_records(args.chunks, args.strategy, args.limit)

    for batch_number, batch in enumerate(batches(stream, args.upload_batch), start=1):
        if done and batch_number * args.upload_batch <= skip:
            continue
        written += store.upsert([record_to_chunk(r) for r in batch])
        done += len(batch)
        save_progress(args.collection, done, args.strategy)
        if batch_number % 25 == 0:
            elapsed = time.perf_counter() - started
            rate = written / elapsed if elapsed else 0.0
            remaining = (total_expected - done) / rate if rate and total_expected else 0.0
            log.info(
                "%d/%s chunks, %.0f/s, %.1f min elapsed, ~%.1f min left",
                done,
                total_expected or "?",
                rate,
                elapsed / 60,
                remaining / 60,
            )

    elapsed = time.perf_counter() - started
    log.info(
        "indexed %d chunks in %.1f min (%.0f/s); collection now holds %d points",
        written,
        elapsed / 60,
        written / elapsed if elapsed else 0.0,
        store.count(),
    )


if __name__ == "__main__":
    main()
