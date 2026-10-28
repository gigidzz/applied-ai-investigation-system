"""Query the live index and print what comes back.

A test proves a component behaves as specified. This proves the index exists,
holds what it should, and returns sensible documents for real questions —
which is a different claim, and the one worth seeing with your own eyes.

The default queries are chosen to show dense and sparse retrieval failing in
opposite directions, which is the reason both are stored.
"""

from __future__ import annotations

import argparse

from ragsys.config import PROJECT_ROOT, settings
from ragsys.indexing.embedder import Embedder
from ragsys.indexing.sparse import SparseEncoder, SparseStats
from ragsys.indexing.vector_store import QdrantChunkStore
from ragsys.logging import setup_logging

# Each pair is (question, what it is meant to expose).
DEFAULT_QUERIES = [
    ("pay-gw-4.2.1", "an exact release identifier: keyword search should win"),
    (
        "why did card authorisation start failing",
        "pure paraphrase, no shared words with the source: meaning search should win",
    ),
    (
        "which services depend on the payment gateway",
        "a fact stored only in a JSON dependency list",
    ),
    ("when did the rollback finish", "the planted contradiction: two sources disagree"),
]


def show(label: str, hits: list, width: int = 86) -> None:
    print(f"  {label}")
    if not hits:
        print("      (nothing returned)")
        return
    for rank, hit in enumerate(hits, start=1):
        text = " ".join(hit.text.split())
        print(f"      {rank}. score={hit.score:7.3f}  {hit.source_path}")
        print(f"         {text[:width]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--strategy", default=settings.chunk_strategy)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("query", nargs="*", help="ask your own question instead")
    args = parser.parse_args()

    setup_logging("WARNING")

    stats_file = PROJECT_ROOT / "data" / "processed" / f"sparse_stats_{args.strategy}.json"
    stats = SparseStats.load(stats_file)
    store = QdrantChunkStore(
        embedder=Embedder(),
        sparse_encoder=SparseEncoder(stats),
        collection=args.collection,
    )

    total = store.count()
    print(f"\ncollection : {args.collection}")
    print(f"points     : {total:,}")
    print(f"vocabulary : {len(stats.vocabulary):,} terms\n")
    if total == 0:
        raise SystemExit("Collection is empty. Run scripts/build_index.py first.")

    asked = " ".join(args.query)
    pairs = [(asked, "your question")] if asked else DEFAULT_QUERIES

    for question, why in pairs:
        print("=" * 100)
        print(f"Q: {question}")
        print(f"   ({why})\n")
        show("MEANING  (dense embedding)", store.search_dense(question, limit=args.limit))
        print()
        show("KEYWORD  (BM25 sparse)", store.search_sparse(question, limit=args.limit))
        print()


if __name__ == "__main__":
    main()
