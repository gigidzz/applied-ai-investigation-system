"""Compare retrieval configurations on the authored operations questions.

This is a preview of the evaluation harness, not a substitute for it: 23
questions over documents we wrote ourselves is too small a sample to tune
against, and the ranking weights were deliberately set from reasoning rather
than fitted to these numbers. What it answers is narrower and still worth
knowing — whether fusion and the ranking signals move the gold documents into
the result set at all, and whether they repair the specific query that both
retrieval routes got wrong.

Recall here is document recall at the final cut: of the documents an answer
needs, how many does retrieval actually deliver.
"""

from __future__ import annotations

import argparse
import json

from ragsys.config import PROJECT_ROOT, settings
from ragsys.indexing.embedder import Embedder
from ragsys.indexing.sparse import SparseEncoder, SparseStats
from ragsys.indexing.vector_store import QdrantChunkStore
from ragsys.logging import setup_logging
from ragsys.retrieval.retriever import (
    DENSE_ONLY,
    HYBRID_FUSION_ONLY,
    HYBRID_RERANKED,
    SPARSE_ONLY,
    Retriever,
)

QUESTIONS = PROJECT_ROOT / "data" / "eval" / "ops_questions.jsonl"
CONFIGS = [DENSE_ONLY, SPARSE_ONLY, HYBRID_FUSION_ONLY, HYBRID_RERANKED]

# The query both routes failed on, kept as a named case because it is the
# reason the document-type signal exists.
WITNESS = "when did the rollback finish"


def gold_matched(gold: str, retrieved: list[str]) -> bool:
    """Gold paths omit the corpus prefix, so compare by suffix."""
    return any(path.endswith(gold) for path in retrieved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--strategy", default=settings.chunk_strategy)
    args = parser.parse_args()

    setup_logging("ERROR")
    stats = SparseStats.load(
        PROJECT_ROOT / "data" / "processed" / f"sparse_stats_{args.strategy}.json"
    )
    store = QdrantChunkStore(
        embedder=Embedder(),
        sparse_encoder=SparseEncoder(stats),
        collection=args.collection,
    )
    retriever = Retriever(store)

    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    scored = [q for q in questions if q.get("gold_documents")]
    print(f"\n{len(questions)} questions, {len(scored)} with gold documents\n")

    rows = []
    for config in CONFIGS:
        found = 0
        possible = 0
        complete = 0
        distinct_docs = 0
        by_category: dict[str, list[float]] = {}
        for question in scored:
            result = retriever.retrieve(question["question"], config)
            retrieved = [c.hit.source_path for c in result.candidates]
            distinct_docs += len(set(retrieved))
            gold = question["gold_documents"]
            hits = sum(1 for g in gold if gold_matched(g, retrieved))
            found += hits
            possible += len(gold)
            if hits == len(gold):
                complete += 1
            by_category.setdefault(question["category"], []).append(hits / len(gold))
        rows.append(
            {
                "name": config.name,
                "recall": found / possible,
                "complete": complete / len(scored),
                "docs": distinct_docs / len(scored),
                "by_category": {k: sum(v) / len(v) for k, v in by_category.items()},
            }
        )

    print(f"{'configuration':<16} {'doc recall':>11} {'all gold found':>15} {'distinct docs':>14}")
    print("-" * 60)
    for row in rows:
        print(
            f"{row['name']:<16} {row['recall']:>10.1%} {row['complete']:>15.1%} "
            f"{row['docs']:>14.1f}"
        )

    categories = sorted({c for row in rows for c in row["by_category"]})
    print(f"\n{'configuration':<16}" + "".join(f"{c[:16]:>18}" for c in categories))
    print("-" * (16 + 18 * len(categories)))
    for row in rows:
        cells = "".join(f"{row['by_category'].get(c, 0.0):>17.0%} " for c in categories)
        print(f"{row['name']:<16}{cells}")

    print(f"\n{'=' * 78}\nThe query both routes failed on: {WITNESS!r}\n")
    for config in CONFIGS:
        result = retriever.retrieve(WITNESS, config)
        print(f"  {config.name}  (question read as: {result.kind.value})")
        for position, candidate in enumerate(result.candidates[:4], start=1):
            marker = (
                "  <-- gold"
                if (
                    "INC-2026-03-14" in candidate.hit.source_path
                    or "THREAD-2026-03-14" in candidate.hit.source_path
                )
                else ""
            )
            print(f"      {position}. {candidate.hit.source_path}{marker}")
        print()


if __name__ == "__main__":
    main()
