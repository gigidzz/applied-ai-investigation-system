"""Compare the two chunking strategies on retrieval quality.

Stage 2 compared them on their properties — chunk counts, stored words,
boundary behaviour — which says how they differ but not which retrieves
better. This answers that, by indexing each strategy into its own collection
and running the same questions against both.

The comparison is fair by construction: identical corpus, identical questions,
identical retrieval configurations, identical embedding model. The only thing
that differs is where the text was cut.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ragsys.config import PROJECT_ROOT
from ragsys.evaluation.dataset import load
from ragsys.evaluation.harness import Harness
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

CONFIGS = (DENSE_ONLY, SPARSE_ONLY, HYBRID_FUSION_ONLY, HYBRID_RERANKED)
HEADER = f"{'configuration':<16}{'R@5':>9}{'R@10':>9}{'R@15':>9}{'MRR':>8}{'all gold':>10}"


def row(label: str, stats: dict[str, Any]) -> str:
    return (
        f"{label:<16}{stats['recall@5']:>8.1%}{stats['recall@10']:>9.1%}"
        f"{stats['recall@15']:>9.1%}{stats['mrr']:>8.3f}{stats['all_gold_found']:>10.1%}"
    )


def build(strategy: str, collection: str) -> Harness:
    stats = SparseStats.load(PROJECT_ROOT / "data" / "processed" / f"sparse_stats_{strategy}.json")
    store = QdrantChunkStore(
        embedder=Embedder(), sparse_encoder=SparseEncoder(stats), collection=collection
    )
    return Harness(Retriever(store))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural", default="ragsys_chunks")
    parser.add_argument("--fixed", default="ragsys_chunks_fixed")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "eval" / "chunking.json"))
    args = parser.parse_args()

    setup_logging("ERROR")
    questions = load()
    report: dict[str, Any] = {"questions": len(questions), "strategies": {}}

    for strategy, collection in (("structural", args.structural), ("fixed", args.fixed)):
        harness = build(strategy, collection)
        print(f"\n{strategy.upper()}  ({collection}, {harness.retriever.store.count():,} chunks)")
        print(HEADER)
        print("-" * len(HEADER))
        report["strategies"][strategy] = {}
        for config in CONFIGS:
            result = harness.score_retrieval(questions, config)
            aggregate = result.aggregate()
            report["strategies"][strategy][config.name] = result.as_dict()
            print(row(config.name, aggregate))

    print("\n\nSIDE BY SIDE, best configuration for each")
    print(f"{'strategy':<16}{'R@5':>9}{'R@10':>9}{'R@15':>9}{'MRR':>8}{'all gold':>10}")
    print("-" * 61)
    for strategy in ("structural", "fixed"):
        best = max(
            report["strategies"][strategy].values(),
            key=lambda r: r["overall"]["recall@15"],
        )
        print(row(f"{strategy} ({best['config'][:9]})", best["overall"]))

    # Authored and HotpotQA separated, because a chunking strategy may suit
    # terse operational documents and encyclopedic prose differently.
    print("\nBY SOURCE, hybrid+rerank")
    print(f"{'strategy / group':<16}{'R@5':>9}{'R@10':>9}{'R@15':>9}{'MRR':>8}{'all gold':>10}")
    print("-" * 61)
    for strategy in ("structural", "fixed"):
        entry = report["strategies"][strategy]["hybrid+rerank"]
        for group, stats in sorted(entry["by_group"].items()):
            print(row(f"{strategy[:9]} / {group[:6]}", stats))

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
