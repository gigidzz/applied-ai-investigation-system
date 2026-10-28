"""Run the evaluation and print the report.

Retrieval is scored for every configuration by default, because it is cheap
and comparing configurations is a requirement. Answer scoring is opt-in with
`--answers`, because it costs one generation call and one judge call per
question and a free tier has a daily quota.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ragsys.config import PROJECT_ROOT, settings
from ragsys.evaluation.dataset import load
from ragsys.evaluation.harness import Harness
from ragsys.evaluation.judge import Judge
from ragsys.indexing.embedder import Embedder
from ragsys.indexing.sparse import SparseEncoder, SparseStats
from ragsys.indexing.vector_store import QdrantChunkStore
from ragsys.logging import setup_logging
from ragsys.reasoning.agent import Agent
from ragsys.retrieval.retriever import (
    DENSE_ONLY,
    HYBRID_FUSION_ONLY,
    HYBRID_RERANKED,
    SPARSE_ONLY,
    Retriever,
)

CONFIGS = {c.name: c for c in (DENSE_ONLY, SPARSE_ONLY, HYBRID_FUSION_ONLY, HYBRID_RERANKED)}
REPORT_DIR = PROJECT_ROOT / "data" / "eval"


def row(label: str, stats: dict[str, Any]) -> str:
    if not stats:
        return f"{label:<16} (no questions)"
    return (
        f"{label:<16}{stats['recall@5']:>9.1%}{stats['recall@10']:>10.1%}"
        f"{stats['recall@15']:>10.1%}{stats['mrr']:>8.3f}"
        f"{stats['precision@15']:>8.1%}{stats['all_gold_found']:>10.1%}"
        f"{stats['distinct_documents']:>8.1f}"
    )


HEADER = (
    f"{'configuration':<16}{'R@5':>9}{'R@10':>10}{'R@15':>10}"
    f"{'MRR':>8}{'P@15':>8}{'all gold':>10}{'docs':>8}"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--strategy", default=settings.chunk_strategy)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--answers", action="store_true", help="also score answer quality")
    parser.add_argument("--answer-config", default=HYBRID_RERANKED.name, choices=list(CONFIGS))
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "report.json")
    parser.add_argument(
        "--progress",
        type=Path,
        default=REPORT_DIR / "answer_progress.jsonl",
        help="append each scored answer here, and resume from it",
    )
    parser.add_argument("--fresh", action="store_true", help="ignore and overwrite saved progress")
    args = parser.parse_args()

    setup_logging("WARNING" if not args.answers else "INFO")
    questions = load()
    if args.limit:
        questions = questions[: args.limit]

    stats = SparseStats.load(
        PROJECT_ROOT / "data" / "processed" / f"sparse_stats_{args.strategy}.json"
    )
    store = QdrantChunkStore(
        embedder=Embedder(), sparse_encoder=SparseEncoder(stats), collection=args.collection
    )
    harness = Harness(Retriever(store))

    print(f"\n{len(questions)} questions, collection {args.collection}\n")
    report: dict[str, Any] = {"questions": len(questions), "retrieval": {}, "answers": None}

    print("RETRIEVAL")
    print(HEADER)
    print("-" * len(HEADER))
    retrieval_reports = {}
    for name in args.configs:
        result = harness.score_retrieval(questions, CONFIGS[name])
        retrieval_reports[name] = result
        report["retrieval"][name] = result.as_dict()
        print(row(name, result.aggregate()))

    print("\nBEST CONFIGURATION, SPLIT BY SOURCE  (never averaged together)")
    best = max(retrieval_reports.values(), key=lambda r: r.aggregate()["recall@15"])
    print(HEADER)
    print("-" * len(HEADER))
    for group, scores in sorted(best.groups.items()):
        print(row(group, best.aggregate(scores)))

    print(f"\n{best.config_name} BY CATEGORY")
    print(HEADER)
    print("-" * len(HEADER))
    for category, agg in best.by_category().items():
        print(row(category[:15], agg))

    if args.answers:
        print("\nANSWERS  (one generation call and one judge call per question)")
        agent = Agent(Retriever(store))
        judge = Judge(generator_model=getattr(agent.client, "model", None))
        if args.fresh and args.progress.exists():
            args.progress.unlink()
        answers = harness.score_answers(
            questions,
            agent,
            judge,
            config_name=args.answer_config,
            progress=args.progress,
        )
        report["answers"] = answers.as_dict()
        overall = answers.aggregate()
        print(f"\n  generator : {answers.generator_model}")
        print(f"  judge     : {answers.judge_model}")
        print(f"  verdicts  : {overall['verdicts']}")
        for key in (
            "correctness",
            "grounded",
            "citation_validity",
            "citation_relevance",
            "abstention_recall",
            "abstention_precision",
            "contradiction_detection",
        ):
            value = overall.get(key)
            print(f"  {key:<24} {'n/a' if value is None else f'{value:.1%}'}")
        print(f"  model_failures           {overall['model_failures']}")
        print(f"  mean tokens per question {overall['mean_tokens']:.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}\n")


if __name__ == "__main__":
    main()
