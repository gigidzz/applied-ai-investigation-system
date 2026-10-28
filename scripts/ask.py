"""Ask the system a question from the command line.

The same Agent the web interface will call, so what is printed here is what
the deployed service returns. Exists mainly so retrieval and reasoning can be
inspected together on real questions before there is a user interface.
"""

from __future__ import annotations

import argparse
import json

from ragsys.config import PROJECT_ROOT, settings
from ragsys.indexing.embedder import Embedder
from ragsys.indexing.sparse import SparseEncoder, SparseStats
from ragsys.indexing.vector_store import QdrantChunkStore
from ragsys.logging import setup_logging
from ragsys.reasoning.agent import Agent, AgentConfig
from ragsys.reasoning.llm import DEFAULT_GEMINI_MODELS
from ragsys.retrieval.retriever import Retriever


def build_agent(strategy: str, collection: str, rounds: int) -> Agent:
    stats = SparseStats.load(PROJECT_ROOT / "data" / "processed" / f"sparse_stats_{strategy}.json")
    store = QdrantChunkStore(
        embedder=Embedder(),
        sparse_encoder=SparseEncoder(stats),
        collection=collection,
    )
    return Agent(Retriever(store), config=AgentConfig(max_tool_rounds=rounds))


def show(answer) -> None:
    print("=" * 94)
    print(f"Q: {answer.question}")
    print("=" * 94)
    if answer.failed:
        print(f"\n  MODEL FAILURE: {answer.error}")
    if answer.abstained:
        print(f"\n  ABSTAINED: {answer.abstain_reason or '(no reason given)'}")
    if answer.text:
        print(f"\n{answer.text}\n")
    if answer.contradictions:
        print("  CONTRADICTIONS REPORTED")
        for item in answer.contradictions:
            print(f"    - {item.description}  {list(item.labels)}")
        print()
    if answer.citations:
        print("  SOURCES CITED")
        for citation in answer.citations:
            print(f"    {citation.label:<4} {citation.source_path}")
        print()
    if answer.invalid_labels:
        print(f"  INVALID CITATIONS: {list(answer.invalid_labels)}")
    if answer.steps:
        print("  STEPS")
        for step in answer.steps:
            print(f"    - {step}")
        print()
    print(
        f"  grounded={answer.is_grounded}  confidence={answer.confidence}  "
        f"sources={answer.context_size}  tool_calls={answer.tool_calls}  "
        f"tokens={answer.prompt_tokens}+{answer.output_tokens}"
    )
    requested = settings.llm_model or DEFAULT_GEMINI_MODELS[0]
    if answer.model and answer.model != requested:
        print(f"  answered by {answer.model}, not {requested}: earlier models were unavailable")
    elif answer.model:
        print(f"  model: {answer.model}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--strategy", default=settings.chunk_strategy)
    parser.add_argument("--rounds", type=int, default=2, help="max tool-calling rounds")
    parser.add_argument("--json", action="store_true", help="print the raw answer object")
    args = parser.parse_args()

    setup_logging("WARNING")
    question = " ".join(args.question).strip()
    if not question:
        raise SystemExit("ask.py 'your question here'")

    answer = build_agent(args.strategy, args.collection, args.rounds).answer(question)
    if args.json:
        print(json.dumps(answer.as_dict(), indent=2, ensure_ascii=False))
    else:
        show(answer)


if __name__ == "__main__":
    main()
