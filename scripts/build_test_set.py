"""Assemble the combined evaluation question set into one file.

The count has to be visible in one place. Authored questions and HotpotQA
questions live in separate files because they were produced by different
processes, but "how many test questions are there" should not require adding
two numbers and explaining a sampling rule.
"""

from __future__ import annotations

import argparse

from ragsys.evaluation.dataset import (
    COMBINED_FILE,
    DEFAULT_HOTPOT_SAMPLE,
    SEED,
    build,
    save,
    summarise,
)
from ragsys.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotpot", type=int, default=DEFAULT_HOTPOT_SAMPLE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    setup_logging("WARNING")
    questions = build(hotpot_count=args.hotpot, seed=args.seed)
    save(questions)

    target = COMBINED_FILE.relative_to(COMBINED_FILE.parents[2])
    print(f"\n{len(questions)} questions -> {target}\n")
    for category, count in summarise(questions).items():
        print(f"  {count:3d}  {category}")
    authored = sum(1 for q in questions if q.is_authored)
    sampled = len(questions) - authored
    print(f"\n  {authored} authored, {sampled} sampled from HotpotQA (seed {args.seed})")
    print(f"  {sum(1 for q in questions if q.unanswerable)} expect an abstention")
    contradictions = sum(1 for q in questions if q.contradictory)
    print(f"  {contradictions} expect a contradiction to be reported\n")


if __name__ == "__main__":
    main()
