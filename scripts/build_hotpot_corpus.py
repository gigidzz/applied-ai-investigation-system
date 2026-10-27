"""Convert HotpotQA into a flat corpus plus a labelled question set.

HotpotQA ships one row per *question*, with the candidate paragraphs nested
inside each row. We need the inverse: one record per *paragraph*, because
paragraphs are the unit we retrieve. This script performs that inversion and
writes two files:

    data/corpus/hotpot/paragraphs.jsonl   one paragraph per line
    data/eval/hotpot_questions.jsonl      one question per line, with the
                                          titles of the paragraphs that
                                          genuinely support the answer

The second file is the retrieval ground truth. It is what makes Recall@K and
MRR computable without hand-labelling anything.

Subset choice, and why it is reproducible:

    config  'distractor' — each question carries 10 paragraphs (2 supporting,
            8 topically-similar distractors). The alternative, 'fullwiki',
            requires the whole Wikipedia dump: millions of paragraphs, far
            past a free-tier vector database.
    split   'validation' in full, all 7,405 questions, no sampling. A named
            split is byte-identical for everyone, with no random seed to get
            wrong. The train split would yield roughly 900,000 paragraphs and
            several hours of embedding.

If deduplication leaves fewer than --min-records unique paragraphs, the script
tops up from the train split taking questions in their original order, which
keeps the result deterministic.

Usage:
    uv run python scripts/build_hotpot_corpus.py
    uv run python scripts/build_hotpot_corpus.py --limit 50      # smoke test
    uv run python scripts/build_hotpot_corpus.py --revision <sha> # exact replay
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The macOS python.org build ships without a wired-up CA bundle, so plain HTTPS
# fails with CERTIFICATE_VERIFY_FAILED. truststore delegates to the operating
# system's own certificate store, which is already populated.
import truststore

truststore.inject_into_ssl()

from datasets import disable_progress_bars, load_dataset  # noqa: E402  (must follow the TLS fix)
from huggingface_hub import HfApi  # noqa: E402

from ragsys.config import settings  # noqa: E402
from ragsys.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("scripts.build_hotpot_corpus")

DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
PRIMARY_SPLIT = "validation"
TOPUP_SPLIT = "train"
DEFAULT_MIN_RECORDS = 50_000


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


def paragraph_id(title: str) -> str:
    """Stable id for a paragraph, derived from its Wikipedia title.

    Title is the right identity here because HotpotQA's own supporting-fact
    labels reference paragraphs by title. Using a hash of the title rather
    than a counter means the id does not change if we later process the
    questions in a different order.
    """
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()
    return f"hotpot:{digest[:16]}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Stats:
    questions_seen: int = 0
    paragraph_slots: int = 0
    unique_paragraphs: int = 0
    duplicate_titles: int = 0
    title_text_conflicts: int = 0
    empty_paragraphs: int = 0
    missing_gold: int = 0
    lengths: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #


def iter_paragraphs(row: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield (title, text) for each candidate paragraph attached to a question.

    The 'context' field is column-oriented: parallel lists of titles and of
    sentence lists. Sentences already carry their own trailing spaces, but we
    normalise whitespace rather than trusting that.
    """
    context = row["context"]
    for title, sentences in zip(context["title"], context["sentences"], strict=True):
        text = " ".join(s.strip() for s in sentences if s and s.strip())
        yield title, text


def gold_titles(row: dict[str, Any]) -> list[str]:
    """Titles of the paragraphs that genuinely support the answer.

    supporting_facts is also column-oriented, and names the same title once
    per supporting sentence, so duplicates are expected and removed here.
    Order is preserved for readability.
    """
    seen: dict[str, None] = {}
    for title in row["supporting_facts"]["title"]:
        seen.setdefault(title, None)
    return list(seen)


def convert(
    split: str,
    revision: str,
    corpus_out: Any,
    questions_out: Any,
    seen_titles: dict[str, str],
    stats: Stats,
    limit: int | None = None,
    stop_at: int | None = None,
) -> None:
    """Stream one split, writing paragraph and question records as we go.

    seen_titles maps title -> content_hash, and is how duplicates are detected
    across the whole run rather than only within a single question.
    """
    log.info("loading %s/%s split=%s", DATASET, CONFIG, split)
    dataset = load_dataset(DATASET, CONFIG, split=split, revision=revision)
    total = len(dataset) if limit is None else min(limit, len(dataset))
    log.info("%s rows available, processing %s", f"{len(dataset):,}", f"{total:,}")

    for index in range(total):
        row = dataset[index]
        stats.questions_seen += 1

        for title, text in iter_paragraphs(row):
            stats.paragraph_slots += 1

            if not text:
                stats.empty_paragraphs += 1
                continue

            digest = content_hash(text)
            previous = seen_titles.get(title)
            if previous is not None:
                stats.duplicate_titles += 1
                if previous != digest:
                    # Same title, different body. Rare, but worth counting: it
                    # tells us whether title alone is a safe identity.
                    stats.title_text_conflicts += 1
                continue

            seen_titles[title] = digest
            stats.unique_paragraphs += 1
            stats.lengths.append(len(text.split()))

            corpus_out.write(
                json.dumps(
                    {
                        "doc_id": paragraph_id(title),
                        "title": title,
                        "text": text,
                        "source": f"hotpotqa/{CONFIG}/{split}",
                        "format": "jsonl",
                        "content_hash": digest,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        gold = gold_titles(row)
        if not gold:
            stats.missing_gold += 1

        questions_out.write(
            json.dumps(
                {
                    "question_id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    # 'bridge' needs a fact from one paragraph to locate the
                    # second; 'comparison' needs the same fact from two and a
                    # comparison between them.
                    "reasoning_type": row["type"],
                    "difficulty": row["level"],
                    "gold_titles": gold,
                    "gold_doc_ids": [paragraph_id(t) for t in gold],
                    "category": "multi_hop",
                    "source": f"hotpotqa/{CONFIG}/{split}",
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        if stop_at is not None and stats.unique_paragraphs >= stop_at:
            log.info("reached %s unique paragraphs, stopping early", f"{stop_at:,}")
            return

        if stats.questions_seen % 1000 == 0:
            log.info(
                "%s questions -> %s unique paragraphs",
                f"{stats.questions_seen:,}",
                f"{stats.unique_paragraphs:,}",
            )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def resolve_revision(explicit: str | None) -> str:
    """Return the dataset commit to read, recording it so runs are replayable."""
    if explicit:
        return explicit
    sha = HfApi().dataset_info(DATASET).sha
    log.info("resolved %s to revision %s", DATASET, sha)
    return sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="process only N questions")
    parser.add_argument("--revision", default=None, help="dataset commit SHA to pin to")
    parser.add_argument("--min-records", type=int, default=DEFAULT_MIN_RECORDS)
    parser.add_argument("--no-topup", action="store_true", help="never read the train split")
    args = parser.parse_args()

    setup_logging()
    disable_progress_bars()

    corpus_dir = settings.corpus_dir / "hotpot"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    settings.eval_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpus_dir / "paragraphs.jsonl"
    questions_path = settings.eval_dir / "hotpot_questions.jsonl"
    manifest_path = corpus_dir / "MANIFEST.json"

    revision = resolve_revision(args.revision)
    stats = Stats()
    seen_titles: dict[str, str] = {}
    splits_used = [PRIMARY_SPLIT]

    with (
        corpus_path.open("w", encoding="utf-8") as corpus_out,
        questions_path.open("w", encoding="utf-8") as questions_out,
    ):
        convert(
            PRIMARY_SPLIT,
            revision,
            corpus_out,
            questions_out,
            seen_titles,
            stats,
            limit=args.limit,
        )

        short_by = args.min_records - stats.unique_paragraphs
        if short_by > 0 and args.limit is None and not args.no_topup:
            log.warning(
                "%s unique paragraphs is %s short of %s; topping up from '%s' "
                "in original row order",
                f"{stats.unique_paragraphs:,}",
                f"{short_by:,}",
                f"{args.min_records:,}",
                TOPUP_SPLIT,
            )
            splits_used.append(TOPUP_SPLIT)
            convert(
                TOPUP_SPLIT,
                revision,
                corpus_out,
                questions_out,
                seen_titles,
                stats,
                stop_at=args.min_records,
            )

    lengths = sorted(stats.lengths)
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "revision": revision,
        "splits_used": splits_used,
        "limit": args.limit,
        "questions": stats.questions_seen,
        "paragraph_slots_seen": stats.paragraph_slots,
        "unique_paragraphs": stats.unique_paragraphs,
        "duplicate_titles_skipped": stats.duplicate_titles,
        "same_title_different_text": stats.title_text_conflicts,
        "empty_paragraphs_skipped": stats.empty_paragraphs,
        "questions_without_gold_labels": stats.missing_gold,
        "word_count": {
            "min": lengths[0] if lengths else 0,
            "median": int(statistics.median(lengths)) if lengths else 0,
            "p95": lengths[int(len(lengths) * 0.95)] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
            "mean": round(statistics.mean(lengths), 1) if lengths else 0,
        },
        "outputs": {
            "corpus": str(corpus_path.relative_to(Path.cwd())),
            "questions": str(questions_path.relative_to(Path.cwd())),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    log.info("wrote %s", corpus_path)
    log.info("wrote %s", questions_path)
    log.info("wrote %s", manifest_path)
    print("\n" + json.dumps(manifest, indent=2))

    if stats.unique_paragraphs < args.min_records and args.limit is None:
        log.error(
            "only %s unique paragraphs; requirement is %s",
            f"{stats.unique_paragraphs:,}",
            f"{args.min_records:,}",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
