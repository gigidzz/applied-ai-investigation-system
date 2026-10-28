# Development roadmap

The system is built in sequential stages. Each stage ends in something
runnable, and each consumes what the previous one produced — retrieval cannot
be ranked before vectors exist, and prompts cannot usefully be tuned before
retrieval quality is measured.

This document is the plan. The source code deliberately does not reference
stage numbers: comments in code explain why the code is the way it is, not
when it was written.

| # | Stage | Purpose | Status |
|---|---|---|---|
| 0 | Foundations | Importable package, locked dependencies, validated configuration, secrets kept out of version control, test harness | done |
| 1 | Corpus | Acquire and normalise the document set; build the labelled evaluation question set; profile the data before designing against it | done |
| 2 | Ingestion | Per-format loaders, metadata extraction, two comparable chunking strategies, duplicate detection, add/update/delete | done |
| 3 | Indexing | Embedding model, vector store schema with dense and sparse vectors, idempotent batch upsert | done |
| 4 | Retrieval | Semantic search, keyword search, custom fusion and ranking, optional cross-encoder rerank | next |
| 5 | Reasoning | Query decomposition, tool calling, multi-source synthesis, inline citations, contradiction detection, abstention | |
| 6 | Evaluation | Automated harness over 40+ questions; Recall@K, MRR, answer correctness, citation accuracy, abstention precision; configuration comparison | |
| 7 | Interface | HTTP API and web UI for document management, querying and source inspection | |
| 8 | Hardening | Test coverage, retry and timeout behaviour, caching, token and cost accounting, observability | |
| 9 | Deployment | Public URL on free-tier hosting, persistent vector storage, protected secrets | |
| 10 | Documentation | Architecture write-up, evaluation report, AI usage report | |

## Dependency order

```
0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10
        │       │   │   │
    documents vectors │ answers
                  search
```

Stage 6 precedes stage 8 on purpose: optimisation without measurement is
guesswork, and the evaluation harness is what turns "this feels better" into
a number that can be defended in the report.

## Corpus

Two sources, each doing a job the other cannot.

**HotpotQA** (`distractor` config, `validation` split in full, pinned revision
`1908d6af`) yields **66,581 unique Wikipedia paragraphs** and **7,405
multi-hop questions**, each labelled with the paragraphs that support its
answer. That labelling is what makes retrieval metrics computable without
hand-annotation. Every question is 2-hop and marked `hard`, so HotpotQA
supplies no simple questions and no contradictions.

**A synthetic operations corpus** of **86 documents across six formats** for
the fictional company Northwind Payments, generated from
`scripts/ops_corpus/facts.py`. It carries four planted contradictions, three
deliberate gaps and two near-duplicate pairs, all declared in
`data/corpus/ops/DESIGN.md`, plus the four-document scenario the brief's
example question describes. It is committed to the repository because the
evaluation labels reference these exact files.

`galileo-ai/ragbench` was considered and rejected as a corpus: its questions
are single-document and its twelve domains are mutually unrelated, so
cross-document reasoning between them is meaningless. Its annotated
faithfulness and relevance scores would make a useful *secondary* evaluation
set at stage 6, if time allows.

`xanhho/2WikiMultihopQA` was rejected because HuggingFace returns HTTP 501 for
its parquet conversion and `datasets` 5.x no longer runs dataset scripts, so it
cannot be loaded with a pinned revision.

## Measured facts that constrain the design

From `docs/CORPUS_PROFILE.md`, regenerable with
`uv run python scripts/profile_corpus.py`:

- **66,964 records, 5.8 million words.**
- Median record is **78 words**; 95th percentile **177**; longest **1,378**.
- A 300-word chunk size would split **0.5%** of records, making chunking a
  no-op. Defaults are therefore 160 words with 40 overlap, which splits 7.4%.
- The chunking comparison is between **structural** (respect paragraph and
  heading boundaries) and **fixed** (uniform overlapping windows), not between
  two sizes.
- **18 distinct texts appear more than once.** HotpotQA was deduplicated by
  title, since its labels are title-based, which leaves identical bodies under
  different titles. Ingestion must hash content.
- ISO dates appear in **97.9%** of operations records but **0.6%** of the
  corpus overall. Date filtering must boost, never exclude, or a query
  mentioning a month would discard all 66,581 Wikipedia paragraphs.

## Ingestion

`src/ragsys/ingestion/` turns every supported file into uniform chunks. One
entry point, `ingest_file`, serves both the offline bulk build and the
single-document upload in the interface, so uploaded documents cannot be
processed differently from pre-loaded ones.

Seven loaders — pdf, docx, csv, md, json, jsonl, txt. Two of them do not map
one file to one document: a CSV yields one document per row, because each
deploy is independently citable, and a PDF yields one per page, so a citation
can name the page. An unreadable file is logged and skipped rather than
aborting a pass over 67,000 records.

Metadata extraction recovers the fields citations need — date, author,
severity, service, referenced incidents and releases — using generic patterns
rather than knowledge of this corpus. A document's own date is taken only from
its first 400 characters: a date deeper in the body is almost always a
reference to something else, and treating it as the document's date would
wrongly boost a runbook for every query about March.

Four document states drive add/update/delete. NEW and CHANGED are stored,
CHANGED first removing the previous version's chunks so the system cannot cite
text that no longer exists. UNCHANGED is skipped, which avoids re-running the
most expensive step in the pipeline. DUPLICATE is skipped with a record of
which document owns the text — the case identity-by-name cannot catch, and
content hashing found 26 of them that Stage 1's title-based deduplication
had missed.

### Measured results

From one run over the full corpus, 5.8 seconds, no failures:

| | structural | fixed |
|---|---|---|
| chunks | 72,294 | 72,441 |
| chunks per document | 1.08 | 1.08 |
| documents split | 4,961 (7.4%) | 4,961 (7.4%) |
| total words | 5,817,555 | 6,037,313 |
| median chunk | 76 words | 77 words |
| 95th percentile | 150 words | 160 words |

The 7.4% figure is exactly what profiling predicted for a 160-word limit,
which is the measurement that set the default. The strategies differ in two
visible ways: `fixed` holds 219,758 more words, because neighbouring windows
repeat 40 words of each other, and its chunks cluster at exactly 160 words
while `structural` chunks stop at whatever sentence or block boundary falls
below the limit.

Both preserve every content word of every document — asserted over the whole
committed operations corpus in `tests/test_chunking.py`, which caught two
heading-loss bugs during development.

## Indexing

`src/ragsys/indexing/` builds two representations of every chunk, because the
two fail in opposite directions. A dense embedding connects "card
authorisation failing" to "payment gateway returned 503" while treating
`pay-gw-4.2.1` and `pay-gw-4.2.0` as nearly the same text. A sparse term
vector separates those two releases exactly and cannot connect the paraphrase
at all. Retrieval merges both rankings, so both are stored under one point id.

**The dense half** uses `BAAI/bge-small-en-v1.5` at 384 dimensions, chosen for
deployability: it runs on CPU, which free hosting provides and a GPU does not,
and 384 dimensions is a third the storage of a 1024-dimension model. Vectors
are normalised at encode time, so cosine similarity reduces to a dot product.

bge models are trained asymmetrically — a query carries a short instruction
prefix and a passage carries none. `embed_documents` and `embed_query` are
separate methods rather than one method with a flag, because applying the
prefix to the wrong side degrades retrieval silently.

The model's output dimension is asserted against the configured value when the
model loads. A mismatch would otherwise appear as the vector store rejecting
every write, with nothing pointing at the real cause.

**The sparse half** is BM25, implemented directly rather than taken from a
library, because the score has to be split across two places: the term
frequency and length-normalisation factors are baked into the stored vector at
index time, and the inverse document frequency is applied to the query at
search time. The vector store only computes a dot product, so

```
stored_value(term) x query_value(term)   summed over shared terms   =   BM25
```

This identity is asserted in `tests/test_indexing.py` against an independent
BM25 implementation, over four queries and the whole test corpus, because the
entire hybrid design rests on it.

Tokenisation is hand-written for the same reason the corpus is partly
synthetic: identifiers are the point. A general tokeniser splits
`pay-gw-4.2.1` into fragments and loses exactly the string that dense
retrieval is worst at. Here a compound token is kept whole *and* emitted as
its parts, so a query for "payment gateway" still matches a document writing
"payment-gateway" while the full release number survives as a term of its own.

No stop-word list is used. Inverse document frequency already drives the
weight of a term appearing in most documents close to zero, and a hand-kept
list would be one more thing to justify.

### Measured results

| | |
|---|---|
| chunks indexed | 72,294 (`structural`) |
| points stored | 72,294 — exact match, no failures |
| build time | 15.7 min at 77 chunks/s end to end |
| distinct terms | 188,481 |
| sparse statistics on disk | 2.8 MB |
| statistics pass | 1.9 s |
| embedding throughput | 100 chunks/s at batch 64 |
| upload throughput | 527 chunks/s at batch 256 |

The two batch sizes are deliberately different and the first build got this
wrong. Embedding is fastest at 64 texts per forward pass, but every upsert
pays one round trip to the cluster in London regardless of its size, so
uploading 64 at a time spent as long waiting on the network as on the model —
about 44 chunks/s overall. Decoupling them, so the embedder subdivides a
larger upload batch internally, roughly doubles the end-to-end rate.

### What the index shows

`scripts/check_index.py` runs a question through both routes and prints them
side by side. Two results justify the hybrid design better than any argument
could.

**An exact identifier, where meaning search misleads.** Asked for
`pay-gw-4.2.1`, dense retrieval ranked the wrong releases above the right one:

```
MEANING   1. 0.777  release: pay-gw-4.6.0 ...        <- wrong release
          2. 0.723  release: paymen-4.0.9 ...        <- wrong release
          3. 0.720  release: pay-gw-4.2.1 ...        <- correct, ranked third

KEYWORD   1. 39.917 release: pay-gw-4.2.1 ...        <- correct, ranked first
          2. 39.854 INC-2026-03-14: Payment authorisation failures ...
```

Version strings are nearly indistinguishable in meaning space, exactly as
predicted. Dense retrieval is not slightly worse here, it is wrong.

**A pure paraphrase, where keyword search collapses.** Asked "why did card
authorisation start failing", with no words shared with the source document:

```
MEANING   1. 0.696  INC-2026-02-17: Double-entry ledger recording every authorisation ...
          2. 0.692  INC-2026-03-14: Payment authorisation failures ...

KEYWORD   1. 18.655 This is why many players will place a chip on their cards ...
          2. 16.306 Richard Timothy Jones (born January 16, 1972) ...
```

BM25 matched common words and returned Wikipedia paragraphs about card games
and actors. The two methods fail in opposite directions on the same corpus,
which is the whole argument for running both.

### The failure this exposed

Asked "when did the rollback finish", **both** routes returned generic
rollback *runbooks* and neither returned `INC-2026-03-14.pdf` or
`THREAD-2026-03-14-incident.txt` — the two documents that disagree about the
answer, and the entire point of evaluation question C1.

The cause is that the question names no incident, release or date, so the word
"rollback" dominates, and documents *about* rollback procedure outrank a
timeline entry that happens to record one finishing. Neither similarity
measure can fix this, because both are working correctly: the query genuinely
is more similar to a rollback runbook.

This is what retrieval scoring has to address, and it is the reason the
ranking stage needs corpus-specific signals rather than a better similarity
measure.

### Design decisions

**Statistics are written to disk, not recomputed.** BM25 weights are only
comparable if the document frequencies used at index time and at query time
are identical. The deployed service loads `sparse_stats_structural.json`
rather than reading the corpus, which it does not have.

**Point ids are derived, not assigned.** A chunk id is 16 hexadecimal
characters, which is exactly 64 bits, so it converts to Qdrant's integer id
without truncation. Re-running the build therefore overwrites rather than
duplicates, which is what makes the index idempotent and the bulk load
resumable.

**The expensive pass is resumable.** Embedding the full corpus takes about a
quarter of an hour; losing that to a dropped connection near the end is not an
acceptable failure mode, so progress is recorded per batch.

**One strategy is indexed in full.** `structural` is the production index and
is what satisfies the corpus-size requirement. `fixed` is indexed over a
subset into a separate collection purely to compare the two strategies at
stage 6; indexing both in full would double both the embedding time and the
stored vectors to answer a question a subset answers just as well.

## Design decisions already fixed

**Python 3.12, pinned `>=3.12,<3.13`.** PyTorch publishes no wheels for 3.13
or later, so installing on a newer interpreter would attempt a source build.
Free hosting platforms also run 3.12, and matching local to deployed avoids
debugging version differences at release time.

**One ingestion path.** Bulk corpus loading and interactive single-document
upload call the same function. Two separate paths would drift, and uploaded
documents would behave differently from pre-loaded ones.

**Bulk ingestion runs offline.** Embedding the full corpus takes roughly
25 minutes of CPU, which free hosting tiers will not sustain. The corpus is
embedded locally and written to a hosted vector store; the deployed service
only reads, plus small interactive uploads.

**Hybrid retrieval, not semantic alone.** Embeddings handle paraphrase well
and exact identifiers badly — release numbers and error codes are precisely
where meaning-similarity misleads. Keyword search is the inverse. The two
fail in opposite directions, so both run and their rankings are merged.

**A separate judge model.** Answer correctness is scored by a model other than
the one that produced the answer, because models systematically prefer their
own output.
