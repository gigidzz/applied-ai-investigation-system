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
| 4 | Retrieval | Semantic search, keyword search, custom fusion and ranking, optional cross-encoder rerank | done |
| 5 | Reasoning | Query decomposition, tool calling, multi-source synthesis, inline citations, contradiction detection, abstention | done |
| 6 | Evaluation | Automated harness over 40+ questions; Recall@K, MRR, answer correctness, citation accuracy, abstention precision; configuration comparison | done |
| 7 | Interface | HTTP API and web UI for document management, querying and source inspection | done |
| 8 | Hardening | Test coverage, retry and timeout behaviour, caching, token and cost accounting, observability | mostly absorbed |
| 9 | Deployment | Public URL on free-tier hosting, persistent vector storage, protected secrets | next |
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

## Retrieval

`src/ragsys/retrieval/` decides which chunks reach the prompt. Two routes
produce two rankings; generation can be given only one list.

**Scores cannot be added.** Cosine similarity is bounded between -1 and 1 and
BM25 is unbounded. Measured on one query, the best dense score was 0.777 and
the best sparse score 39.917, so a sum lets sparse decide everything — and the
same query's paraphrase counterpart had sparse retrieval confidently returning
Wikipedia paragraphs about card games. Per-query normalisation is the other
obvious repair and is fragile, because the normaliser is the top score, so a
query where everything is mediocre has its best mediocre result rescaled to
look excellent.

**Reciprocal rank fusion** discards the scores and keeps positions. Its useful
property is rewarding agreement: a chunk both routes rank highly beats a chunk
either ranks first alone. On the release-identifier query this moves the
correct release from third to first, with roughly twice the score of anything
else, purely because it was the one document both routes returned.

The fusion constant is 60. At that value the gap between rank 1 and rank 3 is
about 3%, so appearing in both rankings outweighs topping one of them, which
is the right trade when each route is known to be confidently wrong on
different kinds of question.

**Fusion cannot repair agreement on the wrong answer.** Asked "when did the
rollback finish", all three of dense, sparse and fusion returned generic
rollback runbooks, and none returned the incident report or chat thread that
record a completion time. Neither engine malfunctioned: the question genuinely
is more similar to a document about rollback procedure than to a timeline
entry mentioning one. What is missing is not a better similarity measure but
the knowledge that "when did X finish" asks for a record, not a procedure.

### The custom ranking component

Four signals, applied multiplicatively over the fused candidates:

```
final = fusion x (1 + w1*metadata + w2*type + w3*recency - w4*crowding)
```

Multiplicative because fusion scores are small and their scale depends on the
fusion constant, so an additive bonus would need retuning whenever that
changed. As a multiplier, a weight of 0.6 means "a full metadata match is
worth a 60% boost" at any scale.

1. **Metadata agreement.** Whether the question's identifiers — release,
   incident, ADR, date, severity, service — match the fields ingestion
   extracted. This is the signal neither route can produce: both see only
   chunk text, and neither knows a chunk is *tagged* `service:
   payment-gateway`. Service names are tested in reverse, taking the name the
   chunk declares and looking for it in the question in both spellings,
   because a question says "payment gateway" where the corpus writes
   "payment-gateway".
2. **Document type, conditioned on question type.** A temporal question
   actively *demotes* runbooks rather than merely not boosting them, because a
   runbook is a plausible match for the words and a poor match for the intent.
3. **Recency, as a boost that is never negative.** Dates appear in 97.9% of
   operations records but 0.6% of the corpus, so penalising their absence
   would suppress all 66,581 Wikipedia paragraphs.
4. **Crowding.** Each additional chunk from an already-represented document is
   discounted, capped at 75% so a question whose answer genuinely spans one
   report can still be served. Selection is greedy because the cost of a chunk
   is not a property of the chunk but of the chunk given the current result
   set.

Reciprocal rank fusion is a published algorithm and is not claimed as
original. What is specific to this system is the re-scoring layer: these four
signals, their weights, the question-type classification, and the decision to
boost rather than filter on dates. Each was added in response to a failure
measured on this corpus rather than anticipated in advance.

### Measured results

Document recall at the final cut over the 23 authored operations questions,
`scripts/compare_retrieval.py`:

| configuration | doc recall | all gold found | distinct docs |
|---|---|---|---|
| dense-only | 80.0% | 73.9% | 8.7 |
| sparse-only | 80.0% | 78.3% | 9.6 |
| hybrid-fusion | 85.7% | 78.3% | 9.9 |
| hybrid+rerank | 82.9% | 78.3% | 8.9 |

**These differences are inside the noise and must not be read as a ranking.**
23 questions carry about 53 gold documents in total, so the gap between 85.7%
and 82.9% is roughly one and a half documents, and the cross-document category
holds three questions, where one lost document moves the figure by eleven
points. The sample is far too small to choose a configuration, which is why
stage 6 measures against HotpotQA's 7,405 externally labelled questions.

What is *not* noise is the query the signals were built for:

```
dense-only      1-3. three runbooks    4. hotpot/paragraphs.jsonl
sparse-only     1-3. three runbooks    4. hotpot/paragraphs.jsonl
hybrid-fusion   1-3. three runbooks    4. hotpot/paragraphs.jsonl
hybrid+rerank   1-3. three runbooks    4. INC-2026-03-14.pdf   <- gold
```

Only the re-ranked configuration retrieved a gold document at all. The
document-type signal worked in the intended direction and was too weak to
finish the job: three runbooks still hold the top positions. Arithmetically a
runbook keeps 72% of its score under a -0.8 preference at weight 0.35, while
the incident report gains 35%, and the runbooks start from a higher fusion
score because both routes agreed on them — so the correction lands just short.

One consistency bug was found by tracing a query rather than by a test.
Metadata agreement is a ratio of matched weight to *expected* weight, and the
service check was expected whenever a **chunk** declared a service, while
every other field was expected only when the **question** named one. A chunk
matching a release exactly therefore scored 0.56 instead of 1.0 for also
carrying a service tag the question never mentioned. The check now applies
only when the chunk's service is named, or when the question names a different
one, with releases and incident references stripped first because both are
service-shaped. Re-running the comparison afterwards produced **identical**
figures: the fix changed scores without changing which documents reached the
final fifteen for these 23 questions, so it is recorded as a correctness fix
and not as an improvement.

The weights are deliberately **not** adjusted to fix this. Fitting them to 23
questions written for this corpus would be marking our own homework, and the
figures above would stop meaning anything. They are tuned at stage 6 against
HotpotQA's external labels, and whatever those weights turn out to be is what
these operations numbers get reported at.

## Reasoning

`src/ragsys/reasoning/` turns retrieved chunks into a cited answer. One rule
governs everything: the model may use only the text it is given. That is what
makes citations checkable, abstention meaningful, and contradiction reporting
possible at all — a model answering from its own parameters cannot distinguish
knowing from guessing.

**The client is standard-library HTTP, not a vendor SDK.** The deployed image
stays small, there is no second dependency to pin against a free-tier host,
and tool calling is a handful of JSON fields rather than a framework. This
interpreter cannot complete a TLS handshake without `truststore`, because the
python.org build ships no certificate bundle on macOS, so the client installs
it lazily before its first request.

**Citations are validated in code, never trusted.** Every `[Sn]` in the answer
is matched against the labels actually supplied, and anything left over is
recorded in `invalid_labels` rather than dropped. A model citing `S18` when
fifteen sources were given is a verifiable fault, not a question of quality,
so it is not sent to a judge. The model's own list of claimed labels is
checked too: a label listed but never written still asserts that a source was
relevant.

**Structured output is carried in the prompt, with the provider's schema as a
fallback.** Gemini cannot combine strict JSON output with tool calling in one
request, and tool calling is a requirement, so the output contract lives in
the system prompt and is parsed here. A reply that will not parse triggers one
retry with the provider's own schema and no tools. That path fired on a real
question during development, so it is exercised rather than theoretical.

**Tool calling is one tool: `search_documents`.** It earns its place because
the second query of a multi-part question is only knowable after reading the
answer to the first. Labels are handed out by a register that keeps them
stable across rounds, since a chunk the model saw as `S4` must still be `S4`
after a tool call appends more, or the citations it writes point at the wrong
documents.

**A model outage is a result, not an exception.** `Agent.answer` returns an
`Answer` carrying the error and an abstention, so one HTTP failure cannot
abort a run over the whole question set.

### Model fallback

The client tries several models in order. Both reasons were observed against
this key within one minute: `gemini-2.5-flash` answered 404 with a message
naming its replacement, and `gemini-3.8-flash` answered 503 "experiencing high
demand" however long the backoff. Of seven flash models probed, three were
unusable and four answered immediately. Failing a question while an equivalent
model sits idle is not acceptable behaviour on a free tier, so 404 and 503
advance to the next candidate while 400 and 403 — our own mistakes — abort.

### Measured behaviour

Three real questions against the live index and a live model.

**The planted contradiction**, `when did the rollback of pay-gw-4.2.1 finish`:

```
According to the incident report timeline, the rollback was complete at 15:10
[S1], with the incident marked resolved at 15:10 UTC [S5]. However, the
incident chat thread records r.achterberg stating at 15:40 that the rollback
finished on all pods [S8].

CONTRADICTIONS: S1 states 15:10, while S8 states 15:40  ['S1', 'S8']
grounded=True  confidence=low  sources=15  tool_calls=0
```

Both times reported, each attributed, neither chosen. `confidence` is `low`
because the evidence conflicts, which is the correct self-assessment.

**A deliberate gap**, `what does the kyc-service runbook say about escalation`
— the corpus has 18 runbooks and deliberately none for kyc-service:

```
ABSTAINED: The provided documents do not contain the runbook or escalation
           instructions for kyc-service.
grounded=True  confidence=high  sources=21  tool_calls=2
```

Note `tool_calls=2` and `sources=21`. The model searched twice, listed the
runbooks it did find, and only then abstained. That is the multi-step
requirement and the abstention requirement satisfied by the same trace.

**Cross-document synthesis**, the brief's own example question:

```
The March disruption (INC-2026-03-14) was caused by release pay-gw-4.2.1
deployed to payment-gateway, which reduced the per-pod database connection
pool from 50 to 10 [S2][S5]. The affected services were payment-gateway,
checkout-api, and merchant-portal [S2].

grounded=True  confidence=high  sources=15
```

Correct on all three parts. Worth recording that it cited the incident report
and the monthly operations report, while the answer key names the change log,
the incident report and the service catalogue. The answer is right and the
documents cited are not the documents expected, which is exactly why stage 6
must score retrieval and answer correctness separately rather than assuming
one implies the other.

### A known retrieval gap, recorded rather than patched

Tracing an arbitrary question, `why did checkout-api fail in March`, exposed a
real failure: the top results were a **May** incident and change-log rows, the
March post-mortem never reached the result set, and the system abstained.

The cause is measurable. The operations corpus writes dates in ISO form and
never writes month names — `2026-03-14` appears in one chunk, while the token
`march` appears in **3,616** chunks, almost all Wikipedia prose about people
named March and the verb. So inverse document frequency gives `march` a *low*
weight of 3.00 and points it at noise, while nothing connects the word to
`03`. The embedding does not encode that calendar relationship either, and the
question classifier reads "in March" as `factual` rather than temporal, so the
date signal never fires.

The gap is not marginal: **6 of the 23 authored questions mention a month
name**, a quarter of the set.

The obvious repair — rewriting `March` to `2026-03` before searching — was
measured and rejected. **211 of HotpotQA's 7,405 questions mention a month**,
and they mean years such as 631, 1937 and 2012: "From March 631 to April 631,
Farrukhzad Khosrau V was the king of..." Assuming the corpus year would make
those questions worse, and rewriting a user's words before retrieval breaks
cases nobody enumerated in advance.

Two sound repairs remain, both deferred until stage 6 can measure them:

- **Query-time**, parse a month and year into a date range and boost chunks
  whose `meta_date` falls inside it. Reuses the existing metadata and the
  established rule that dates boost and never filter. No re-indexing.
- **Index-time**, normalise each date into a written form alongside the ISO
  one, so the question needs no rewriting at all. Cleaner, and the usual
  answer in production systems, but it changes the sparse index and so
  requires a rebuild.

The reason for deferring is the same reason stage 6 precedes stage 8 in this
plan: there is currently no harness that could demonstrate either repair helped
the operations questions without harming the 211 HotpotQA questions. Changing
retrieval on the evidence of one failing question, with no baseline to compare
against, is how a system acquires fixes nobody can justify later.

What the trace did confirm is that grounding works. A system answering from
its own knowledge would have reported the May outage as the cause with every
appearance of confidence. This one declined to answer.

## Evaluation

`src/ragsys/evaluation/` scores the system against **47 questions** assembled
into one file, `data/eval/test_set.jsonl`, so the count is visible in one
place rather than requiring two numbers and a sampling rule to be explained.

| category | count | source |
|---|---|---|
| simple | 13 | authored |
| cross-document synthesis | 7 | authored |
| contradiction | 4 | authored |
| unanswerable | 3 | authored |
| multi-step | 20 | HotpotQA, externally labelled |

Four of the synthesis questions were added at this stage, because three was
too thin for a category the brief names explicitly: with three questions one
failure moves the figure by 33 points, so the number carries no information.
They were written from `scripts/ops_corpus/facts.py`, which predates all
retrieval code, and deliberately **without running retrieval on them first**,
so the gold sets record where the facts live rather than what the system
happens to return. That they were authored later than the rest is recorded
here rather than left for someone to notice.

The HotpotQA sample is drawn with a fixed seed. An evaluation whose question
set changes between runs cannot support a claim that a change improved
anything.

**Retrieval and answers are measured separately and never averaged.** They
fail independently, and stage 5 demonstrated it: the synthesis question was
answered correctly from the incident report and the monthly operations report
while the answer key names the change log and the service catalogue. Scoring
them as one number would have recorded that as a partial failure.

**The two groups are also never averaged.** HotpotQA is 20 externally labelled
Wikipedia questions; the authored set is 27 questions over documents we wrote.
A blended figure would let a good retrieval score conceal bad contradiction
handling.

### Metrics, and the failure each one catches

**Recall@K** — did the needed documents arrive at all. The failure no prompt
can repair. **MRR** — how high the first one ranked, because only the top of
the list is read closely. **Precision@15** — reported, not optimised: a chunk
can be useful context without being in the answer key. **Citation validity**
— the fraction of citations pointing at a source that was actually supplied,
which is arithmetic rather than judgement and is the fabrication check.
**Citation relevance** — overlap between cited documents and the answer key,
reported rather than maximised for the reason above. **Abstention precision
and recall**, kept separate: a system that abstains constantly scores
perfectly on one and terribly on the other. **Contradiction detection** —
measured only on the questions where disagreement exists.

Unanswerable questions are scored in code rather than by the judge. The
correct behaviour is known exactly, so a model call would be both a wasted
request and a less reliable answer than a boolean.

### A bug the inconsistency exposed

The first run reported **0.0% recall on HotpotQA alongside 75% of gold
documents found** — figures that cannot both be true. The cause was in the
harness, not the system: it offered the matcher both identifier forms by
concatenating them, so fifteen results became thirty entries and every
HotpotQA gold document landed at rank 16 or later, outside every `@k` window.
The rank-unbounded measures still found them, which is what made the
contradiction visible. Rank-based metrics require exactly one identifier per
retrieved item, and two regression tests now assert it.

### Measured results, retrieval

47 questions, `structural` chunks, 72,294-point index.

| configuration | R@5 | R@10 | R@15 | MRR | all gold found |
|---|---|---|---|---|---|
| dense-only | 68.4% | 77.7% | 80.9% | 0.717 | 66.0% |
| sparse-only | 66.3% | 76.6% | 79.8% | 0.698 | 66.0% |
| hybrid-fusion | 75.9% | 79.1% | 85.2% | **0.775** | 72.3% |
| hybrid+rerank | 69.0% | 77.0% | **85.5%** | 0.706 | **74.5%** |

**Hybrid retrieval is worth about 5 points of recall@15 over either route
alone**, and 6 to 8 points on retrieving every needed document. That is the
system's central design decision, and on 47 questions it is no longer a
measurement inside the noise.

The comparison between fusion alone and fusion plus the ranking signals shows
a real trade rather than a winner. The reranker retrieves **more complete**
gold sets — 74.5% against 72.3% — while ranking the first gold document
**lower**, MRR 0.706 against 0.775. That is the crowding penalty behaving
exactly as designed: spreading results across documents improves the chance of
collecting every document a synthesis question needs, and costs depth on the
single best one. Which trade is preferable depends on whether answers need one
document or several, so both configurations are reported rather than one being
declared correct.

| | R@15 | MRR | all gold found |
|---|---|---|---|
| authored (27) | 84.1% | 0.557 | 74.1% |
| HotpotQA (20) | 87.5% | 0.908 | 75.0% |

HotpotQA's MRR of 0.908 means the gold paragraph is usually rank 1, which is
expected: its questions name their entities explicitly, so both routes agree.
The authored questions are harder by that measure because several need three
to five documents rather than one.

By category, on `hybrid+rerank`:

| category | R@5 | R@15 | all gold found |
|---|---|---|---|
| simple | 96.2% | 100.0% | 100.0% |
| contradiction | 75.0% | 87.5% | 75.0% |
| multi-step | 75.0% | 87.5% | 75.0% |
| cross-document synthesis | 27.4% | 67.1% | 42.9% |
| unanswerable | 0.0% | 50.0% | 33.3% |

**Simple questions are solved**: 100% recall by rank 10, every gold document
found. **Cross-document synthesis is the weakest category** at 67.1% recall
and only 42.9% of questions retrieving every needed document — which is the
honest consequence of asking for three to five specific documents inside
fifteen slots, and the clearest target for further work.

The unanswerable row should not be read as a failure. Those questions have no
answer in the corpus, and their "gold" documents are where a reader would
reasonably look, so low recall against them is close to meaningless. What
matters for that category is whether the system abstains, which is an answer
metric rather than a retrieval one.

### Free-tier limits, measured

The first attempt at answer scoring produced **46 model failures out of 47
questions**, all HTTP 429. The diagnosis is in the run log: 46 quota refusals
to the generator and 93 to the judge. Question 1 succeeded; everything after
it was refused, and the reported figures — 8.5% correctness, 2.1% grounded —
measure the rate limiter, not the system.

Three causes, each now fixed:

**No client-side pacing.** The run issued requests as fast as retrieval could
feed it. The free tier limits requests per minute and returns 429 for the
remainder of the window once exceeded, so an unpaced run converts its whole
quota into refusals. Requests are now paced to `llm_requests_per_minute`,
with the interval shared process-wide, because the generator and the judge
draw on one quota and pacing them separately would exceed it by exactly a
factor of two.

**Backoff designed for the wrong failure.** 429 was handled by the same
exponential schedule as a transient 503, capped at 8 seconds. A per-minute
quota cannot clear in 8 seconds, so each retry was another refusal. 429 now
has its own pause, `llm_quota_backoff_seconds`, and 503 keeps the short one.
A quota refusal means wait, not try harder.

**Too many calls per question.** The agent offered its search tool even on the
final permitted round, so the model would request a search that could not be
run, costing a round trip to learn nothing. Measured at one wasted call in
four on a question needing two searches. Tools are now withheld on the last
round.

**Answer scoring is resumable.** The binding constraint is a daily quota
rather than wall-clock time, so losing 46 scored questions to an interruption
is expensive in a way that re-running a 16-minute index build is not. Each
outcome is appended as it completes and replayed on the next run; a question
recorded as failed is retried rather than replayed, since a quota refusal is
missing data and not a result.

The cost model this implies, per answerable question: one generation call,
plus one per tool round the model chooses, plus one judge call. At two tool
rounds that is four calls, so a 47-question run is roughly 150 to 190
requests. Against a daily free-tier allowance in the low hundreds, a full
answer evaluation is close to a day's entire budget — which is why the
retrieval evaluation, costing nothing, carries four configurations while
answer scoring runs once.

## Defects found, and how each was found

Recorded because *how* a bug surfaced is more useful than the bug. Almost none
of these were found by reading the code: they were found by a test asserting a
property, or by two measurements disagreeing with each other.

| # | Defect | How it surfaced | Fix |
|---|---|---|---|
| 1 | `\b20\d{2}-\d{2}-\d{2}\b` missed `2026-03-14T14:20:00Z` — no word boundary between `4` and `T` | ISO date coverage reported 17.8% where 97.9% was expected | lookarounds instead of `\b` |
| 2 | Generic runbooks were accidental near-duplicates, Jaccard 0.5–0.65 | the duplicate profiler found 10 pairs where 1 was planted | derive each runbook from its own service's history |
| 3 | The corpus was not byte-reproducible: `python-docx` stamps zip members with wall-clock time | two builds produced different bytes | rewrite the zip with a fixed timestamp |
| 4 | A document's date was read from anywhere in its text, so a runbook was dated from "added after INC-2026-03-14" | inspecting extracted metadata | read the date only from the first 400 characters |
| 5 | `structural` chunking produced output identical to `fixed` on every PDF | the two strategies gave the same chunk counts | pypdf text has zero blank lines; added a sentence-level fallback |
| 6 | Headings were used for metadata and then discarded, losing their words | a property test asserting no content word is lost | hold a heading-only block and prepend it to the next |
| 7 | Citation paths were absolute, and would have been baked into every stored chunk | reading a chunk record | paths relative to the corpus root |
| 8 | Index build ran at 6 chunks/s, extrapolating to three hours | the measured rate contradicted the estimate | the embedder is CPU-bound at 64 and the upload latency-bound at 256; decouple the two batch sizes |
| 9 | Metadata agreement diluted an exact release match to 0.56 because the chunk declared a service the question never mentioned | tracing one query's signals | the service check applies only when a service is named |
| 10 | `INC-2026-03-14` contains `2026-03-14`, so an identifier match counted as a missing date | a test asserting an exact match scores 1.0 | strip identifiers before reading dates |
| 11 | A retired model alias returns 404 and a popular one returns 503, failing a question while an equivalent model is idle | the first live call failed | a fallback chain across models |
| 12 | Rank metrics read 0.0% recall on HotpotQA while 75% of gold documents were found | two measurements that cannot both be true | one identifier per rank, not two concatenated lists |
| 13 | Contradiction questions were credited for abstaining, because a null reference answer was read as "should abstain" | a script printing "7 expect an abstention" when 3 do | key abstention on the category alone |
| 14 | The judge was told the required behaviour for a contradiction question was to abstain, marking correct answers wrong | the category scored 0% correct while contradiction detection scored 50% | grade against the behaviour the answer key describes |
| 15 | An unpaced evaluation run exhausted the free tier after one question: 46 failures | the run's own output | pace requests, give 429 its own long backoff, stop offering a tool on the final round |
| 16 | 429 did not trigger the model fallback, so a question failed while a working model sat one line below in the list | asking a question by hand after the quota ran out | 429 moves on immediately, and exhausted models are remembered |
| 17 | The answer never recorded which model produced it, so a fallback was invisible | reading the output of a question answered by a fallback | `Answer.model`, reported when it differs from the model requested |
| 18 | `--progress` was defined but never passed to the harness, so a run could not resume | the progress file did not exist after a run | wire it through, and assert it by parsing the call site |
| 19 | Re-uploading an unchanged file **deleted it from the index**: replacing removes the file's chunks first, then change detection reported UNCHANGED and skipped the write that would have restored them | uploading the same file twice through the running API | discard the previous state before ingesting, so every document in an upload is NEW and the write always happens |
| 20 | An upload wrote **both** chunking strategies into a live index built with one, so each uploaded passage was stored twice under two chunkings and an uploaded document occupied twice the result slots of a bulk-loaded one | walking through what one uploaded file became in the database, and finding two rows both numbered `chunk 0` | uploads use `settings.chunk_strategy` alone; writing every strategy is correct only for the offline build that stage 6 compares |

Two patterns are worth naming.

Defects 19 and 20 were both found by running the system and looking at
what it had actually stored, rather than by any test. Defect 19 is the
exception that proves the point about coverage: the API
had a test for re-uploading a file, but it re-uploaded *different* content
and so exercised the CHANGED path while the destructive case was UNCHANGED.
The test that mattered was the one nobody thought to write, and it took
running the real thing to find it.

**Most of these were found by a property test or a contradiction between two
numbers, not by review.** Defects 6, 10 and 12 were caught by assertions;
defects 8, 13 and 14 were caught because two measurements disagreed. That is
the argument for measuring the same thing two ways wherever it is cheap.

**Three of the twenty are the same mistake**: an identifier that contains a
date (4, 10) or a null that means two different things (13, 14). Both are
cases where a value's *shape* was trusted instead of its *meaning*.

## Stage 8, and why it has no separate section

Hardening was planned as a stage of its own and ended up being done inside the
stages that needed it, which is the better outcome: each piece was added in
response to a failure that had actually occurred rather than from a checklist.

| Stage 8 item | Where it happened | State |
|---|---|---|
| Test coverage | throughout | **216 tests**, property-based where it matters |
| Retry behaviour | stage 5, corrected in stage 6 | retryable statuses separated from quota refusals and from models that cannot serve at all |
| Timeouts | stages 3 and 5 | on every model call and every vector-store call |
| Rate limiting | stage 6 | paced process-wide, after an unpaced run produced 46 failures |
| Token and cost accounting | stage 5 | recorded per answer, aggregated per evaluation run, and the free-tier budget is worked out in this document |
| Error handling | stage 5 | a model outage becomes a recorded result rather than an exception, so one failure cannot abort a run of 47 questions |
| Observability | stages 0 and 5 | structured logging with module names; no tracing or metrics backend |
| **Caching** | **not done** | **cut** |

**Caching is the one item genuinely dropped**, and it is a bonus in the brief
rather than a requirement. The obvious candidate is an answer cache keyed on
the question, which would help a demo where the same question is asked twice
and nothing else — the evaluation asks each question once, and a grader is
unlikely to repeat one. Embedding the query is the only other repeated cost,
and at a few milliseconds it is not worth a cache's invalidation problem.

What remains genuinely missing from a production-hardening point of view is
request tracing and a metrics backend. Both are listed as bonus items, both
need infrastructure a free tier does not provide, and neither is claimed.
## Interface

`src/ragsys/api/` is deliberately thin, and the structure is the argument.

```
app.py        the application: routers wired, errors mapped, static served
deps.py       the expensive objects, built once
schemas.py    the published HTTP contract
routers/      ask.py, documents.py, health.py
static/       index.html, app.js, style.css
```

**There is no service layer in the API, because there is no business logic in
the API.** The one piece that looked like it belonged here — handling an
upload — is an ingestion concern: a command-line importer would want the same
behaviour, so `ingest_upload` lives in `ingestion/pipeline.py` and is tested
without starting a server. The test for whether something belongs in the web
layer is whether a non-HTTP caller could want it.

**There is no repository package either.** `ChunkStore` has been a protocol
with three implementations since stage 2, which is the repository. What was
genuinely missing was one query: the store holds chunks while the interface
needs files, so `list_documents` groups by source path. A three-page PDF is
three documents with ids ending `#page=1`, `#page=2`, `#page=3`, and offering
three rows to delete separately would be an interface that misleads.

**Dependencies are injected rather than imported.** The embedding model costs
roughly twenty-five seconds and 130 MB, so it is built once. The reason to use
`Depends` rather than a module global is testing: `tests/test_api.py` runs all
nineteen cases against an in-process Qdrant, a stub embedder and a scripted
model, with no cluster, no weights and no network.

**Domain errors map to status codes in exactly one place.** An unsupported
upload is 415, a missing index or unreachable model is 503, and nothing is a
500 — a 500 would mean a case we failed to anticipate. A new router inherits
the mapping without doing anything.

**The health check reports readiness without causing it.** Touching the model
would block for twenty-five seconds, long enough for a hosting platform's
probe to decide the container is dead and restart it, turning the health check
into the outage it exists to detect. It reads `Embedder.is_loaded` instead,
and the live endpoint returns `model_loaded: false` until the first question.

**Uploads replace rather than accumulate.** Change detection alone is not
enough: if a new version of a PDF has fewer pages, the surplus pages' chunks
belong to document ids the new file no longer produces, so nothing would ever
update them and the system would keep citing pages that no longer exist. An
upload therefore deletes its file's chunks before ingesting. A filename is
also reduced to its final component, since a browser may send one containing
path separators and joining that to a directory writes outside it.

**The front end has no build step.** Plain HTML, CSS and JavaScript: nothing
to install in the deployment image and nothing that can fail at build time.
Citations render as clickable chips that scroll to and highlight the source
chunk, and every retrieved source is shown rather than only the cited ones —
an uncited but retrieved chunk is often the most informative thing on the
page, because it shows what the model chose not to use.

### Verified against the live index

`GET /api/health` reported 72,294 chunks with `model_loaded: false`. Asking
"When did the rollback of pay-gw-4.2.1 finish?" through `POST /api/ask`
returned:

```
The rollback of pay-gw-4.2.1 finished at 15:10 according to the post-mortem
timeline [S1], or at 15:40 according to the chat thread [S8].

contradiction : S1 states 15:10, while S8 states 15:40   ['S1', 'S8']
grounded=True  confidence=low  sources=15  invalid citations=[]
```

`GET /api/documents` listed all 86 operations files newest first, excluding
the bulk Wikipedia corpus, which is indexed but not individually manageable.
An upload of a `.png` returned 415 with the supported extensions named.

### The document lifecycle, verified end to end

The API tests run against an in-process Qdrant with a stub embedder, which
proves the HTTP contract and nothing about embeddings. So the whole lifecycle
was also run against the live index with the real model, using a fictional
`zeppelin-gateway` service so that no answer could come from the existing
corpus.

| step | result |
|---|---|
| ask before uploading | abstained: "no information about a service named zeppelin-gateway" |
| upload | 2 chunks written, 72,294 to 72,296, listed as `uploads/zeppelin-runbook.md` |
| ask again | "77 connections per pod [S2][S15]", citing the uploaded file |
| ask a paraphrase | answered from the escalation section |
| re-upload unchanged | still 2 chunks, still answerable |
| re-upload edited | answers the new value, 0 stale chunks retrievable |
| delete | 2 chunks removed, back to 72,294 |
| ask once more | abstained again |

The paraphrase step is the one that proves embedding rather than plumbing.
Asked "Who should be paged if airship mooring telemetry stops arriving?" —
a question sharing almost no words with the document — the system answered
from the escalation section, on vectors created seconds earlier.

Two defects were found by running this and nothing else found them. An upload
wrote **both** chunking strategies into an index built with one, storing every
uploaded chunk twice; and re-uploading an unchanged file **deleted it**,
because replacement removes the file's chunks first and change detection then
reported UNCHANGED and skipped the write. Both are in the defect log as 19 and
20, and both now have regression tests.

**Deleting `change-log.csv` is allowed, and that is deliberate.** It is one
document in the interface despite producing 298 chunks, one per deploy, and it
is gold for two evaluation questions. Protecting particular files from
deletion would be an arbitrary special case in a document-management
interface, and the file is committed to the repository, so deleting it from
the index is recoverable by uploading it again.

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
