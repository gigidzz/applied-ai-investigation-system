# Evaluation report

Reproduce with:

```bash
uv run python scripts/build_test_set.py          # assemble the 47 questions
uv run python scripts/evaluate.py                # retrieval, 4 configurations
uv run python scripts/evaluate.py --answers      # answer quality, 1 configuration
```

Generator `gemini-3.6-flash`, judge `gemini-3.5-flash`, index 72,294
`structural` chunks, 47 questions.

## The question set

| category | n | source | what it tests |
|---|---|---|---|
| simple | 13 | authored | a fact stated in one document |
| cross-document synthesis | 7 | authored | combining 2 to 5 documents |
| contradiction | 4 | authored | two sources that disagree |
| unanswerable | 3 | authored | an answer that is absent |
| multi-step | 20 | HotpotQA | two-hop reasoning, externally labelled |

Authored and HotpotQA results are reported separately throughout and never
averaged. They measure different properties over incomparable data: 20
externally labelled Wikipedia questions against 27 questions over documents
written for this project.

## Retrieval

| configuration | R@5 | R@10 | R@15 | MRR | all gold found |
|---|---|---|---|---|---|
| dense-only | 68.4% | 77.7% | 80.9% | 0.717 | 66.0% |
| sparse-only | 66.3% | 76.6% | 79.8% | 0.698 | 66.0% |
| hybrid-fusion | 75.9% | 79.1% | 85.2% | **0.775** | 72.3% |
| hybrid+rerank | 69.0% | 77.0% | **85.5%** | 0.706 | **74.5%** |

**Hybrid retrieval is worth about five points of recall@15 over either route
alone**, and six to eight points on retrieving every document a question
needs. That is the central design decision of the system and it is now
measured rather than argued.

**Fusion and the ranking signals trade against each other rather than one
winning.** The reranker retrieves more *complete* gold sets, 74.5% against
72.3%, while ranking the first gold document *lower*, MRR 0.706 against 0.775.
That is the crowding penalty behaving as designed: spreading results across
documents improves the chance of collecting all five documents a synthesis
question needs, and costs depth on the single best one. Which is preferable
depends on whether answers need one document or several, so both are reported.

| category | R@5 | R@15 | all gold found |
|---|---|---|---|
| simple | 96.2% | 100.0% | 100.0% |
| contradiction | 75.0% | 87.5% | 75.0% |
| multi-step | 75.0% | 87.5% | 75.0% |
| cross-document synthesis | 27.4% | 67.1% | 42.9% |
| unanswerable | 0.0% | 50.0% | 33.3% |

The unanswerable row is close to meaningless and should not be read as a
failure: those questions have no answer in the corpus, and their "gold"
documents are merely where a reader would reasonably look.

## Answer quality

| metric | value |
|---|---|
| correctness (correct 1.0, partial 0.5) | **89.4%** |
| grounded | **100.0%** |
| citation validity | **100.0%** |
| citation relevance | 40.9% |
| abstention recall | **100.0%** |
| abstention precision | 75.0% |
| contradiction detection | **50.0%** |
| model failures | 0 |
| mean tokens per question | 3,525 |

Verdicts: 41 correct, 2 partial, 4 incorrect.

| | correctness | citation validity |
|---|---|---|
| authored (27) | 92.6% | 100% |
| HotpotQA (20) | 85.0% | 100% |

**Citation validity of 100% is the strongest result here.** Across 47
questions and every citation written, the system never once cited a source it
had not been given. Invalid citations are counted by arithmetic rather than
judged, so this is not a model's opinion: it is the count of labels that did
not resolve, and it is zero.

**Grounding of 100%** means every answer either cited its sources or
explicitly abstained. Nothing was asserted uncheckably.

**Citation relevance of 40.9% needs interpreting, not fixing.** The citations
are all valid; they often point at documents the answer key does not list.
This was observed directly: the flagship synthesis question was answered
correctly from the incident report and the monthly operations report, while
the key names the change log and the service catalogue. The same facts live in
both. The metric is reported rather than optimised for exactly this reason.

# Failure analysis

Six of 47 answers were not fully correct. Each was re-run through retrieval to
establish whether the documents it needed had arrived.

| question | category | verdict | R@15 | diagnosis |
|---|---|---|---|---|
| C2 | contradiction | incorrect | 100% | **reasoning** |
| C4 | contradiction | partial | 50% | mixed |
| S-cause-INC-2026-02-17 | simple | partial | 100% | **reasoning** |
| 5a724866… | multi-step | incorrect | 50% | mixed |
| 5a8b8b31… | multi-step | incorrect | 100% | **reasoning** |
| 5addc7e3… | multi-step | incorrect | 100% | **reasoning** |

**Four of six failures had every needed document in the prompt.** This is the
most useful finding in the evaluation and it inverts the obvious improvement
plan: retrieval is not the bottleneck. Better search would fix at most two of
these six.

## Failure 1: contradictions between a stale source and a later one

Contradiction detection is 50% — two of four planted conflicts flagged, two
missed — and the two misses share a shape.

**Caught**, both conflicts *within incident documentation*:

- **C1**, the post-mortem says the rollback finished at 15:10, the chat thread
  says 15:40. Both times reported, each attributed.
- **C3**, the post-mortem says 40% of authorisations failed, the monthly
  report says 12%. Both figures reported.

**Missed**, both conflicts where one source is *superseded by a later one*:

- **C2**, "which team owns the settlement queue?" The catalogue says Payments
  Core; ADR-0007, dated later and marked accepted, transfers it to Platform.
  The judge: *"asserts a single owner as fact instead of surfacing both
  claims, noting the later accepted ADR-0007, and flagging the catalogue as
  stale."* Every needed document was retrieved.
- **C4**, which Postgres version the ledger runs on. The answer took ADR-0009's
  version and did not mention that the runbook disagrees.

The distinction is real. Noticing that two documents state different numbers
for the same event is a comparison. Noticing that one document **supersedes**
another requires reasoning about dates, status fields and document authority —
that ADR-0007 is later, that it is marked `accepted`, and that this makes the
catalogue stale rather than merely different.

**How to improve it.** The dates and statuses are already in the prompt: every
source block carries `date:` and the ADR text carries `Status: accepted`. The
model is not using them. The fix is therefore prompt-level rather than
retrieval-level — an explicit instruction that when two sources disagree, the
later or explicitly accepted document should be identified as superseding and
the earlier one flagged as possibly stale. That is cheap to try and directly
testable against C2 and C4.

## Failure 2: reasoning failures on two-step Wikipedia questions

Three of 20 multi-step questions were wrong, and two of those three had **all**
their gold paragraphs supplied:

- *"what activity do they have in common"* — answered with an explanation of
  owner earnings instead of naming the activity, "business".
- *"which areas could still deny same-sex marriage after Windsor"* — discussed
  the Sixth Circuit and Obergefell instead of naming American Samoa and
  tribal lands.

Both are failures to extract the specific thing asked for from text that was
present. This is a model-capability limit, not a system defect: the generator
is a small, fast, free-tier model chosen for deployability on CPU-adjacent
free hosting.

**How to improve it.** Run the three failures against a larger model and
measure. If a larger model fixes them, the system is correct and the model is
the constraint, which is a configuration choice rather than a bug. The harness
supports this: the generator model is a config value and answer scoring is a
single flag.

## Failure 3: one false abstention

Abstention recall is 100% — all three unanswerable questions were correctly
refused. Abstention precision is 75%, because of one refusal that should not
have happened: asked for a football club's colloquial name, the system
abstained, *"falsely claiming the sources do not mention it"*. Half the gold
paragraphs had been retrieved.

**This is a deliberate trade showing its cost.** The system prompt is
emphatic about not answering beyond the evidence: *"Partial evidence is not
sufficient evidence"*, *"do not guess, infer beyond the text, or fill gaps
with plausible detail"*. That produced zero hallucinated citations and perfect
abstention recall, at the price of one over-cautious refusal in 47 questions.

For a system whose purpose is evidence-grounded answers, that is the right
direction to be wrong in. Softening the instruction would likely trade this
false abstention for hallucinations, which are harder to detect and worse to
ship. The trade is documented rather than silently tuned.

## Failure 4: one harsh judgement, recorded for honesty

`S-cause-INC-2026-02-17` was scored *partial* for identifying the root cause
(a Postgres volume at 94% capacity from a silently failed archival job) while
omitting a secondary consequence the reference mentions. This is arguably a
correct answer marked down, and it illustrates the limitation below rather
than a defect in the system.

# Limitations of this evaluation

- **The judge is from the same model family as the generator.** `gemini-3.5-flash`
  grading `gemini-3.6-flash` reduces self-preference bias without eliminating
  it, because both share training lineage. A genuinely independent judge needs
  a second provider. Correctness of 89.4% should be read as optimistic.
- **Eleven of the thirteen simple questions are two templates**, six asking
  which team owns a service and five asking an incident's root cause. The
  96% score in that category reflects two question shapes answered repeatedly
  rather than thirteen independent probes.
- **Three unanswerable questions is a small sample** for a behaviour as
  important as abstention, though each tests a different kind of absence: an
  event mentioned with no report, partial information, and a document
  referenced but not present.
- **Answer quality is measured for one configuration only.** Retrieval is
  compared across four, because it costs nothing; answer scoring costs roughly
  150 to 190 model requests, which is close to a day's entire free-tier
  allowance. The configuration comparison therefore rests on retrieval.
- **The two chunking strategies were not compared on retrieval quality.** They
  are compared on chunk counts, word totals and boundary behaviour; which one
  retrieves better would need `fixed` chunks indexed into a second collection
  and was not run.
- **A scoring bug was found and fixed mid-evaluation**, and is recorded because
  it changed the numbers. Contradiction questions carry a null reference
  answer, since no single value is correct, and the first version rendered
  that to the judge as "the correct behaviour is to abstain". All four
  contradiction questions were marked incorrect while the independent
  contradiction measure showed half had been handled correctly. Grading them
  against the required behaviour described in the answer key moved overall
  correctness from 85.1% to 89.4%.

# What to do next, in order of evidence

1. **Prompt for superseding documents.** Two of four contradiction misses are
   this one pattern, the needed documents are already retrieved, and the dates
   are already in the prompt. Highest expected gain per unit of work.
2. **Try a larger generator on the three multi-step failures.** Establishes
   whether the remaining errors are the system's or the model's.
3. **Broaden the simple questions** beyond two templates, then re-run.
4. **Improve synthesis retrieval**, the weakest category at 42.9% of questions
   retrieving every needed document — by retrieving more candidates per route,
   or by running one retrieval per sub-question.
5. **An independent judge** from a second provider, which would let the
   correctness figure be stated without the optimism caveat.
