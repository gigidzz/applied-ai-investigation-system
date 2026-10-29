# Evaluation report

How I tested this system, what I measured, what the results were, and what the
failures say about it.

Every figure here is produced by a committed script and saved under
`data/eval/`. Nothing is estimated or typed in by hand.

```bash
uv run python scripts/build_test_set.py                        # assemble the 57 questions
uv run python scripts/evaluate.py                              # retrieval, four configurations (no model calls)
uv run python scripts/evaluate.py --answers --compare-judges    # answer quality, both judges
uv run python scripts/compare_chunking.py                      # the two chunking strategies
```

---

# Part 1 — Approach

## Four things can go wrong, so I measure four things

The system retrieves passages, hands them to a model, and requires every claim
to cite the passage it came from. It refuses when the passages do not support
an answer, and reports disagreement rather than picking a side when two
documents conflict. That gives four independent failure modes, and fixing one
does nothing for the others:

1. The right documents never arrive. No prompt can repair this.
2. They arrive too far down the list to be read closely.
3. They arrive and the answer is still wrong.
4. The answer is right but uncheckable, or cites something it was never given.

A single blended score would hide which is happening, so I never blend them.
Nor do I average the two corpora together: they measure different properties
over incomparable data, and a good retrieval score would conceal bad
contradiction handling.

## The documents

**66,964 records from two sources that do different jobs.**

**HotpotQA** — 66,581 Wikipedia paragraphs from the `distractor` validation
split at a pinned revision. Every question is labelled with the paragraphs
supporting its answer, which is what makes retrieval measurable without
hand-annotating anything. All two-hop, all answerable, never contradictory.

**A synthetic operations corpus** — 86 documents for a fictional payments
company across six formats: post-mortems as PDF, decision records as DOCX, a
298-row deployment log as CSV, runbooks and monthly reports as Markdown, a
service catalogue as JSON, and chat threads as plain text.

Synthetic deliberately, because **contradictions and gaps have to be planted
to be known.** You cannot find a contradiction in a borrowed corpus without
reading all of it. This one carries four planted contradictions, three
deliberate gaps and two near-duplicate pairs, all declared in
`data/corpus/ops/DESIGN.md` before any retrieval code existed.

## The question set

57 questions, built by `scripts/build_test_set.py` into
`data/eval/test_set.jsonl`. The HotpotQA sample uses a fixed seed, so the set
is identical on every run — an evaluation whose questions drift cannot show
that anything improved.

| category | n | source | what it tests |
|---|---|---|---|
| simple | 20 | authored | a fact stated in one document |
| multi-step | 20 | HotpotQA | two-hop reasoning, externally labelled |
| cross-document synthesis | 7 | authored | combining 2 to 5 documents |
| unanswerable | 6 | authored | an answer that is absent |
| contradiction | 4 | authored | two sources that disagree |

The six unanswerable questions each probe a *different* kind of absence, which
matters more than the count: a service that does not exist, a well-formed
identifier that was never used, an event mentioned with no report behind it,
information present only in part, a document referenced but not included, and
a concept absent from the corpus entirely.

## The four retrieval configurations

Each question is searched twice — dense (384-dimension embeddings, meaning) and
sparse (BM25, exact terms) — because the two fail in opposite directions on
the same corpus. Asked for `pay-gw-4.2.1`, dense search ranked `pay-gw-4.6.0`
higher; asked "why did card authorisation start failing", sparse search
returned Wikipedia paragraphs about poker chips. The configurations isolate one
decision each, so a difference in the numbers points at a single cause.

| configuration | what it does |
|---|---|
| **dense-only** | meaning search alone |
| **sparse-only** | keyword search alone |
| **hybrid-fusion** | both, merged by reciprocal rank fusion, which discards the incomparable scores and rewards agreement between the routes |
| **hybrid+rerank** | the above, re-scored by four signals: metadata agreement, document type given question type, recency, and a penalty for repeating a document |

## The metrics

**Only one uses a model.** Everything else is arithmetic, deliberately.

| metric | decided by | the failure it catches |
|---|---|---|
| recall@K | arithmetic | the needed documents are not in the top K |
| MRR | arithmetic | they are present but ranked low |
| precision@15 | arithmetic | reported, not optimised — unused context is cheap, a missing document is not |
| citation validity | arithmetic | a citation names a passage never supplied: a fabricated reference |
| citation relevance | arithmetic | answered from documents the key does not list; a diagnostic, not a target |
| abstention recall | boolean | answered a question with no answer in the corpus |
| abstention precision | boolean | refused a question it could answer |
| contradiction detection | boolean | two sources disagreed and the answer silently picked one |
| **correctness** | **a model judge** | the answer is wrong in substance |

Correctness needs a judge because it cannot be string-matched — "15:10" and
"ten past three" share no characters — and hand grading does not scale to 57
questions re-run for every change. Unanswerable questions skip the judge
entirely: the correct behaviour is known exactly, so a boolean is both cheaper
and more reliable.

## Judging, and how I handled self-preference

Grading a model with a model has one known weakness: models prefer output from
their own family. The usual mitigation is to use a different model. I grade
every answer **twice** and publish the difference.

| role | model | relation to the generator |
|---|---|---|
| generator | `gemini-3.6-flash` | — |
| judge, reported throughout | `openai/gpt-oss-120b` | different provider, no shared lineage |
| second judge | `gemini-3.5-flash` | same family as the generator |

The gap between them is self-preference measured on this system's own answers
rather than quoted from a paper. Where they disagree I adjudicate by hand
against the answer key, and the result does not uniformly favour either.

## How I tested the system

This evaluation is one of four layers, and the slowest and least precise of
them. A correctness figure would be worthless if the code underneath were
untested.

**1. 262 automated tests, no network.** `uv run pytest`, about two seconds.
Nothing calls Qdrant, an embedding model or an LLM — the vector store is a
protocol so tests pass a fake, and model clients are substituted. That is why
it is fast enough to run after every edit, which is the only reason it gets
run after every edit. The heaviest areas are retrieval (51), reasoning (50)
and evaluation (43); the rest cover indexing, the API through `TestClient`,
chunking, loaders for all six formats, the ingestion pipeline, metadata,
config, deduplication and the scripts.

Two groups exist because of specific bugs. Each ranking signal is tested **in
isolation**, because testing them only in combination hides one pulling the
wrong way. And `tests/test_scripts.py` parses every script to assert each
declared `argparse` flag is actually read — added after `--compare-judges` was
declared in one edit while the code reading it was lost in another, so the
flag was accepted, silently ignored, and a run reported a two-judge comparison
it had not performed. It carries a negative control, because a check that
cannot fail is not a check.

**2. The measuring code is tested like the system.** A wrong metric does not
crash; it reports a plausible number. So the harness is tested too: metrics
against hand-computed examples, one identifier per rank, only the
`unanswerable` category expecting an abstention, contradiction questions
graded on required behaviour, a model outage recorded as unscored rather than
incorrect, the sample reproducible from its seed, and a run resumable from
saved progress. Most of those tests exist because the bug they describe
actually happened — Part 3 lists them with what each would have cost.

**3. I tested every feature by hand, including in the deployed service.** The
lifecycle no unit test covers — upload a document, ask a question needing it,
delete it, ask again — I ran myself, along with a question carrying a planted
contradiction and a question with no answer in the corpus. Automated tests
confirm the API contract; only this confirms the container, the cloud index
and the model API work together.

**4. This evaluation.** The only layer that can say whether the answers are any
good, and the only one costing money and twenty minutes.

**What I did not test.** No property-based testing: every test uses chosen
examples, so a tokeniser bug on an unanticipated input shape would slip
through. No load or concurrency testing. And no adversarial documents — a file
whose text tries to instruct the model — which for a system that puts
retrieved text straight into a prompt is the most interesting gap.

---

# Part 2 — Results

57 questions. Generator `gemini-3.6-flash`, graded by `openai/gpt-oss-120b` and again by
`gemini-3.5-flash`. Saved to `data/eval/report_final.json`.

| | |
|---|---|
| correctness | **86.8%** |
| every claim carries a citation | **100%** |
| every citation names a passage that was supplied | **100%** |
| questions with no answer in the corpus, correctly refused | **100%** |
| planted contradictions reported | 50% — the open problem, analysed in Part 3 |
| model failures | 0 |

**The two citation results are the strongest findings here, and the two that
need no judge.** Across 57 questions there was no claim without a citation
and no citation naming a passage the model was not given. That is what makes
an answer checkable, it is arithmetic rather than opinion, and it holds
because it is enforced in code rather than asked for in the prompt.

## Retrieval

| configuration | R@5 | R@10 | R@15 | MRR | P@15 | every gold document found |
|---|---|---|---|---|---|---|
| dense-only | 67.8% | 78.1% | 80.7% | 0.679 | 13.1% | 68.4% |
| sparse-only | 67.0% | 75.4% | 79.8% | 0.650 | 12.2% | 68.4% |
| hybrid-fusion | 74.0% | 80.1% | 86.0% | 0.729 | 13.7% | 75.4% |
| **hybrid+rerank** | 67.4% | 77.5% | 86.3% | 0.677 | 13.6% | 77.2% |

**Hybrid retrieval is the central design decision, and it is now measured
rather than argued:** fusing both routes beats either alone at every depth,
and by a wider margin on retrieving *every* document a question needs.

**Fusion and the ranking signals trade against each other rather than one
winning.** The re-ranker retrieves more complete gold sets while ranking the
first gold document lower — the crowding penalty behaving as designed, since
spreading results across documents improves the chance of collecting all five
documents a synthesis question needs and costs depth on the single best one.
Which is preferable depends on whether answers need one document or several,
so I report both.

### By source

| | R@5 | R@10 | R@15 | MRR | every gold document found |
|---|---|---|---|---|---|
| authored (37) | 63.3% | 76.1% | 85.7% | 0.552 | 78.4% |
| hotpot (20) | 75.0% | 80.0% | 87.5% | 0.908 | 75.0% |

The MRR gap is the clearest signal here. HotpotQA questions have two
supporting paragraphs phrased close to the question, so the first lands near
the top. The authored questions ask about documents describing the same
services in different ways, so the first useful one sits lower even when
everything needed is eventually found.

### By category

| | R@5 | R@10 | R@15 | MRR | every gold document found |
|---|---|---|---|---|---|
| simple (20) | 82.5% | 95.0% | 100.0% | 0.627 | 100.0% |
| multi-step (20) | 75.0% | 80.0% | 87.5% | 0.908 | 75.0% |
| cross-document synthesis (7) | 27.4% | 38.1% | 67.1% | 0.380 | 42.9% |
| unanswerable (6) | 33.3% | 50.0% | 58.3% | 0.368 | 50.0% |
| contradiction (4) | 75.0% | 87.5% | 87.5% | 0.750 | 75.0% |

**Cross-document synthesis is the weakest category on every measure**, and
that is the honest headline of this table: those questions need two to five
documents at once, and the deeper the set the likelier one is missed. This is
where retrieval work would pay off most.

The unanswerable row is not a failure — those questions have no answer, and
their "gold" documents are only where a reader would reasonably look. The
abstention metrics are what judge that behaviour.

## Answer quality

| metric | value | what it means |
|---|---|---|
| correctness | **86.8%** | graded correct by the independent judge; partial counts a half |
| grounded | **100.0%** | every claim carries a citation |
| citation validity | **100.0%** | every citation names a passage that was actually supplied |
| citation relevance | 45.9% | overlap between cited documents and the answer key |
| abstention recall | 100.0% | of the questions with no answer, the share refused |
| abstention precision | 100.0% | of the refusals, the share that were right to refuse |
| contradiction detection | 50.0% | of the questions with a planted conflict, the share reported |
| model failures | 0 | calls that never returned |
| mean tokens per question | 3,576 | |

Verdicts: 47 correct, 5 incorrect, 5 partial.

**Citation relevance is the one weak-looking number that is not a defect.** It
measures overlap between the documents cited and the documents the key lists,
so answering correctly from a document the key did not anticipate scores
badly — and on HotpotQA, where the key names two paragraphs out of 66,000, it
scores near zero while correctness stays high. It is a diagnostic, not a
target, and the clearest illustration of why I report these separately rather
than blending them.

**Abstention is 100% on both measures**: all six unanswerable questions were
refused, and nothing answerable was refused. One caveat, because the figure is
cleaner than the behaviour behind it — in the previous run the system wrongly
refused a two-hop question, and in this run it answered that question and got
it wrong instead. The refusal became a wrong answer rather than a correct one,
so precision improved without the behaviour improving. Read it as "no false
refusals in this run", not as a property of the system.

### By source

| | n | correctness |
|---|---|---|
| authored (operations corpus) | 37 | 90.5% |
| HotpotQA (Wikipedia) | 20 | 80.0% |

Deliberately not averaged: they measure different things over incomparable
data.

### By category

| category | n | correctness | verdicts |
|---|---|---|---|
| simple | 20 | 92.5% | 17 correct, 3 partial |
| multi-step | 20 | 80.0% | 16 correct, 4 incorrect |
| cross-document synthesis | 7 | 85.7% | 5 correct, 2 partial |
| unanswerable | 6 | 100.0% | 6 correct |
| contradiction | 4 | 75.0% | 3 correct, 1 incorrect |

### Every answer that was not fully correct

| question | category | verdict | why |
|---|---|---|---|
| `C2` | contradiction | incorrect | The answer asserts a single ownership claim and does not surface both claims, note ADR-0007, or flag the catalogue as stale as required. |
| `S-cause-INC-2026-02-17` | simple | partial | includes the primary cause but omits the latency and ledger impact |
| `X-downstream` | cross-document synthesis | partial | lists checkout-api and merchant-portal and explains dependency, but omits payment-gateway from the referenced list |
| `X-ledger-blast-radius` | cross-document synthesis | partial | Lists correct services but does not specify the exact teams, only says their owning teams need notification |
| `G6` | simple | partial | Provides the correct 99.9% monthly availability but omits latency and error budget details |
| `S-status-ADR-0010` | simple | partial | states proposed but omits the 'not accepted' part |
| `5a7248665542990c210a40b1` | multi-step | incorrect | Answer does not provide the colloquial name 'Rooster' |
| `5a7c4a6455429907fabeef53` | multi-step | incorrect | adds incorrect extra information about San Francisco, contradicting the reference |
| `5a8b8b3155429949d91db5d7` | multi-step | incorrect | The answer does not state the shared activity 'business' and instead describes owner earnings as a valuation method, missing the reference fact. |
| `5addc7e35542997545bbbdbe` | multi-step | incorrect | does not mention American Samoa or tribal lands, missing the reference facts |

Each is diagnosed in Part 3, including the two where I attempted a fix and the
one where the failure was in the question rather than the system.

## Self-preference, measured

| | correctness |
|---|---|
| independent judge, `openai/gpt-oss-120b` | **86.8%** |
| same-family judge, `gemini-3.5-flash` | 92.1% |
| gap | +5.3% |
| verdicts identical | 87.7% |

The judges agree on 87.7% of verdicts, so the gap is a consistent lean
rather than noise: where they differ, the model related to the generator is
the more generous one.

### Which judge was actually right

A lower score is not automatically a better judge, so I read all
7 disagreements against the answer key.

| question | independent judge | same-family judge |
|---|---|---|
| `C4` | correct | incorrect |
| `S-cause-INC-2026-02-17` | partial | correct |
| `X-downstream` | partial | correct |
| `X-ledger-blast-radius` | partial | correct |
| `S-status-ADR-0010` | partial | correct |
| `5a7248665542990c210a40b1` | incorrect | correct |
| `5a7c4a6455429907fabeef53` | incorrect | correct |

**Five of the seven go to the independent judge and two to the same-family
one.** It is stricter and right more often, but not authoritative, and the two
it loses are instructive. On `C4` it marked the answer correct because the
Postgres version matches the reference, when the thing that question exists to
test is whether the disagreement was *reported* — it graded the fact instead
of the required behaviour. On `S-status-ADR-0010` it withheld full marks
because the answer said "proposed" without also saying "not accepted", which
is pedantry: saying a decision is proposed does convey that it has not been
accepted.

**One disagreement shows the mechanism of self-preference directly, not just
its size.** On a two-hop question the reference answer is a football club's
colloquial name, "Rooster". The system never said it, and the independent
judge marked it incorrect for that reason. The same-family judge marked it
**correct**, reasoning that the answer *"correctly identifies Atlético
Mineiro, whose colloquial name is Rooster"* — it supplied the missing fact
from its own knowledge and then credited the answer for containing it.

That is worth more than the percentage gap. A judge sharing the generator's
training lineage knows the same things the generator knows, which makes it
prone to reading an answer as complete because it can fill the gap itself. It
is why the independent figure is the one reported throughout, and it is
visible only because every answer was graded twice.

This adjudication used the judges' stated reasons and the answer key, not a
re-reading of every answer in full.

## The chunking strategies, compared

Two strategies, each indexed into its own collection, same questions, same
four configurations, same embedding model. The only variable is where the text
was cut. Best configuration for each, `hybrid+rerank`:

| | R@5 | R@10 | R@15 | MRR | every gold document found |
|---|---|---|---|---|---|
| **structural** | 68.6% | 78.9% | **87.9%** | **0.689** | 78.6% |
| **fixed** | 68.6% | **81.5%** | **87.9%** | 0.660 | 78.6% |

**They tie on what matters most and differ in a way that makes sense.**
Recall@15 is identical to the decimal, as is the proportion of questions
retrieving every document they need. Where they part: `fixed` finds more by
rank 10 (+2.6 points), because its 40-word overlap means a fact near a
boundary appears whole in two chunks and has two chances to surface;
`structural` ranks the first correct document higher (MRR 0.689 against
0.660, holding in both corpora separately), because that same duplication puts
two near-identical chunks in the ranking and the second displaces something
else.

**So the default was right, and now for a stated reason rather than an
assumption.** `structural` matches on recall and completeness, ranks better,
and stores **219,758 fewer words** — 3.8% less to embed, store and pay for.

Reproduce with `uv run python scripts/compare_chunking.py`.

---

# Part 3 — Analysis

Three kinds of thing went wrong, and they need different responses: defects in
the **measurement**, a defect in the **question set**, and failures in the
**system**.

## Defects in the measurement

Each produced a believable number that was wrong, which is the specific danger
of evaluating anything: a broken metric does not crash, it reports. All are
fixed, and each now has a regression test.

| what was wrong | how it surfaced | what it would have cost |
|---|---|---|
| Two identifiers offered per retrieved item, so fifteen results became thirty entries and every HotpotQA gold document landed at rank 16 or later | recall read **0.0%** while the rank-unbounded measure said 75% of gold documents were found — figures that cannot both be true | HotpotQA recall reported as 0% when it is 87.5% |
| "Should abstain" keyed on a null reference answer; contradiction questions also carry a null, because no single value is correct | a script printed "7 expect an abstention" when only 3 did | the system **credited for abstaining** on contradiction questions — the opposite of correct |
| The judge handed that same null as "the correct behaviour is to abstain" | the contradiction category scored **0%** while the independent contradiction measure said 50% were handled right | all four contradiction questions marked wrong while two were being answered correctly |
| The judge prompt described each verdict but never named the JSON field it belongs in | a second provider replied `{"score":"correct"}` — valid JSON, wrong key, scored as unscored | invisible while one provider was used, since that one is handed a schema it must obey |
| An unpaced run issued requests as fast as retrieval could feed them | the run's own output: **46 model failures out of 47** | the first answer evaluation produced no usable data at all |
| `--progress` declared but never passed to the harness | no progress file existed after a run | a run interrupted near the end lost every scored question |
| `--compare-judges` declared, and the code reading it lost in a later edit | the flag was accepted and silently ignored | a run claiming to measure self-preference would have measured nothing |
| An unanswerable question's premise written by checking for a word rather than the concept | the system answered it, with a citation that holds up | the system marked wrong for correctly matching a term to its synonym, across two full runs |

Two patterns are worth naming.

**Most were caught by two measurements disagreeing, not by review.** Recall
against gold-found. Contradiction correctness against contradiction detection.
A printed count against the question set. None would have been found by
reading the code, because the code did exactly what it said.

**One was hidden by a check that could not fail.** The pacing fix shipped
alongside a test run piped into `tail`, which reports its own exit code rather
than the test runner's — so a failing suite looked identical to a passing one.
That is also why `tests/test_scripts.py` carries a negative control.

## A defect in the question set

One question was wrong for two runs and was penalising the system for being
right. I report it in full because an answer key is code too.

`G6` asked *"What uptime SLA is defined for merchant-portal?"* and was
classified unanswerable, on the stated premise that no document defines an SLA
for any service. The system answered *"The availability target defined for
merchant-portal is 99.9% measured monthly [S8]"*, with a valid citation.

Checking the corpus: `runbook-merchant-portal.md` says *"Availability target
99.9% measured monthly. p99 latency budget 800ms. Error budget is 43 minutes
per month."* The string `SLA` appears in no operations document — but the
concept does, under different wording. **The system was right and the question
was wrong**, because its premise had been written by searching for the word
rather than the thing.

**The first attempted fix would have made it worse.** Before the corpus was
checked this looked like a failure to abstain, and the response was a prompt
rule: if a question asks for a named artefact and no source states it,
abstain. That would have instructed the model to refuse exactly this question,
and every other question where the corpus uses a synonym. It was reverted.

`G6` is now a `simple` question whose answer is the availability target, so it
tests synonym matching — the thing it actually exercises. A replacement
unanswerable question, `G7`, asks what encryption standard is required for
payment data at rest; I verified the premise by searching for `encrypt`,
`cipher`, `AES`, `PCI` and `at rest`, all of which appear in zero operations
documents. It is a better question than the one it replaces, because a
payments corpus is exactly where such a policy would be expected.

Two further disclosures. The corrected reference answer initially listed the
latency budget as well, which is not uptime, so the run in Part 2 scores `G6`
as partial for omitting a figure the question never asked for; the reference
has since been trimmed and the figures not recomputed, because re-running to
improve a number this report explains is not worth a day's quota. And
changing a question after the system fails it is the shape of score-chasing —
the distinction here is that the question's own premise was false and the
answer was supported by a citation that holds up. It is recorded here, in the
question file, and in the replacement's own text, so it cannot be found later
without the explanation attached.

## Failures in the system

The contradiction category was the one worth working on, because its two
failures have completely different causes — one a reasoning failure a prompt
could move, one a retrieval limit no prompt can.

### Improved but not fixed: a stale record superseded by a later decision

`C2` asks which team owns the settlement queue. The service catalogue says
`owner_team: Payments Core` with `catalogue_version: 2026-01-10`; `ADR-0007`,
dated 2026-02-11 and accepted, transfers ownership to the Platform team and
notes that *"the service catalogue must be updated."*

**Retrieval was not the problem** — both documents arrive at ranks 1 and 2
under every configuration. The system answered with the transfer alone, cited
only the ADR, recorded no contradiction, and never mentioned that the
catalogue a reader would otherwise consult still names the wrong team.

**Diagnosis.** The disagreement rule was written for two sources *stating
different facts*. The model does not classify "an old record plus a notice it
changed" that way: it reads a timeline, resolves it correctly, and reports the
resolution.

**The first prompt change did nothing** — a rule about later dates and
accepted statuses left the answer unchanged, because dates were never the
difficulty. **The second one names the thing itself**: when one source records
current state and another says that state changed, state what the record still
says, state the change, and record that the record is stale. It then produces
answers like *"The service configurations state that the settlement queue is
owned by the Payments Core team [S1][S4]. However, an accepted Architecture
Decision Record (ADR-0007) states that ownership transfers..."* with a
contradiction entry.

**But measured, not assumed:**

| | contradiction reported |
|---|---|
| before the change | 0 of 1 observed, and 0 across two full runs |
| after the change | **3 of 6** |

It is a real improvement over behaviour that never happened, and it is not a
fix. The run in Part 2 is one of the failing samples, which is why
contradiction detection reads 50% there.

**This is the most useful methodological finding in the evaluation.**
Generation is not deterministic, so one successful run cannot distinguish a
fix from a lucky sample — and the pull to accept it is strong, because the
good sample arrives immediately after the change and looks like cause and
effect. The first version of this section claimed a fix on exactly that basis.

**What would actually fix it** is not a longer prompt but making the
requirement checkable rather than requested: the system already computes
whether a contradiction was reported, so generation could be re-run when a
question retrieves a state record and a later decision about the same subject
while the answer reports no conflict. That moves the guarantee out of the
model's discretion — which is where the citation guarantee already lives, and
why citation validity is 100%.

### Not fixable by prompting: a transitive contradiction

`C4` asks which Postgres version the ledger runs on. `ADR-0009` says the
ledger moved to Postgres 16 and that *"Postgres 15 is no longer in use
anywhere in the payments estate"*; `runbook-payment-gateway-v2.md` still says
*"The database is Postgres 15."*

**The conflicting document is absent from the top 15 under all four
configurations** — dense, sparse, fusion and re-ranked alike — so this is not
the re-ranker's doing. Searching the claim itself, `Postgres 15`, does not
surface it either; the deployment log dominates those results.

**Why.** That runbook is about `payment-gateway` and never mentions the
ledger. There is no lexical and no semantic path from "the ledger" to it, and
a system that returned it here would be returning an off-topic document by
luck. The two documents conflict only *transitively*, through the estate-wide
scope of the ADR's claim.

Detecting it needs something the system does not do: after forming a candidate
answer, retrieve on the **claim** rather than the question's subject and check
the result for conflicts. That is the most valuable piece of future work this
evaluation identified — and reporting it as a prompt problem would have been
wrong, which an earlier draft of this report did.

### Not fixed: multi-step reasoning on Wikipedia questions

Four failures, all with the supporting paragraphs retrieved and the chain
between them left incomplete. Three share a shape: the question asks for a
named thing, the name sits in a retrieved paragraph, and the answer does not
state it. In the fourth the answer is right and then adds a claim the
reference contradicts — the only failure of that kind, and the one a citation
check cannot catch, because the added claim cites a passage genuinely
supplied.

Establishing whether this is the system or the model takes one cheap
experiment: re-run those four against a larger generator. The harness supports
it — the generator is a config value and answer scoring is one flag — and it
has not been done because the free tier is the constraint. So this is a stated
limitation rather than a finding.

### Not fixed, deliberately: over-caution

The prompt is emphatic about not answering beyond the evidence — *"partial
evidence is not sufficient evidence"*, *"do not guess, infer beyond the text,
or fill gaps with plausible detail"* — and that is why citation validity is
100% and nothing is fabricated.

Its cost is refusing a question the sources do answer. **In this run that cost
is zero**, with abstention precision at 100%. That is not evidence the
tendency is gone: in the previous run the same instruction produced one false
refusal on a two-hop question, and in this run the system answered that same
question and got it wrong instead. The failure moved rather than improved.

I keep the trade. Softening the instruction would exchange refusals for
fabrications, which are harder to detect and worse to ship. For a system whose
purpose is evidence-grounded answers, refusing too often is the right
direction to be wrong in — and because it is chosen on purpose, it is reported
rather than quietly tuned away.

## Limitations

- **Correctness is graded by a model.** The reported figure comes from a model
  sharing no lineage with the generator, and every answer was graded again by
  one from the generator's family so the difference could be published. Hand
  adjudication did not uniformly favour the independent judge. Read the figure
  as approximate.

- **Generation is not deterministic, and this governs how everything else
  should be read.** One question asked six times produced the required
  behaviour three times. Questions moved between verdicts across full runs
  with no change to the system. So a difference of one or two questions
  carries no information, a prompt change cannot be validated by a single
  sample, and every figure in Part 2 is one draw rather than a fixed property.
  Three figures are stable across every run — grounded, citation validity and
  model failures — and they are stable because they are enforced in code
  rather than requested in a prompt.

- **Eleven of the twenty simple questions are two templates**, six asking
  which team owns a service and five asking an incident's root cause, so that
  category's high score partly reflects two shapes answered repeatedly. It was
  13 questions before and was broadened to 20 for exactly this reason.

- **Six unanswerable questions is a small sample** for a behaviour this
  important, though each probes a different kind of absence.

- **Answer quality is measured for one retrieval configuration.** Retrieval is
  compared across four because it is free; answer scoring costs about 170
  requests, close to a day's whole allowance. So the configuration comparison
  rests on retrieval, and "better retrieval gives better answers" is an
  inference here, not a measurement.

- **The chunking comparison uses one chunk size**, so it compares boundary
  policy rather than size, and it was run on the 56-question set before the
  question-set correction — the strategies were measured against each other
  under identical conditions, so the comparison holds, but its absolute
  figures are not comparable to the retrieval table in Part 2.

- **Earlier figures are not comparable to these.** The set grew from 47 to 57
  questions, a scoring defect was fixed mid-way, and one answer key was
  corrected.

## What to do next, in order of evidence

1. **Claim-scoped retrieval for contradiction detection.** The remaining
   contradiction failure is unreachable by prompting and the design that would
   reach it is specific: after forming a candidate answer, retrieve on the
   claim and check for conflicts. Highest value, and the only item the
   evaluation points at unambiguously.
2. **Enforce contradiction reporting in code**, the way citation validity
   already is, instead of relying on a prompt that works half the time.
3. **Settle whether the multi-step failures are the system's or the model's**
   by re-running those four against a larger generator. Cheap, and decides
   whether further work is warranted.
4. **Improve synthesis retrieval**, the weakest category on every measure —
   either more candidates per route before fusion, or one retrieval per
   sub-question.
5. **Add adversarial documents and test against them.** A document whose text
   instructs the model is the obvious attack on a system that puts retrieved
   text into a prompt, and nothing here tests for it.
