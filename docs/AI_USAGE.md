# AI usage

I built this project with an AI coding assistant. This document records what I
used it for, how the work was actually divided, where its judgement was wrong,
and which decisions were mine, so that the division of labour is on the record
rather than left to be guessed at.

## Summary

The assistant wrote most of the implementation code and the first draft of most
of the documentation. I set the direction: what to build, in what order, what
to measure, what to reject, and when to stop and verify. Every proposal went
through me before it was acted on, and I had the last word in that phase. On
several occasions I overruled the assistant's recommendation, and in each of
those cases the recommendation turned out to be wrong.

My working rule throughout was that **nothing was built before it had been
explained to me.** Each stage began with the assistant explaining the mechanism
in question — what an embedding is, what BM25 actually computes, why rank
fusion throws the scores away, what a vector index is indexing — and I did not
let the build start until that explanation held up to my questions. After each
change landed I went back through it and had the implementation details walked
through again, against the committed code rather than against the plan. The
point was to be able to defend every part of this system afterwards, not to
produce code quickly.

## How the work was divided

| Area | How it worked |
|---|---|
| Problem framing, scope, stage order | Mine. The assistant proposed an eleven-stage plan; I revised it against the brief and approved the order |
| Architecture | I decided to delegate it. I was inclined towards the MVC layout I knew, asked for the trade-off against a layered package structure, read the reasoning, and accepted it on the merits |
| Implementation code | Written by the assistant, to decisions I had already signed off |
| Reviewing implementation | Mine. After each commit I had the details explained against the code that had landed, and I rejected explanations that did not hold up |
| Test discipline | My standing instruction: no new behaviour without tests. The assistant wrote them |
| Manual verification | Mine. I tested every new feature by hand through the interface, locally and in the deployed service, rather than treating a passing suite as evidence |
| Corpus design | My requirement that the contradictions, gaps and near-duplicates be specified *before* any retrieval code existed, so the evaluation had known answers to find |
| Evaluation design | Proposed by the assistant, then redirected by me: I had the question set strengthened and the whole report rewritten for a human reader rather than a list of metrics |
| Hosting | Mine, after I rejected the assistant's first answer as too narrow and made it compare the alternatives |
| Debugging | The assistant diagnosed; several defects were found only because I refused to accept a result that looked wrong |
| Correctness claims | My standing instruction that nothing be claimed that cannot be reproduced. Every figure in `docs/EVALUATION.md` traces to a committed script and a saved report under `data/eval/` |
| Commit history and attribution | Mine. I approved every commit message and rejected the ones I did not like |

## Technical decisions I made, including against the assistant's advice

**1. Running the embedding model under ONNX Runtime instead of PyTorch.** The
assistant recommended against it: the PyTorch path worked, and the change
carried risk for no functional gain. I pushed back — if the smaller runtime
did the same arithmetic, the memory saving had to matter somewhere. It did.
Resident memory fell from 626 MB to 345 MB, which brought the service under
the 512 MB ceiling of every free hosting tier and is the direct reason this
system is deployed at no cost. I required the vectors to be checked against
the PyTorch output before committing to it; they agreed to 0.999999 cosine
similarity, so the 72,294-vector index did not have to be rebuilt. Had I taken
the recommendation, there would have been no free deployment path.

**2. Keeping the independent judge after it lowered the score.** When grading
moved to a model from a different provider, correctness dropped and I asked
whether that judge was simply worse. I would not accept either answer on
assertion, so I had every disagreement between the two judges adjudicated by
hand against the answer key. The stricter judge turned out to be right more
often, so I kept it. I was not willing to pick the judge that flattered the
system, and not willing to call the stricter one better without checking.

**3. Requiring the chunking strategies to be measured, not assumed.** Running
the comparison cost thirty-five minutes and I was told it would probably just
confirm the existing default. I wanted the number anyway. It did confirm the
default, and it also showed why: the two strategies tie on recall@15 at 87.9%,
`structural` ranks the first correct document higher at MRR 0.689 against
0.660, and it stores 219,758 fewer words. That is a defensible default instead
of an inherited one.

**4. Delegating the package architecture, on the merits.** I was inclined
towards the MVC layout I already knew. I asked for the trade-off against a
layered package structure split by stage — ingestion, indexing, retrieval,
reasoning, evaluation, api — read the reasoning, and accepted it because the
retrieval and reasoning layers genuinely have no business knowing about each
other. The decision to delegate was mine and so was the acceptance.

**5. Where the explanation lives.** I refused to have stage numbers or
project-narrative comments in the source files and required that material to
go into the markdown documents. The code reads as code because I insisted on
it.

**6. Adding and removing documents at runtime.** I raised this as a
requirement that had not been planned for, which is how the upload and delete
lifecycle exists. Raising it late is also what exposed a data-loss bug:
re-uploading an unchanged file deleted it, because deletion ran before an
ingestion that then skipped the file as unchanged.

**7. Testing every new feature by hand, myself.** This is how several defects
were found that no automated test would have caught, because the tests were
written against the same assumptions as the code. I ran each new feature
through the interface rather than taking a passing suite as evidence: I
uploaded documents, asked questions that needed them, deleted them and asked
again; I asked a question with a planted contradiction to see whether both
sources came back; I asked a question with no answer in the corpus to see
whether it refused; and I asked questions in the deployed service, not only
locally, because the container, the cloud index and the model API only meet
each other there. Working through the interface is how I found that the
document table's "chunks" column was ambiguous next to a delete button, and
it is how I checked that deleting a document removed every chunk belonging to
it rather than one row.

## Where the assistant's output was wrong

I keep this section because it is the honest record of what working with these
tools is like, and because most of these were caught by me not accepting the
first answer.

**Edits reported as applied that had silently not been.** Four times, the
assistant matched text that the formatter had since reshaped, so the
replacement found nothing, changed nothing, and reported success. One of these
removed the code behind an evaluation flag, so the flag was accepted and
ignored: the run announced a two-judge comparison and used one judge. There is
now a test, `tests/test_scripts.py`, that reads every script as a syntax tree
and asserts each declared flag is actually read, with a negative control so
the check cannot silently stop checking.

**A test command that could not fail.** The assistant used `pytest | tail -2`,
where the exit code belongs to `tail`, not to the test runner. A failing suite
looked exactly like a passing one, and an evaluation was started on broken
code. The general lesson — a check that cannot report failure is not a check —
is now written into the evaluation report, because it is the single most useful
thing this project taught me about testing.

**A failure diagnosed backwards.** The assistant reported that the system had
wrongly answered a question about an uptime SLA and proposed a prompt rule to
make it abstain more readily. I asked for the corpus to be checked first. The
documents do state an availability target of 99.9% with an error budget — the
concept is there under different wording. The system had been right, the
question was wrong, and the proposed rule would have turned a correct synonym
match into a refusal across the whole question set. The rule was reverted and
the question was rewritten.

**A claim in the report that the evidence did not support.** The report stated
that both missed contradictions had all their documents already in the prompt,
and therefore needed only a prompt change. Checking it showed that for one of
them the conflicting document is absent from the top fifteen under all four
retrieval configurations, so no prompt could have fixed it. That failure is a
retrieval limitation and is now described as one.

**A fix declared from a single successful run.** The assistant reported the
contradiction fix as working after it worked once. Running the same question
six times showed the improved behaviour in three of them. It is a real
improvement over never doing it, and it is not a fix, and the report now says
so with the count.

**Recommending against the change that made deployment possible.** Covered
above, and listed again here because it is the clearest case of the assistant
being confidently wrong on a judgement call rather than on a detail.

**Declaring the work finished before it was.** "The evaluation is done"
preceded several rounds of discovering that it was not — numbers left over
from an earlier run, a scoring defect that credited abstention on the wrong
questions, and whole sections still describing a 47-question set after it had
grown to 57. I stopped treating completion claims as information fairly early.

## What I did not use it for

- No figure in the evaluation report is estimated, typed in by hand, or quoted
  from memory. Every number is produced by a committed script and written to a
  report file under `data/eval/`.
- Answers are graded by a model from a different provider than the one
  generating them, and the difference between the two judges is published
  rather than hidden. Where they disagreed, I had the cases adjudicated by hand
  against the answer key, and the result went against the preferred judge in
  two of them.
- The assistant is not named in the commit history or anywhere in the
  repository. This document is the project's statement of AI usage, which is
  where that information belongs.

## My assessment

The assistant was most valuable where the work was mechanical and I had
already decided what should happen: loaders for six file formats, a 262-test
suite, the metric harness, and first drafts long enough that reviewing them
was faster than writing them.

It was least reliable exactly where it sounded most confident — recommending
against the one change that made free deployment possible, reporting edits it
had not made, declaring a fix from one sample, and announcing the work finished
three times before it was. The review process described in this document is
not incidental to the result. Without it this project would have shipped with a
silently single-judge evaluation, a data-loss bug on re-upload, no chunking
comparison, an unreproducible set of figures, a prompt rule that broke a
correct answer, and a report claiming a retrieval failure could be fixed with
better prompting.
