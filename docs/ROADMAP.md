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
| 1 | Corpus | Acquire and normalise the document set; build the labelled evaluation question set; profile the data before designing against it | next |
| 2 | Ingestion | Per-format loaders, metadata extraction, two comparable chunking strategies, duplicate detection, add/update/delete | |
| 3 | Indexing | Embedding model, vector store schema with dense and sparse vectors, idempotent batch upsert | |
| 4 | Retrieval | Semantic search, keyword search, custom fusion and ranking, optional cross-encoder rerank | |
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
