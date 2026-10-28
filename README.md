# ragsys

An agentic retrieval system for question answering over large, heterogeneous
document collections.

The system answers questions whose answers are not contained in any single
document: it decomposes the question, retrieves across the corpus, connects
facts from separate sources, cites every claim, flags contradictions between
sources, and declines to answer when the corpus does not support one.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the development plan and the
rationale behind the fixed design decisions.

## Stack

| Concern | Choice |
|---|---|
| Language | Python 3.12, pinned `>=3.12,<3.13` |
| Environment & dependencies | [uv](https://docs.astral.sh/uv/) with a committed `uv.lock` |
| Configuration | pydantic-settings, `.env`-driven, secrets as `SecretStr` |
| Lint & format | ruff |
| Tests | pytest |

## Setup

```bash
# install uv once
brew install uv

# create the environment exactly as locked
uv sync

# configure
cp .env.example .env

# verify
uv run pytest
uv run ruff check .
```

No credentials or network access are required to run the test suite.

## Running it locally

Everything below assumes `uv sync` has been run and `.env` holds a
`QDRANT_URL`, a `QDRANT_API_KEY` and an `LLM_API_KEY`.

### 1. Build the index, once

```bash
uv run python scripts/build_chunks.py      # 87 files  -> 144,735 chunks   (~6 s)
uv run python scripts/build_index.py       # embed and upload             (~16 min)
```

The second step is the slow one: it runs every chunk through the embedding
model on CPU and uploads the vectors. It is resumable with `--resume`, and
only needs doing again when the corpus or the embedding model changes.

### 2. Ask a question from the terminal

```bash
uv run python scripts/ask.py "which team owns the ledger service"
```

Add `--json` to see the whole answer object: citations, every retrieved
source, token counts and the model that answered.

### 3. Run the web interface

```bash
uv run uvicorn ragsys.api.app:app --reload --port 8000
```

Then open **http://127.0.0.1:8000**. The page has three parts: ask a question
and read the answer with clickable citations, click a citation to inspect the
chunk behind it, and upload or delete documents.

The first question takes about twenty-five seconds longer than the rest,
because the embedding model loads on first use. `GET /api/health` reports
`model_loaded` so you can see whether that has happened yet.

### 4. Check the evaluation

```bash
uv run python scripts/build_test_set.py    # assemble the 47 questions
uv run python scripts/evaluate.py          # retrieval, four configurations (free)
uv run python scripts/evaluate.py --answers  # answer quality (~150 model calls)
```

Retrieval scoring needs no model and takes seconds. Answer scoring is close to
a day's free-tier allowance, so it is opt-in and resumable.

Results and the failure analysis are in [docs/EVALUATION.md](docs/EVALUATION.md).

### The HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/ask` | a question in, a cited answer out |
| `GET` | `/api/documents` | source files in the index |
| `POST` | `/api/documents` | upload and index a file |
| `DELETE` | `/api/documents?source_path=…` | remove a file and its chunks |
| `GET` | `/api/health` | index size and model readiness |

Interactive documentation is generated at **http://127.0.0.1:8000/docs**.

### Logs

Logs go to stderr, which is where container platforms collect them; there is
no log file by design, since a file inside an ephemeral container is written
for nobody to read. Set `LOG_LEVEL=INFO` in `.env` to see retrieval decisions,
tool searches and model fallbacks, or `DEBUG` for everything.

## Layout

```
src/ragsys/
  config.py        all configuration, declared once and validated at import
  logging.py       central logging setup
  ingestion/       loaders, metadata, chunking, change detection
  indexing/        embeddings, BM25 sparse vectors, the Qdrant store
  retrieval/       rank fusion and the custom ranking signals
  reasoning/       the model client, prompts, tool calling, citations
  evaluation/      metrics, the question set, the judge, the harness
  api/             routers, schemas, dependencies, and the web page
tests/             pytest suite
scripts/           runnable entry points for ingestion and evaluation
docs/              architecture, evaluation report, planning
data/              corpora and evaluation sets; gitignored, rebuilt by scripts
```

## Configuration

All settings are declared in `src/ragsys/config.py` and may be overridden by
environment variables or by `.env`. Secrets never enter version control:
`.env` is gitignored, and `.env.example` documents the expected keys with
empty values.

Credentials are read through `require()`, which fails immediately with the
name of the missing variable rather than allowing a `None` to propagate.
