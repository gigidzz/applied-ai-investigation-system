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

## Layout

```
src/ragsys/        the package
  config.py        all configuration, declared once and validated at import
  logging.py       central logging setup
tests/             pytest suite
scripts/           runnable entry points for ingestion and evaluation
docs/              architecture and planning documents
data/              corpora and evaluation sets; gitignored, rebuilt by scripts
```

## Configuration

All settings are declared in `src/ragsys/config.py` and may be overridden by
environment variables or by `.env`. Secrets never enter version control:
`.env` is gitignored, and `.env.example` documents the expected keys with
empty values.

Credentials are read through `require()`, which fails immediately with the
name of the missing variable rather than allowing a `None` to propagate.
