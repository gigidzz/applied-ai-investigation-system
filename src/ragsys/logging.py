"""Logging setup, configured once, centrally.

Bulk ingestion processes tens of thousands of records. When something fails at
record 31,402, print() tells you nothing useful; a timestamped level and module
name tells you where to look.
"""

from __future__ import annotations

import logging
import sys

from ragsys.config import settings

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"


class DropMessagesContaining(logging.Filter):
    """Drops individual log records, leaving the rest of a logger intact.

    Preferred over raising a logger's level when only one of its messages is
    noise. Silencing a whole logger to hide one line also hides the lines you
    would want, which is a trade worth refusing when a filter costs this little.
    """

    def __init__(self, *fragments: str) -> None:
        super().__init__()
        self.fragments = fragments

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in self.fragments)


def setup_logging(level: str | None = None) -> None:
    """Configure root logging. Safe to call repeatedly; only the first call acts.

    Attaching a second handler to the root logger makes every record print
    twice, so the idempotency guard is load-bearing rather than defensive.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level or settings.log_level)

    # Third-party libraries are chatty at INFO; we only want their warnings.
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "urllib3",
        "datasets",
        "huggingface_hub",
        "fsspec",
        "filelock",
        "sentence_transformers",
        "qdrant_client",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # torch reports on every import that process redirects are unsupported on
    # macOS. This project never uses torch.distributed, so the notice can
    # never be relevant and the whole logger is silenced.
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

    # The Hub advises setting a token for faster downloads, on every run, for
    # a model that is already cached. Dropped by message rather than by
    # silencing the logger: the same logger also reports download retries and
    # rate limiting, which are exactly what a deployment that cannot fetch
    # the model would need to say.
    logging.getLogger("huggingface_hub.utils._http").addFilter(
        DropMessagesContaining("unauthenticated requests")
    )

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Call as get_logger(__name__)."""
    return logging.getLogger(name)
