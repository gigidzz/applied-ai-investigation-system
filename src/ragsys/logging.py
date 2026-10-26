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
    for noisy in ("httpx", "urllib3", "sentence_transformers", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Call as get_logger(__name__)."""
    return logging.getLogger(name)
