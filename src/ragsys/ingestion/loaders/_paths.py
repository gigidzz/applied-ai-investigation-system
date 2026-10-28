"""Shared helpers for turning a file path into identifiers and citations."""

from __future__ import annotations

from pathlib import Path

from ragsys.config import PROJECT_ROOT, settings


def document_id(path: Path, suffix: str | None = None) -> str:
    """A readable, stable id for a document.

    Derived from the path relative to the corpus directory, so the id doubles
    as a human-readable reference in citations: 'ops/incidents/INC-2026-03-14.pdf'
    rather than an opaque hash. `suffix` distinguishes several documents from
    one file, as in '...csv#row=142' or '...pdf#page=2'.
    """
    try:
        base = path.resolve().relative_to(settings.corpus_dir.resolve()).as_posix()
    except ValueError:
        # Outside the corpus directory: an upload, or a test fixture.
        base = path.name
    return f"{base}#{suffix}" if suffix else base


def citation_path(path: Path) -> str:
    """The path to show a user in a citation.

    Relative to the project root, never absolute. An absolute path would embed
    the machine it was ingested on into every stored chunk, so a corpus built
    on a laptop would cite '/Users/someone/...' once deployed.
    """
    resolved = path.resolve()
    for root in (settings.corpus_dir.resolve(), PROJECT_ROOT.resolve()):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return path.name
