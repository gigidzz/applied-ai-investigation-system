"""The tool the model is allowed to call.

One tool, deliberately. Tool calling earns its place here because a question
like "which change caused the March disruption and which services were
affected" needs two different searches, and only the model knows the second
one after reading the result of the first. Giving it a search function lets it
make that decision instead of us guessing the sub-queries in advance.

The tool returns formatted text rather than objects because that is what goes
back into the conversation, and it registers every chunk it returns so the new
sources carry labels that continue the existing numbering.
"""

from __future__ import annotations

from typing import Any

from ragsys.logging import get_logger
from ragsys.retrieval.retriever import Retriever

log = get_logger(__name__)

SEARCH_TOOL_NAME = "search_documents"

SEARCH_TOOL: dict[str, Any] = {
    "name": SEARCH_TOOL_NAME,
    "description": (
        "Search the document collection for passages relevant to a query. "
        "Use it when the supplied sources do not contain something you need, "
        "for example a second fact required to answer a multi-part question. "
        "Returns labelled sources you can cite."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": (
                    "What to look for, as a short phrase or question. Name "
                    "specific services, releases or incident identifiers when "
                    "you know them."
                ),
            }
        },
        "required": ["query"],
    },
}


class SourceRegister:
    """Hands out source labels and remembers what each one refers to.

    Labels must stay stable for the whole conversation: a chunk the model saw
    as S4 in the first turn has to still be S4 after a tool call adds more,
    or the citations it writes will point at the wrong documents.
    """

    def __init__(self) -> None:
        self._by_chunk: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def add(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Register entries, relabelling them, and return only the new ones."""
        fresh: list[dict[str, Any]] = []
        for entry in entries:
            chunk_id = str(entry.get("chunk_id", ""))
            if chunk_id in self._by_chunk:
                continue
            labelled = dict(entry)
            labelled["label"] = f"S{len(self._order) + 1}"
            self._by_chunk[chunk_id] = labelled
            self._order.append(chunk_id)
            fresh.append(labelled)
        return fresh

    @property
    def all(self) -> list[dict[str, Any]]:
        return [self._by_chunk[chunk_id] for chunk_id in self._order]

    def __len__(self) -> int:
        return len(self._order)


class ToolBox:
    """Runs the tools the model asks for."""

    def __init__(self, retriever: Retriever, register: SourceRegister, limit: int = 6) -> None:
        self.retriever = retriever
        self.register = register
        self.limit = limit

    @property
    def declarations(self) -> list[dict[str, Any]]:
        return [SEARCH_TOOL]

    def search(self, query: str) -> list[dict[str, Any]]:
        """Retrieve for a model-supplied query and register anything new."""
        result = self.retriever.retrieve(query)
        return self.register.add(result.as_context()[: self.limit])
