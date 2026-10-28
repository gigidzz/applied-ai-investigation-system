"""The retrieval pipeline: two searches, one fusion, one re-scoring pass.

Everything that decides which chunks reach the prompt lives behind one call,
so the interface, the evaluation harness and the reasoning stage cannot drift
apart by each assembling the steps differently.

Configuration is a value object rather than module state, because comparing
configurations is a requirement: dense-only against hybrid, and fusion alone
against fusion plus signals, have to run side by side in one process.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date

from ragsys.config import settings
from ragsys.indexing.vector_store import Hit, QdrantChunkStore
from ragsys.logging import get_logger
from ragsys.retrieval.fusion import DEFAULT_RRF_K, Candidate, reciprocal_rank_fusion
from ragsys.retrieval.signals import (
    QuestionKind,
    RerankWeights,
    classify_question,
    document_kind,
    metadata_agreement,
    recency,
    type_preference,
)

log = get_logger(__name__)

DENSE_ROUTE = "dense"
SPARSE_ROUTE = "sparse"

# A single document may not be discounted into irrelevance: a question whose
# answer genuinely spans several chunks of one report still needs them.
MAX_CROWDING_PENALTY = 0.75


@dataclass(frozen=True)
class RetrievalConfig:
    """One named way of retrieving, so configurations can be compared."""

    name: str = "hybrid+rerank"
    top_k: int = field(default_factory=lambda: settings.retrieval_top_k)
    final_k: int = field(default_factory=lambda: settings.rerank_top_k)
    rrf_k: int = DEFAULT_RRF_K
    weights: RerankWeights = field(default_factory=RerankWeights)
    route_weights: dict[str, float] | None = None
    use_dense: bool = True
    use_sparse: bool = True
    reference_date: date | None = None

    def variant(self, name: str, **changes) -> RetrievalConfig:
        return replace(self, name=name, **changes)


# The configurations the evaluation harness compares. Each isolates one
# decision, so a difference in the numbers points at a single cause.
DENSE_ONLY = RetrievalConfig(name="dense-only", use_sparse=False, weights=RerankWeights.off())
SPARSE_ONLY = RetrievalConfig(name="sparse-only", use_dense=False, weights=RerankWeights.off())
HYBRID_FUSION_ONLY = RetrievalConfig(name="hybrid-fusion", weights=RerankWeights.off())
HYBRID_RERANKED = RetrievalConfig(name="hybrid+rerank")


@dataclass
class RetrievalResult:
    """What was returned, and enough detail to explain why."""

    question: str
    kind: QuestionKind
    config_name: str
    candidates: list[Candidate]
    route_counts: dict[str, int] = field(default_factory=dict)
    considered: int = 0

    @property
    def documents(self) -> list[str]:
        """Distinct source documents, in rank order."""
        seen: dict[str, None] = {}
        for candidate in self.candidates:
            seen.setdefault(candidate.doc_id, None)
        return list(seen)

    def as_context(self) -> list[dict]:
        """The shape the reasoning stage needs: text plus citation fields."""
        return [
            {
                "label": f"S{position}",
                "chunk_id": candidate.chunk_id,
                "doc_id": candidate.doc_id,
                "source_path": candidate.hit.source_path,
                "text": candidate.text,
                "score": round(candidate.final_score, 6),
                "date": candidate.payload.get("meta_date"),
                "service": candidate.payload.get("meta_service"),
                "kind": document_kind(candidate.payload),
            }
            for position, candidate in enumerate(self.candidates, start=1)
        ]


class Retriever:
    """Runs the configured routes, fuses them, then re-scores."""

    def __init__(self, store: QdrantChunkStore, config: RetrievalConfig | None = None) -> None:
        self.store = store
        self.config = config or HYBRID_RERANKED

    def retrieve(self, question: str, config: RetrievalConfig | None = None) -> RetrievalResult:
        cfg = config or self.config
        rankings: dict[str, list[Hit]] = {}

        if cfg.use_dense:
            rankings[DENSE_ROUTE] = self.store.search_dense(question, limit=cfg.top_k)
        if cfg.use_sparse:
            rankings[SPARSE_ROUTE] = self.store.search_sparse(question, limit=cfg.top_k)
        if not rankings:
            raise ValueError("a configuration must enable at least one route")

        fused = reciprocal_rank_fusion(rankings, k=cfg.rrf_k, weights=cfg.route_weights)
        kind = classify_question(question)
        self._score(question, kind, fused, cfg)
        selected = self._select(fused, cfg)

        return RetrievalResult(
            question=question,
            kind=kind,
            config_name=cfg.name,
            candidates=selected,
            route_counts={route: len(hits) for route, hits in rankings.items()},
            considered=len(fused),
        )

    def _score(
        self,
        question: str,
        kind: QuestionKind,
        candidates: list[Candidate],
        cfg: RetrievalConfig,
    ) -> None:
        """Apply the per-chunk signals. Crowding is deferred: it is positional."""
        weights = cfg.weights
        for candidate in candidates:
            agreement = metadata_agreement(question, candidate.payload)
            preference = type_preference(kind, candidate.payload)
            freshness = recency(candidate.payload, cfg.reference_date)
            multiplier = (
                1.0
                + weights.metadata * agreement
                + weights.doc_type * preference
                + weights.recency * freshness
            )
            # Signals may demote but never invert a ranking into nonsense.
            multiplier = max(multiplier, 0.05)
            candidate.signals = {
                "metadata": round(agreement, 4),
                "doc_type": round(preference, 4),
                "recency": round(freshness, 4),
                "multiplier": round(multiplier, 4),
            }
            candidate.final_score = candidate.fusion_score * multiplier

    def _select(self, candidates: list[Candidate], cfg: RetrievalConfig) -> list[Candidate]:
        """Greedy selection with a growing penalty for repeating a document.

        Greedy rather than a single sort because the penalty depends on what
        has already been chosen: the cost of a chunk is not a property of the
        chunk, it is a property of the chunk given the current result set.
        """
        pool = sorted(candidates, key=lambda c: (-c.final_score, c.chunk_id))
        if cfg.weights.crowding <= 0.0:
            chosen = pool[: cfg.final_k]
            for candidate in chosen:
                candidate.signals["crowding"] = 0.0
            return chosen

        selected: list[Candidate] = []
        per_document: Counter[str] = Counter()
        remaining = list(pool)

        while remaining and len(selected) < cfg.final_k:
            best_index = 0
            best_score = float("-inf")
            best_penalty = 0.0
            for position, candidate in enumerate(remaining):
                penalty = min(
                    cfg.weights.crowding * per_document[candidate.doc_id],
                    MAX_CROWDING_PENALTY,
                )
                adjusted = candidate.final_score * (1.0 - penalty)
                if adjusted > best_score:
                    best_score = adjusted
                    best_index = position
                    best_penalty = penalty
            winner = remaining.pop(best_index)
            winner.signals["crowding"] = round(best_penalty, 4)
            winner.final_score = best_score
            per_document[winner.doc_id] += 1
            selected.append(winner)

        return selected
