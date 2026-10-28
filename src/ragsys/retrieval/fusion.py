"""Merging rankings whose scores cannot be compared.

Cosine similarity is bounded between -1 and 1; BM25 is unbounded and scales
with query length and corpus statistics. A measured example: for the query
`pay-gw-4.2.1` the best dense score was 0.777 and the best sparse score was
39.917. Adding them lets the sparse score decide everything, and the same
query's paraphrase counterpart showed sparse retrieval confidently returning
Wikipedia paragraphs about card games — so letting it dominate is not a
theoretical concern.

Normalising each list per query is the other obvious repair and it is fragile:
the normaliser depends on the top score, so a query where every result is
mediocre has its best mediocre result rescaled to look excellent.

Reciprocal rank fusion sidesteps the problem by discarding the scores and
keeping only positions. Its useful property is that it rewards *agreement*:
a chunk both routes rank highly beats a chunk either one ranks first alone,
which is the right trade when each route is known to be confidently wrong on
different kinds of question.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ragsys.indexing.vector_store import Hit

# The constant in 1/(k + rank). At 60 the gap between rank 1 and rank 3 is
# about 3%, so appearing in both rankings outweighs topping one of them. A
# much smaller k would make each route's first place nearly decisive, which
# defeats the purpose of fusing at all.
DEFAULT_RRF_K = 60


@dataclass
class Candidate:
    """One chunk under consideration, with the evidence for its position.

    Every intermediate number is kept rather than collapsed into one score,
    because the interface has to show why a source was chosen, and because a
    ranking that cannot be explained cannot be debugged.
    """

    hit: Hit
    ranks: dict[str, int] = field(default_factory=dict)
    fusion_score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0

    @property
    def chunk_id(self) -> str:
        return self.hit.chunk_id

    @property
    def doc_id(self) -> str:
        return self.hit.doc_id

    @property
    def text(self) -> str:
        return self.hit.text

    @property
    def payload(self) -> dict:
        return self.hit.payload

    @property
    def routes(self) -> tuple[str, ...]:
        """Which retrieval routes returned this chunk."""
        return tuple(sorted(self.ranks))

    @property
    def agreed(self) -> bool:
        """True when more than one route returned it: the strongest signal."""
        return len(self.ranks) > 1


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Hit]],
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[Candidate]:
    """Fuse named rankings into one, ordered best first.

    `weights` lets one route count for more than another, which the evaluation
    harness uses to compare a balanced hybrid against a dense-leaning one
    without changing any other part of the pipeline.
    """
    merged: dict[str, Candidate] = {}
    for route, hits in rankings.items():
        weight = 1.0 if weights is None else weights.get(route, 1.0)
        for position, hit in enumerate(hits, start=1):
            candidate = merged.get(hit.chunk_id)
            if candidate is None:
                candidate = Candidate(hit=hit)
                merged[hit.chunk_id] = candidate
            candidate.ranks[route] = position
            candidate.fusion_score += weight / (k + position)

    ordered = sorted(merged.values(), key=lambda c: (-c.fusion_score, c.chunk_id))
    for candidate in ordered:
        candidate.final_score = candidate.fusion_score
    return ordered
