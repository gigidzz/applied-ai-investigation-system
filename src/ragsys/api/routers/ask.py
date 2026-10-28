"""POST /api/ask

Three lines of work, and that is correct rather than lazy. Everything this
endpoint could plausibly own lives somewhere better: the retrieval
configuration in `deps`, the error-to-status mapping in the app's exception
handlers, and the response shaping in the schema. A service layer here would
forward one call and add a file.
"""

from __future__ import annotations

from fastapi import APIRouter

from ragsys.api.deps import AgentDep
from ragsys.api.schemas import AskRequest, AskResponse

router = APIRouter(prefix="/api", tags=["ask"])


@router.post("/ask", response_model=AskResponse, summary="Ask a question")
def ask(request: AskRequest, agent: AgentDep) -> AskResponse:
    return AskResponse.from_domain(agent.answer(request.question))
