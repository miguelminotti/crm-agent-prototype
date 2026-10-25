import asyncio
from functools import cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from pydantic import BaseModel

from crm_agent.agent.chat_models import build_chat_anthropic_model
from crm_agent.agent.graph import ApprovalDecision, build_graph
from crm_agent.crm.repository import InMemoryCRMRepository
from crm_agent.tools.crm_tools import make_crm_tools

app = FastAPI()


class PendingToolCallDTO(BaseModel):
    name: str
    args: dict[str, Any]
    id: str | None


class PendingApprovalDTO(BaseModel):
    tool_calls: list[PendingToolCallDTO]


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    answer: str | None
    pending_approval: PendingApprovalDTO | None


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    reason: str | None = None


_thread_locks: dict[str, asyncio.Lock] = {}


# Cached: built once and shared by every request, so the repo and the checkpointer (memory) survive between requests
@cache
def get_graph() -> CompiledStateGraph[MessagesState]:
    repo = InMemoryCRMRepository()
    return build_graph(
        build_chat_anthropic_model(),
        make_crm_tools(repo),
        checkpointer=InMemorySaver(),
        approval_required_tools=["create_company", "create_contact"],
    )


@app.post("/threads/{thread_id}/messages")
async def create_message(
    thread_id: str,
    body: MessageRequest,
    graph: Annotated[CompiledStateGraph[MessagesState], Depends(get_graph)],
) -> MessageResponse:
    lock = _acquire_thread_lock_or_409(thread_id)

    async with lock:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
        state: StateSnapshot = await graph.aget_state(config)
        if state.interrupts:
            raise HTTPException(status_code=409, detail="Pending approval for this thread")

        result = await graph.ainvoke({"messages": [HumanMessage(content=body.message)]}, config)
        return _to_response(result)


@app.post("/threads/{thread_id}/approval")
async def approve_message(
    thread_id: str,
    body: ApprovalDecisionRequest,
    graph: Annotated[CompiledStateGraph[MessagesState], Depends(get_graph)],
) -> MessageResponse:
    lock = _acquire_thread_lock_or_409(thread_id)

    async with lock:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
        state: StateSnapshot = await graph.aget_state(config)
        if state.created_at is None:
            raise HTTPException(status_code=404, detail="Thread not found")

        if not state.interrupts:
            raise HTTPException(status_code=409, detail="No pending approval for this thread")

        result = await graph.ainvoke(
            Command(resume=ApprovalDecision(approved=body.approved, reason=body.reason)), config
        )
        return _to_response(result)


def _to_response(result: dict[str, Any] | Any) -> MessageResponse:
    interrupt_present = "__interrupt__" in result
    pending_approval = (
        PendingApprovalDTO.model_validate(result["__interrupt__"][0].value) if interrupt_present else None
    )
    answer = None if interrupt_present else result["messages"][-1].text
    return MessageResponse(answer=answer, pending_approval=pending_approval)


def _acquire_thread_lock_or_409(thread_id: str) -> asyncio.Lock:
    if thread_id not in _thread_locks:
        _thread_locks[thread_id] = asyncio.Lock()
    elif _thread_locks[thread_id].locked():
        raise HTTPException(status_code=409, detail="Thread is currently locked")
    return _thread_locks[thread_id]
