from functools import cache
from typing import Annotated

from fastapi import Depends, FastAPI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from crm_agent.agent.chat_models import build_chat_anthropic_model
from crm_agent.agent.graph import build_graph
from crm_agent.crm.repository import InMemoryCRMRepository
from crm_agent.tools.crm_tools import make_crm_tools

app = FastAPI()


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    answer: str
    pending_approval: str | None


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
def create_message(
    thread_id: str,
    body: MessageRequest,
    graph: Annotated[CompiledStateGraph[MessagesState], Depends(get_graph)],
) -> MessageResponse:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
    result = graph.invoke({"messages": [HumanMessage(content=body.message)]}, config)

    pending_approval = result["__interrupt__"][0].value["question"] if "__interrupt__" in result else None
    return MessageResponse(answer=result["messages"][-1].text, pending_approval=pending_approval)
