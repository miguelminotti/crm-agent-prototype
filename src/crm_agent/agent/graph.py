from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt


def build_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    approval_required_tools: Sequence[str] | None = None,
) -> CompiledStateGraph[MessagesState]:
    """
    Build a graph representation of the interactions between the model and the tools.

    Args:
        model (BaseChatModel): The chat model to use for the agent.
        tools (Sequence[BaseTool]): A sequence of tools available to the agent.
        checkpoint (BaseCheckpointSaver[str] | None, optional): The checkpoint saver to use for the graph compilation.
        Defaults to None.

    Returns:
        CompiledStateGraph[MessagesState]: The graph representation of the agent's interactions.
    """

    # Bind tools
    runnable = model.bind_tools(tools)

    def model_node(state: MessagesState) -> dict[str, list[BaseMessage]]:
        answer = runnable.invoke(state["messages"])
        return {"messages": [answer]}

    def route_node(state: MessagesState) -> str:
        model_answer: AIMessage | None = (
            state["messages"][-1] if state["messages"] and isinstance(state["messages"][-1], AIMessage) else None
        )
        if model_answer is None or not model_answer.tool_calls: # TODO: fix the multiple tool_calls scenario
            return END
        elif approval_required_tools is None or all(
            call["name"] not in approval_required_tools for call in model_answer.tool_calls
        ):
            return "tools"
        elif approval_required_tools is not None and all(
            call["name"] in approval_required_tools for call in model_answer.tool_calls
        ):
            return "approval"
        else:
            return "tools"

    def approval_node(state: MessagesState) -> dict[str, list[BaseMessage]]:
        model_answer: AIMessage | None = (
            state["messages"][-1] if state["messages"] and isinstance(state["messages"][-1], AIMessage) else None
        )
        tool_calls = model_answer.tool_calls if model_answer else []
        tool_calls_str = ", ".join(f"{call['name']}: {call['args']}" for call in tool_calls)
        interrupt(
            {
                "question": f"Você aprova essa escrita no CRM? {tool_calls_str}",
            }
        )
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("model", model_node)
    builder.add_edge(START, "model")
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("approval", approval_node)
    builder.add_conditional_edges("model", route_node)
    builder.add_edge("approval", "tools")
    builder.add_edge("tools", "model")

    graph = builder.compile(checkpointer=checkpointer)

    return graph
