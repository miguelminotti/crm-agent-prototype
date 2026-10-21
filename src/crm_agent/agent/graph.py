from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def build_graph(model: BaseChatModel, tools: Sequence[BaseTool]) -> CompiledStateGraph[MessagesState]:
    """
    Build a graph representation of the interactions between the model and the tools.

    Args:
        model (BaseChatModel): The chat model to use for the agent.
        tools (Sequence[BaseTool]): A sequence of tools available to the agent.

    Returns:
        CompiledStateGraph[MessagesState]: The graph representation of the agent's interactions.
    """

    # Bind tools
    runnable = model.bind_tools(tools)

    def model_node(state: MessagesState) -> dict[str, list[BaseMessage]]:
        answer = runnable.invoke(state["messages"])
        return {"messages": [answer]}

    builder = StateGraph(MessagesState)
    builder.add_node("model", model_node)
    builder.add_edge(START, "model")
    builder.add_node("tools", ToolNode(tools))
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    graph = builder.compile()

    return graph
