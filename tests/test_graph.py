import json
from collections.abc import Sequence

import pytest
from conftest import FakeMessagesListChatModelBindedTools
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.messages.utils import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import ValidationError

from crm_agent.agent.graph import build_graph
from crm_agent.crm.repository import InMemoryCRMRepository


def test_run_graph_with_fake_model(repo: InMemoryCRMRepository, crm_tools: Sequence[BaseTool]) -> None:
    # Turn 0. "Cadastra a Maria da Acme" - Human
    # Turn 1. tool_calls - find_companies - AIMessage
    # Turn 2. content JSON with 'Acme' company - ToolMessage
    # Turn 3. tool_calls - create_contact - AIMessage
    # Turn 4. content JSON with new contact - ToolMessage
    # Turn 5. content "Maria Cadastrada!" - AIMessage

    # Arrange
    acme_company = repo.create_company(name="Acme LTDA")
    fake_messages: Sequence[AIMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    name="find_companies",
                    args={
                        "query": "Acme",
                    },
                    id="c1",
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    name="create_contact",
                    args={
                        "name": "Maria",
                        "company_id": acme_company.id,
                    },
                    id="c2",
                )
            ],
        ),
        AIMessage(content="Maria Cadastrada!"),
    ]
    fake = FakeMessagesListChatModelBindedTools(responses=fake_messages)

    # Act
    history: list[AnyMessage] = [HumanMessage(content="Cadastra a Maria da Acme")]
    graph = build_graph(fake, crm_tools)
    result = graph.invoke({"messages": history}, {"recursion_limit": 20})
    result_messages = result["messages"]

    # Assert
    expected_types_list = [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    history_types_list = [type(msg) for msg in result_messages]
    assert history_types_list == expected_types_list, "Message type sequence does not match expected sequence"
    if isinstance(result_messages[4].content, str):
        json_result = json.loads(str(result_messages[4].content))
        assert json_result["company_id"] == str(acme_company.id)
    else:
        raise TypeError("Expected result_messages[4].content to be a string containing JSON.")


def test_run_graph_with_inexistent_tool(crm_tools: Sequence[BaseTool]) -> None:
    # Arrange
    fake_messages: Sequence[AIMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    name="non_existent_tool",
                    args={},
                    id="c1",
                )
            ],
        ),
        AIMessage(content="The tool does not exist"),
    ]
    fake = FakeMessagesListChatModelBindedTools(responses=fake_messages)

    # Act & Assert
    history: list[AnyMessage] = [HumanMessage(content="Try to use a non-existent tool")]
    graph = build_graph(fake, crm_tools)
    result = graph.invoke({"messages": history}, {"recursion_limit": 20})
    result_messages = result["messages"]
    assert isinstance(result_messages[-2], ToolMessage)
    assert result_messages[-2].status == "error"


def test_run_graph_with_invalid_tool_args(crm_tools: Sequence[BaseTool]) -> None:
    # Arrange
    fake_messages: Sequence[AIMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    name="create_contact",
                    args={
                        "name": "",
                    },
                    id="c1",
                )
            ],
        ),
        AIMessage(content="Invalid tool arguments"),
    ]
    fake = FakeMessagesListChatModelBindedTools(responses=fake_messages)

    # Act & Assert
    history: list[AnyMessage] = [HumanMessage(content="Try to create a contact with invalid company_id")]
    graph = build_graph(fake, crm_tools)
    result = graph.invoke({"messages": history}, {"recursion_limit": 20})
    result_messages = result["messages"]
    assert isinstance(result_messages[-2], ToolMessage)
    assert result_messages[-2].status == "error"


def test_run_graph_with_forever_asking_for_tools_loop(crm_tools: Sequence[BaseTool]) -> None:
    # Arrange
    fake_messages: Sequence[AIMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    name="create_contact",
                    args={
                        "name": "John Doe",
                    },
                    id="c1",
                )
            ],
        )
        for _ in range(100)
    ]
    fake = FakeMessagesListChatModelBindedTools(responses=fake_messages)

    # Act & Assert
    history: list[AnyMessage] = [HumanMessage(content="Try to create a contact")]
    graph = build_graph(fake, crm_tools)
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": history}, {"recursion_limit": 20})


# Test memory between turns
# On thread "t1":
# Turn 1 (Human): "oi"
# Turn 1 (AI): "Olá, tudo bem?"
# Turn 2 (Human): "cadastra a Acme LTDA"
# Turn 2 (AI): tool - create_company
# Turn 2 (Tool message): create_company with args {"name": "Acme LTDA"}
# Turn 2 (AI): "Empresa Acme LTDA cadastrada com sucesso"
# Rules:
# `invoke` receives only the new message and the memory should retain previous context
def test_memory_between_turns(crm_tools: Sequence[BaseTool]) -> None:
    # Arrange
    fake_messages: Sequence[AIMessage] = [
        AIMessage(content="Olá, tudo bem?"),
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    name="create_company",
                    args={
                        "name": "Acme LTDA",
                    },
                    id="c1",
                )
            ],
        ),
        AIMessage(content="Empresa Acme LTDA cadastrada com sucesso"),
    ]
    fake = FakeMessagesListChatModelBindedTools(responses=fake_messages)

    # Act & Assert
    graph = build_graph(fake, crm_tools, checkpointer=InMemorySaver())

    config: RunnableConfig = {"configurable": {"thread_id": "t1"}, "recursion_limit": 20}

    result = graph.invoke({"messages": [HumanMessage(content="oi")]}, config)
    result_messages = result["messages"]
    assert isinstance(result_messages[-1], AIMessage)
    assert result_messages[-1].content == "Olá, tudo bem?"

    result = graph.invoke({"messages": [HumanMessage(content="cadastra a Acme LTDA")]}, config)
    result_messages = result["messages"]
    assert isinstance(result_messages[-2], ToolMessage)
    assert result_messages[-2].status == "success"
    assert isinstance(result_messages[-1], AIMessage)
    assert result_messages[-1].content == "Empresa Acme LTDA cadastrada com sucesso"

    expected_types_list = [HumanMessage, AIMessage, HumanMessage, AIMessage, ToolMessage, AIMessage]
    actual_types_list = [type(msg) for msg in result_messages]
    assert actual_types_list == expected_types_list


def test_thread_isolation(crm_tools: Sequence[BaseTool]) -> None:
    # Arrange
    ai_message = AIMessage(content="Olá, tudo bem?")
    fake_messages: Sequence[AIMessage] = [
        ai_message,
    ]
    fake = FakeMessagesListChatModelBindedTools(responses=fake_messages)

    graph = build_graph(fake, crm_tools, checkpointer=InMemorySaver())

    config_t1: RunnableConfig = {"configurable": {"thread_id": "t1"}, "recursion_limit": 20}

    result_t1 = graph.invoke({"messages": [HumanMessage(content="oi")]}, config_t1)

    assert result_t1["messages"][-1].content == "Olá, tudo bem?"

    config_t2: RunnableConfig = {"configurable": {"thread_id": "t2"}, "recursion_limit": 20}

    human_message_t2 = HumanMessage(content="bom dia")

    result_t2 = graph.invoke({"messages": [human_message_t2]}, config_t2)

    assert result_t2["messages"][-1] == ai_message
    assert result_t2["messages"][-2] == human_message_t2


def test_recursion_limit_preservation(crm_tools: Sequence[BaseTool]) -> None:
    # Arrange
    fake_messages: Sequence[AIMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    name="create_contact",
                    args={
                        "name": "John Doe",
                    },
                    id="c1",
                )
            ],
        )
        for _ in range(100)
    ]
    fake = FakeMessagesListChatModelBindedTools(responses=fake_messages)

    graph = build_graph(fake, crm_tools, checkpointer=InMemorySaver())

    config: RunnableConfig = {"configurable": {"thread_id": "t1"}, "recursion_limit": 6}

    # Act
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage(content="oi")]}, config)
        graph_state = graph.get_state(config)
        assert graph_state is not None
        assert "messages" in graph_state.values
        assert len(graph_state.values["messages"]) > 0
        assert graph_state.next == ("model",)


def test_readonly_tool_no_interrupt(crm_tools: Sequence[BaseTool]) -> None:
    # Arrange
    human_message = HumanMessage(content="Encontre a empresa Acme")

    fake_messages: Sequence[AIMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    name="find_companies",
                    args={
                        "query": "Acme",
                    },
                    id="c1",
                )
            ],
        ),
        AIMessage(
            content="Não encontrei a empresa Acme",
        ),
    ]
    fake = FakeMessagesListChatModelBindedTools(responses=fake_messages)

    graph = build_graph(
        fake, crm_tools, checkpointer=InMemorySaver(), approval_required_tools=["create_company", "create_contact"]
    )

    config: RunnableConfig = {"configurable": {"thread_id": "t1"}, "recursion_limit": 20}

    # Act
    result = graph.invoke({"messages": [human_message]}, config)

    # Assert
    result_messages = result["messages"]
    expected_types_list = [HumanMessage, AIMessage, ToolMessage, AIMessage]
    actual_types_list = [type(msg) for msg in result_messages]
    assert actual_types_list == expected_types_list
    assert "__interrupt__" not in result


# Write tools trigger an interrupt with a structured payload, and only save after approval
def test_write_tool_triggers_interrupt(repo: InMemoryCRMRepository, crm_tools: Sequence[BaseTool]) -> None:
    # Arrange
    graph, config = _graph_waiting_for_approval(crm_tools, final_answer="Empresa cadastrada com sucesso")

    # Act & Assert
    result = graph.invoke({"messages": [HumanMessage(content="Crie a empresa Acme")]}, config)

    assert repo.find_companies(query="Acme") == []
    assert graph.get_state(config).next == ("approval",)
    assert result["__interrupt__"][0].value == {
        "tool_calls": [{"name": "create_company", "args": {"name": "Acme"}, "id": "c1"}]
    }

    result_after_approval = graph.invoke(Command(resume={"approved": True, "reason": None}), config)

    assert "__interrupt__" not in result_after_approval
    assert repo.find_companies(query="Acme") != []


@pytest.mark.parametrize(
    ("reason", "expected_in_message"),
    [("Nome errado", "Nome errado"), (None, "Nenhum motivo fornecido")],
)
def test_rejected_write_tool_create_company(
    repo: InMemoryCRMRepository, crm_tools: Sequence[BaseTool], reason: str | None, expected_in_message: str
) -> None:
    # Arrange
    graph, config = _graph_waiting_for_approval(crm_tools, final_answer="O que quer mudar?")
    graph.invoke({"messages": [HumanMessage(content="Crie a empresa Acme")]}, config)

    # Act
    result = graph.invoke(Command(resume={"approved": False, "reason": reason}), config)

    # Assert
    assert "__interrupt__" not in result
    assert repo.find_companies(query="Acme") == []

    rejection = result["messages"][-2]
    assert isinstance(rejection, ToolMessage)
    assert rejection.status == "error"
    assert rejection.tool_call_id == "c1"
    assert expected_in_message in str(rejection.content)


def test_invalid_resume_raises_validation_error(crm_tools: Sequence[BaseTool]) -> None:
    # Arrange
    graph, config = _graph_waiting_for_approval(crm_tools, final_answer="Não deveria chegar aqui")
    graph.invoke({"messages": [HumanMessage(content="Crie a empresa Acme")]}, config)

    # Act & Assert
    with pytest.raises(ValidationError):
        graph.invoke(Command(resume={"reason": "sem o campo approved"}), config)


def _graph_waiting_for_approval(
    crm_tools: Sequence[BaseTool], *, final_answer: str
) -> tuple[CompiledStateGraph[MessagesState], RunnableConfig]:
    """Graph whose fake model asks to create the company "Acme" and then gives `final_answer`."""
    fake_messages: Sequence[AIMessage] = [
        AIMessage(content="", tool_calls=[tool_call(name="create_company", args={"name": "Acme"}, id="c1")]),
        AIMessage(content=final_answer),
    ]
    fake = FakeMessagesListChatModelBindedTools(responses=fake_messages)
    graph = build_graph(
        fake, crm_tools, checkpointer=InMemorySaver(), approval_required_tools=["create_company", "create_contact"]
    )
    config: RunnableConfig = {"configurable": {"thread_id": "t1"}, "recursion_limit": 20}
    return graph, config
