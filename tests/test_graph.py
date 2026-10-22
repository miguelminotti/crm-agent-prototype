import json
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.messages.utils import AnyMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from crm_agent.agent.graph import build_graph
from crm_agent.crm.repository import InMemoryCRMRepository


class FakeMessagesListChatModelBindedTools(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


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
