import json
from collections.abc import Sequence

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.tools import BaseTool

from crm_agent.agent.manual_loop import run_agent
from crm_agent.crm.repository import InMemoryCRMRepository


def test_run_agent_with_fake_model(repo: InMemoryCRMRepository, crm_tools: Sequence[BaseTool]) -> None:
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
    fake = FakeMessagesListChatModel(responses=fake_messages)

    history: Sequence[BaseMessage] = [HumanMessage(content="Cadastra a Maria da Acme")]
    history = run_agent(fake, crm_tools, history)
    expected_types_list = [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    history_types_list = [type(msg) for msg in history]
    assert history_types_list == expected_types_list, "Message type sequence does not match expected sequence"
    if isinstance(history[4].content, str):
        json_result = json.loads(str(history[4].content))
        assert json_result["company_id"] == str(acme_company.id)
    else:
        raise TypeError("Expected history[4].content to be a string containing JSON.")


def test_run_agent_with_missing_tool(crm_tools: Sequence[BaseTool]) -> None:
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
        AIMessage(content="Attempted to use a missing tool."),
    ]
    fake = FakeMessagesListChatModel(responses=fake_messages)

    history: Sequence[BaseMessage] = [HumanMessage(content="Try to use a missing tool")]
    history = run_agent(fake, crm_tools, history)

    # Assert
    expected_types_list = [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    history_types_list = [type(msg) for msg in history]
    assert history_types_list == expected_types_list, "Message type sequence does not match expected sequence"
    history_tool_message = history[2]
    assert isinstance(history_tool_message, ToolMessage) and history_tool_message.status == "error", (
        "Expected a ToolMessage with error status."
    )
    assert history_tool_message.tool_call_id == "c1", "Expected tool call ID to be 'c1'."


def test_run_agent_with_validation_error_tool(crm_tools: Sequence[BaseTool]) -> None:
    # Arrange
    fake_messages: Sequence[AIMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    name="create_contact",
                    args={
                        "name": "",  # Invalid value to trigger validation error
                    },
                    id="c1",
                )
            ],
        ),
        AIMessage(content="Attempted to create a contact with invalid data."),
    ]
    fake = FakeMessagesListChatModel(responses=fake_messages)

    history: Sequence[BaseMessage] = [HumanMessage(content="Try to create a contact with invalid data")]
    history = run_agent(fake, crm_tools, history)

    # Assert
    expected_types_list = [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    history_types_list = [type(msg) for msg in history]
    assert history_types_list == expected_types_list, "Message type sequence does not match expected sequence"
    history_tool_message = history[2]
    assert isinstance(history_tool_message, ToolMessage) and history_tool_message.status == "error", (
        "Expected a ToolMessage with error status."
    )
    assert history_tool_message.tool_call_id == "c1", "Expected tool call ID to be 'c1'."
