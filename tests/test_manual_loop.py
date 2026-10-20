import json

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.tools import BaseTool

from crm_agent.agent.manual_loop import run_agent
from crm_agent.crm.repository import InMemoryCRMRepository


def test_run_agent_with_fake_model(
    repo: InMemoryCRMRepository, crm_tools: list[BaseTool]
) -> None:
    # Turn 0. "Cadastra a Maria da Acme" - Human
    # Turn 1. tool_calls - find_companies - AIMessage
    # Turn 2. content JSON with 'Acme' company - ToolMessage
    # Turn 3. tool_calls - create_contact - AIMessage
    # Turn 4. content JSON with new contact - ToolMessage
    # Turn 5. content "Maria Cadastrada!" - AIMessage

    # Arrange
    acme_company = repo.create_company(name="Acme LTDA")
    fake_messages: list[AIMessage] = [
        AIMessage(  # Turn 1 - tool_calls - find_companies
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
        AIMessage(  # Turn 3 - tool_calls - create_contact
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
        AIMessage(
            content="Maria Cadastrada!"
        ),  # Turn 5 - content "Maria Cadastrada!" - AIMessage
    ]
    fake = FakeMessagesListChatModel(responses=fake_messages)
    tools = crm_tools
    # Use the crm_tools fixture instead of calling make_crm_tools directly

    # Turn 0.
    history: list[BaseMessage] = [HumanMessage(content="Cadastra a Maria da Acme")]
    history = run_agent(
        fake, tools, history
    )  # Turn 0 - HumanMessage "Cadastra a Maria da Acme"
    # On run_agent completion, the history should include all AI and Tool messages
    # up to Turn 5.
    # Check message type sequence
    expected_types_list = [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert [
        isinstance(msg, expected_type)
        for msg, expected_type in zip(history, expected_types_list, strict=False)
    ], "Message type sequence does not match expected sequence"
    history4_content = str(history[4].content)
    json_result = json.loads(history4_content)
    assert json_result["company_id"] == str(acme_company.id)
