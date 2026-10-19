import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.tools import BaseTool

from crm_agent.agent.manual_loop import run_agent
from crm_agent.crm.repository import InMemoryCRMRepository
from crm_agent.tools.crm_tools import make_crm_tools


@pytest.fixture
def repo() -> InMemoryCRMRepository:
    return InMemoryCRMRepository()


@pytest.fixture
def crm_tools(repo: InMemoryCRMRepository) -> list[BaseTool]:
    return [t for t in make_crm_tools(repo)]


@pytest.fixture
def crm_tools_dict(crm_tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in crm_tools}


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
    history = [HumanMessage(content="Cadastra a Maria da Acme")]
    history = run_agent(
        fake, tools, history
    )  # Turn 0 - HumanMessage "Cadastra a Maria da Acme"
    # On run_agent completion, the history should include all AI and Tool messages
    # up to Turn 5.
    assert any(
        isinstance(msg, AIMessage) and msg.content == "Maria Cadastrada!"
        for msg in history
    )
    # Check message type sequence
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)
    assert isinstance(history[2], ToolMessage)
    assert isinstance(history[3], AIMessage)
    assert isinstance(history[4], ToolMessage)
    assert isinstance(history[5], AIMessage)
    assert f'"company_id": "{acme_company.id}"' in history[4].content
