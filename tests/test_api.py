from collections.abc import Callable, Iterator

import pytest
from conftest import FakeMessagesListChatModelBindedTools
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import tool_call
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from crm_agent.agent.graph import build_graph
from crm_agent.api import app, get_graph

MakeClient = Callable[[list[AIMessage]], TestClient]


@pytest.fixture
def make_client(crm_tools: list[BaseTool]) -> Iterator[MakeClient]:
    """Returns a function that builds a TestClient whose graph uses a fake model with the given responses."""

    def _make_client(responses: list[AIMessage]) -> TestClient:
        graph = build_graph(
            FakeMessagesListChatModelBindedTools(responses=responses),
            crm_tools,
            checkpointer=InMemorySaver(),
            approval_required_tools=["create_company", "create_contact"],
        )
        app.dependency_overrides[get_graph] = lambda: graph
        return TestClient(app)

    yield _make_client
    app.dependency_overrides.clear()


def test_simple_message_has_answer_and_no_pending_approval(make_client: MakeClient) -> None:
    client = make_client([AIMessage(content="Olá, tudo bem?")])

    response = client.post("/threads/t1/messages", json={"message": "oi"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Olá, tudo bem?", "pending_approval": None}


def test_write_message_returns_pending_approval(make_client: MakeClient) -> None:
    create_acme = AIMessage(
        content="",
        tool_calls=[tool_call(name="create_company", args={"name": "Acme"}, id="c1")],
    )
    client = make_client([create_acme])

    response = client.post("/threads/t1/messages", json={"message": "Crie a empresa Acme"})

    assert response.status_code == 200
    pending_approval = response.json()["pending_approval"]
    assert pending_approval is not None
    assert "create_company" in pending_approval


def test_body_without_message_returns_422(make_client: MakeClient) -> None:
    client = make_client([])

    response = client.post("/threads/t1/messages", json={})

    assert response.status_code == 422
