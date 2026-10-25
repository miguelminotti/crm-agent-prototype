import asyncio
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from conftest import FakeMessagesListChatModelBindedTools
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from crm_agent.agent.graph import build_graph
from crm_agent.api import _thread_locks, app, get_graph
from crm_agent.crm.repository import InMemoryCRMRepository

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


CREATE_ACME = AIMessage(
    content="",
    tool_calls=[tool_call(name="create_company", args={"name": "Acme"}, id="c1")],
)


def test_write_message_returns_pending_approval(make_client: MakeClient) -> None:
    client = make_client([CREATE_ACME])

    response = client.post("/threads/t1/messages", json={"message": "Crie a empresa Acme"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": None,
        "pending_approval": {"tool_calls": [{"name": "create_company", "args": {"name": "Acme"}, "id": "c1"}]},
    }


def test_approval_runs_the_write(make_client: MakeClient, repo: InMemoryCRMRepository) -> None:
    client = make_client([CREATE_ACME, AIMessage(content="Empresa Acme cadastrada!")])
    client.post("/threads/t1/messages", json={"message": "Crie a empresa Acme"})

    response = client.post("/threads/t1/approval", json={"approved": True})

    assert response.status_code == 200
    assert response.json() == {"answer": "Empresa Acme cadastrada!", "pending_approval": None}
    assert repo.find_companies(query="Acme") != []


def test_rejection_with_reason_skips_the_write(make_client: MakeClient, repo: InMemoryCRMRepository) -> None:
    client = make_client([CREATE_ACME, AIMessage(content="Ok, não criei. Qual o nome certo?")])
    client.post("/threads/t1/messages", json={"message": "Crie a empresa Acme"})

    response = client.post("/threads/t1/approval", json={"approved": False, "reason": "Nome errado"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Ok, não criei. Qual o nome certo?", "pending_approval": None}
    assert repo.find_companies(query="Acme") == []


def test_body_without_message_returns_422(make_client: MakeClient) -> None:
    client = make_client([])

    response = client.post("/threads/t1/messages", json={})

    assert response.status_code == 422


def test_approval_without_pending_approval_returns_409(make_client: MakeClient) -> None:
    client = make_client([AIMessage(content="Hello")])
    # Create a thread without any pending approval
    client.post("/threads/t1/messages", json={"message": "Hello"})

    response = client.post("/threads/t1/approval", json={"approved": True})

    assert response.status_code == 409
    assert response.json() == {"detail": "No pending approval for this thread"}


def test_approval_for_nonexistent_thread_returns_404(make_client: MakeClient) -> None:
    client = make_client([])

    response = client.post("/threads/nonexistent/approval", json={"approved": True})

    assert response.status_code == 404
    assert response.json() == {"detail": "Thread not found"}


class GatedFakeModel(FakeMessagesListChatModelBindedTools):
    """Fake model that waits at `gate` before answering, so the test controls when a request finishes."""

    # A new Event per test: an asyncio.Event binds to the first event loop that waits on it, and each test has its own
    gate: asyncio.Event

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        await asyncio.wait_for(self.gate.wait(), timeout=5)  # timeout: a failing test doesn't hang forever
        return await super()._agenerate(*args, **kwargs)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("path", "body"),
    [("/threads/t1/messages", {"message": "Crie a empresa Acme"}), ("/threads/t1/approval", {"approved": True})],
)
async def test_request_while_thread_is_running_returns_409(
    crm_tools: list[BaseTool], path: str, body: dict[str, Any]
) -> None:
    # Arrange
    gate = asyncio.Event()
    _thread_locks.clear()
    graph = build_graph(
        GatedFakeModel(responses=[CREATE_ACME], gate=gate),
        crm_tools,
        checkpointer=InMemorySaver(),
        approval_required_tools=["create_company", "create_contact"],
    )
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Act: the first request starts and stays inside the lock, stuck at the gate
            first = asyncio.create_task(client.post("/threads/t1/messages", json={"message": "Crie a empresa Acme"}))
            while "t1" not in _thread_locks or not _thread_locks["t1"].locked():
                await asyncio.sleep(0)

            second = await client.post(path, json=body)

            gate.set()
            first_response = await first
    finally:
        app.dependency_overrides.clear()

    # Assert
    assert second.status_code == 409
    assert second.json() == {"detail": "Thread is currently locked"}
    assert first_response.status_code == 200


def test_create_message_while_pending_approval_returns_409(make_client: MakeClient) -> None:
    client = make_client([CREATE_ACME, AIMessage(content="Ok, não criei. Qual o nome certo?")])

    # Send a message that triggers a pending approval
    client.post("/threads/t1/messages", json={"message": "Crie a empresa Acme"})

    # Try to send another message while approval is pending
    response = client.post("/threads/t1/messages", json={"message": "Crie a empresa Acme"})

    assert response.status_code == 409
    assert response.json() == {"detail": "Pending approval for this thread"}
