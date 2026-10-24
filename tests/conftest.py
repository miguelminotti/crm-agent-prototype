from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from crm_agent.crm.repository import InMemoryCRMRepository
from crm_agent.tools.crm_tools import make_crm_tools


class FakeMessagesListChatModelBindedTools(FakeMessagesListChatModel):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[LanguageModelInput, AIMessage]:
        return self


@pytest.fixture
def repo() -> InMemoryCRMRepository:
    return InMemoryCRMRepository()


@pytest.fixture
def crm_tools(repo: InMemoryCRMRepository) -> list[BaseTool]:
    return make_crm_tools(repo)


@pytest.fixture
def crm_tools_dict(crm_tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in crm_tools}
