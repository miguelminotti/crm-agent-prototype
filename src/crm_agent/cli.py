from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from crm_agent.agent.graph import build_graph
from crm_agent.crm.repository import InMemoryCRMRepository
from crm_agent.tools.crm_tools import make_crm_tools


def main() -> None:
    repo = InMemoryCRMRepository()
    tools = make_crm_tools(repo)
    thread_id = "thread-2"
    anthropic_model = build_chat_anthropic_model()
    graph = build_graph(
        anthropic_model,
        tools,
        checkpointer=InMemorySaver(),
        approval_required_tools=["create_contact", "create_company"],
    )
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}

    while True:
        user_input = input("Você: ")
        if user_input.lower().strip() in ["sair", "exit", "quit"]:
            break
        human_message = HumanMessage(content=user_input)
        result = graph.invoke({"messages": [human_message]}, config)
        while "__interrupt__" in result:
            resume_input = ask_approval(result["__interrupt__"][0].value["question"])
            result = graph.invoke(Command(resume=resume_input), config)
        print(result["messages"][-1].text)


def build_chat_anthropic_model() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-haiku-5-5",
        max_tokens=16000,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
    )


def ask_approval(question: str) -> str:
    print(question)
    print("s/n?")
    user_input = input("Você: ")
    return "sim" if user_input.lower().strip() in ["s", "sim"] else "não"
