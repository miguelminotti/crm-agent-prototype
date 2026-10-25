from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from crm_agent.agent.chat_models import build_chat_anthropic_model
from crm_agent.agent.graph import ApprovalDecision, InterruptPayload, build_graph
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
            resume_input = ask_approval(result["__interrupt__"][0].value)
            result = graph.invoke(Command(resume=resume_input), config)
        print(result["messages"][-1].text)


def ask_approval(interrupt_payload: InterruptPayload) -> ApprovalDecision:
    print("Pergunta do sistema: Você aprova essas operações no CRM?")
    for i, tool_call in enumerate(interrupt_payload["tool_calls"], start=1):
        print(f"Chamada de ferramenta {i}:")
        print("  Nome:", tool_call["name"])
        print("  Argumentos:", tool_call["args"])
    print("s/n?")
    user_input = input("Você: ")
    if user_input.lower().strip() in ["s", "sim"]:
        decision = ApprovalDecision(approved=True, reason=None)
    else:
        reason = input("Motivo da rejeição: ").strip() or None
        decision = ApprovalDecision(approved=False, reason=reason)

    return decision
