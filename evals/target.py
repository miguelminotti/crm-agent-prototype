import json
import sys
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from crm_agent.agent.chat_models import build_chat_anthropic_model
from crm_agent.agent.graph import ApprovalDecision, build_graph
from crm_agent.crm.repository import InMemoryCRMRepository
from crm_agent.tools.crm_tools import make_crm_tools
from evals.dataset import CRM_AGENT_EVAL_DATASET


def run_agent(inputs: dict[str, Any]) -> dict[str, Any]:
    repo = InMemoryCRMRepository()

    # Seed: the dataset uses ids like "c-1", the repo generates UUIDs, so keep a map to translate back
    seed_ids = {}
    for company in inputs.get("seed", {}).get("companies", []):
        created = repo.create_company(name=company["name"], cnpj=company.get("cnpj"))
        seed_ids[created.id] = company["id"]

    # "Photo" before the invoke, to tell apart what the agent created from the seed
    companies_before = repo.list_companies()
    contacts_before = repo.list_contacts()

    graph = build_graph(
        build_chat_anthropic_model(),
        make_crm_tools(repo),
        checkpointer=InMemorySaver(),
        approval_required_tools=["create_company", "create_contact"],
    )
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 20}

    result = graph.invoke({"messages": [HumanMessage(content=inputs["message"])]}, config)
    interrupts = 0
    while "__interrupt__" in result:
        interrupts += 1
        approval = ApprovalDecision(approved=True, reason=None)
        result = graph.invoke(Command(resume=approval), config)

    tools_called = []
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                tools_called.append(call["name"])

    companies_created = []
    for company in repo.list_companies():
        if company not in companies_before:
            companies_created.append({"name": company.name, "cnpj": company.cnpj})

    contacts_created = []
    for contact in repo.list_contacts():
        if contact not in contacts_before:
            company_id = seed_ids.get(contact.company_id, str(contact.company_id)) if contact.company_id else None
            contacts_created.append({"name": contact.name, "email": contact.email, "company_id": company_id})

    return {
        "answer": result["messages"][-1].text,
        "tools_called": tools_called,
        "interrupts": interrupts,
        "companies_created": companies_created,
        "contacts_created": contacts_created,
    }


if __name__ == "__main__":
    scenario = CRM_AGENT_EVAL_DATASET[int(sys.argv[1]) - 1]
    print(json.dumps(run_agent(scenario["inputs"]), indent=2, ensure_ascii=False))
