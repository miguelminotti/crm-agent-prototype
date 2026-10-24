from typing import Any

from evals.dataset import CRM_AGENT_EVAL_DATASET
from evals.evaluators import companies_match, contacts_match, no_forbidden_tools, tools_match

# Scenario 2: create a contact linked to the seeded company "c-1"
REFERENCE_OUTPUTS: dict[str, Any] = CRM_AGENT_EVAL_DATASET[1]["reference_outputs"]

# Output of `python -m evals.target 2`
OUTPUTS: dict[str, Any] = {
    "answer": "Contato João da Silva criado e vinculado à empresa Acme!",
    "tools_called": ["find_companies", "create_contact"],
    "interrupts": 1,
    "companies_created": [],
    "contacts_created": [{"name": "João da Silva", "email": "joao@acme.com", "company_id": "c-1"}],
}


def test_companies_match_passes() -> None:
    assert companies_match(OUTPUTS, REFERENCE_OUTPUTS)["score"] is True


def test_companies_match_fails_when_unexpected_company_is_created() -> None:
    outputs = {**OUTPUTS, "companies_created": [{"name": "Acme", "cnpj": None}]}

    result = companies_match(outputs, REFERENCE_OUTPUTS)

    assert result["score"] is False
    assert "Acme" in result["comment"]


def test_contacts_match_passes() -> None:
    assert contacts_match(OUTPUTS, REFERENCE_OUTPUTS)["score"] is True


def test_contacts_match_fails_when_linked_to_wrong_company() -> None:
    outputs = {
        **OUTPUTS,
        "contacts_created": [{"name": "João da Silva", "email": "joao@acme.com", "company_id": None}],
    }

    result = contacts_match(outputs, REFERENCE_OUTPUTS)

    assert result["score"] is False
    assert "c-1" in result["comment"]


def test_contacts_match_fails_when_no_contact_is_created() -> None:
    outputs = {**OUTPUTS, "contacts_created": []}

    assert contacts_match(outputs, REFERENCE_OUTPUTS)["score"] is False


def test_tools_match_passes() -> None:
    assert tools_match(OUTPUTS, REFERENCE_OUTPUTS)["score"] is True


def test_tools_match_fails_when_find_is_skipped() -> None:
    outputs = {**OUTPUTS, "tools_called": ["create_contact"]}

    result = tools_match(outputs, REFERENCE_OUTPUTS)

    assert result["score"] is False
    assert "find_companies" in result["comment"]


def test_tools_match_fails_when_a_tool_is_repeated() -> None:
    outputs = {**OUTPUTS, "tools_called": ["find_companies", "find_companies", "create_contact"]}

    assert tools_match(outputs, REFERENCE_OUTPUTS)["score"] is False


def test_no_forbidden_tools_passes() -> None:
    assert no_forbidden_tools(OUTPUTS, REFERENCE_OUTPUTS)["score"] is True


def test_no_forbidden_tools_fails_when_forbidden_tool_is_called() -> None:
    outputs = {**OUTPUTS, "tools_called": ["find_companies", "create_company", "create_contact"]}

    result = no_forbidden_tools(outputs, REFERENCE_OUTPUTS)

    assert result["score"] is False
    assert "create_company" in result["comment"]
