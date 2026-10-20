"""Tests for the CRM tools defined in crm_agent/tools/crm_tools.py."""

import json
from uuid import UUID

import pytest
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from crm_agent.crm.repository import InMemoryCRMRepository


def test_create_company_with_formatted_cnpj(
    repo: InMemoryCRMRepository, crm_tools_dict: dict[str, BaseTool]
) -> None:
    create_company_tool = crm_tools_dict["create_company"]
    input_data = {"name": "Acme LTDA", "cnpj": "12.345.678/0001-95"}
    result = create_company_tool.invoke(input_data)
    result_json = json.loads(result)
    assert "company_id" in result_json
    company_id = result_json["company_id"]
    company = repo.get_company(company_id=UUID(company_id))
    assert result_json["cnpj"] == "12345678000195"
    assert company is not None
    assert company.cnpj == "12345678000195"


def test_create_contact_with_nonexistent_company_id(
    crm_tools_dict: dict[str, BaseTool],
) -> None:
    create_contact_tool = crm_tools_dict["create_contact"]
    input_data = {
        "name": "John Doe",
        "phone": "1234567890",
        "email": "john.doe@example.com",
        "company_id": "00000000-0000-0000-0000-000000000000",
    }
    result = create_contact_tool.invoke(input_data)
    result_json = json.loads(result)
    assert "error" in result_json


def test_create_and_find_company_and_create_contact(
    crm_tools_dict: dict[str, BaseTool],
) -> None:
    create_company_tool = crm_tools_dict["create_company"]
    find_companies_tool = crm_tools_dict["find_companies"]
    create_contact_tool = crm_tools_dict["create_contact"]

    # Create a company
    input_data_company = {"name": "Acme LTDA", "cnpj": "12.345.678/0001-95"}
    result_company = create_company_tool.invoke(input_data_company)
    result_company_json = json.loads(result_company)
    assert "company_id" in result_company_json
    company_id = result_company_json["company_id"]

    # Find the company
    input_data_find = {"query": "Acme LTDA"}
    result_find = find_companies_tool.invoke(input_data_find)
    result_find_json = json.loads(result_find)
    assert any(
        company["company_id"] == company_id for company in result_find_json["companies"]
    )

    # Create a contact with the found company ID
    input_data_contact = {
        "name": "John Doe",
        "phone": "1234567890",
        "email": "john.doe@example.com",
        "company_id": company_id,
    }
    result_contact = create_contact_tool.invoke(input_data_contact)
    result_contact_json = json.loads(result_contact)
    assert "contact_id" in result_contact_json


def test_find_companies_with_no_results(crm_tools_dict: dict[str, BaseTool]) -> None:
    find_companies_tool = crm_tools_dict["find_companies"]
    input_data = {"query": "Nonexistent Company"}
    result = find_companies_tool.invoke(input_data)
    result_json = json.loads(result)
    assert "companies" in result_json
    assert len(result_json["companies"]) == 0


def test_duplicate_company_creation(crm_tools_dict: dict[str, BaseTool]) -> None:
    create_company_tool = crm_tools_dict["create_company"]
    input_data = {"name": "Acme LTDA", "cnpj": "12.345.678/0001-95"}
    result = create_company_tool.invoke(input_data)
    result_json = json.loads(result)
    assert "company_id" in result_json

    # Attempt to create the same company again
    result_duplicate = create_company_tool.invoke(input_data)
    result_duplicate_json = json.loads(result_duplicate)
    assert "error" in result_duplicate_json


def test_create_contact_with_invalid_name(crm_tools_dict: dict[str, BaseTool]) -> None:
    create_contact_tool = crm_tools_dict["create_contact"]
    input_data = {
        "name": "",
        "phone": "1234567890",
        "email": "john.doe@example.com",
        "company_id": None,
    }
    with pytest.raises(ValidationError):
        create_contact_tool.invoke(input_data)


def test_create_company_with_invalid_name(crm_tools_dict: dict[str, BaseTool]) -> None:
    create_company_tool = crm_tools_dict["create_company"]
    input_data = {
        "name": "",
        "cnpj": "12.345.678/0001-95",
    }
    with pytest.raises(ValidationError):
        create_company_tool.invoke(input_data)
