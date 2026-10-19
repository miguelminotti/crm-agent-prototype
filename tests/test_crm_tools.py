# Tests for the CRM tools defined in crm_agent/tools/crm_tools.py
# Cover:
# Create a company with a formatted CNPJ ends with a normalized CNPJ on the repo.
# Create a contact with nonexistent company_id returns an appropriate error message without raising.
# Create a company "Acme LTDA", search with find_companies, extract the id of the returned text and create the contact with it.
# What happens with an invalid input name for both company and contact creation.

from langchain_core.tools import BaseTool
import pytest
from uuid import UUID

from crm_agent.crm.repository import InMemoryCRMRepository
from crm_agent.tools.crm_tools import make_crm_tools
from crm_agent.tools.schemas import CreateCompanyInput, CreateContactInput, FindCompaniesInput

@pytest.fixture
def repo() -> InMemoryCRMRepository:
    return InMemoryCRMRepository()


@pytest.fixture
def crm_tools(repo: InMemoryCRMRepository):
    return make_crm_tools(repo)


def test_create_company_with_formatted_cnpj(repo: InMemoryCRMRepository, crm_tools: list[BaseTool]) -> None:
    create_company_tool = next(tool for tool in crm_tools if tool.name == "create_company")
    input_data = {
        "name": "Acme LTDA",
        "cnpj": "12.345.678/0001-95"
    }
    result = create_company_tool.invoke(input_data)
    assert "Company created with ID" in result
    company_id = result.split()[-1].strip(".")
    company = repo.get_company(company_id=UUID(company_id))
    assert company.cnpj == "12345678000195"


def test_create_contact_with_nonexistent_company_id(repo: InMemoryCRMRepository, crm_tools: list[BaseTool]) -> None:
    create_contact_tool = next(tool for tool in crm_tools if tool.name == "create_contact")
    input_data = {
        "name": "John Doe",
        "phone": "1234567890",
        "email": "john.doe@example.com",
        "company_id": "00000000-0000-0000-0000-000000000000"
    }
    result = create_contact_tool.invoke(input_data)
    assert "error" in result.lower()


def test_create_and_find_company_and_create_contact(repo: InMemoryCRMRepository, crm_tools: list[BaseTool]) -> None:
    create_company_tool = next(tool for tool in crm_tools if tool.name == "create_company")
    find_companies_tool = next(tool for tool in crm_tools if tool.name == "find_companies")
    create_contact_tool = next(tool for tool in crm_tools if tool.name == "create_contact")

    # Create a company
    input_data_company = {
        "name": "Acme LTDA",
        "cnpj": "12.345.678/0001-95"
    }
    result_company = create_company_tool.invoke(input_data_company)
    assert "Company created with ID" in result_company
    company_id = result_company.split()[-1].strip(".")

    # Find the company
    input_data_find = {
        "query": "Acme LTDA"
    }
    result_find = find_companies_tool.invoke(input_data_find)
    assert company_id in result_find

    # Create a contact with the found company ID
    input_data_contact = {
        "name": "John Doe",
        "phone": "1234567890",
        "email": "john.doe@example.com",
        "company_id": company_id
    }
    result_contact = create_contact_tool.invoke(input_data_contact)
    assert "Contact created with ID" in result_contact


def test_create_contact_with_invalid_name(repo: InMemoryCRMRepository, crm_tools: list[BaseTool]) -> None:
    create_contact_tool = next(tool for tool in crm_tools if tool.name == "create_contact")
    input_data = {
        "name": "",
        "phone": "1234567890",
        "email": "john.doe@example.com",
        "company_id": None
    }
    result = create_contact_tool.invoke(input_data)
    assert "error" in result.lower()