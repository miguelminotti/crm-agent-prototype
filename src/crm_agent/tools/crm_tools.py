from uuid import UUID

from langchain_core.tools import BaseTool, tool

from crm_agent.crm.repository import CRMRepository
from crm_agent.tools.schemas import (
    CreateCompanyInput,
    CreateContactInput,
    FindCompaniesInput,
)


def make_crm_tools(repo: CRMRepository) -> list[BaseTool]:
    @tool(args_schema=CreateCompanyInput)
    def create_company(name: str, cnpj: str | None = None) -> str:
        """Create a new company with the given name and optional CNPJ. 
        Use this tool when the user asks to create a new company on the CRM or when trying to associate a contact with a company."""
        company = repo.create_company(name=name, cnpj=cnpj)
        return f"Company created with ID {company.id}."

    @tool(args_schema=CreateContactInput)
    def create_contact(
        name: str,
        phone: str | None = None,
        email: str | None = None,
        company_id: UUID | None = None,
    ) -> str:
        """Create a new contact with the given name, optional phone, email, and associated company ID. 
        Use this tool when the user asks to create a new contact on the CRM or when trying to associate a contact with a company."""
        contact = repo.create_contact(
            name=name, phone=phone, email=email, company_id=company_id
        )
        return f"Contact created with ID {contact.id}."

    @tool(args_schema=FindCompaniesInput)
    def find_companies(query: str) -> str:
        """Find companies matching the given query string. 
        Use this tool when the user asks to search for companies on the CRM or when trying to locate a specific company."""
        companies = repo.find_companies(query=query)
        if not companies:
            return "No companies found matching the query."
        return "\n".join(
            f"ID: {company.id}, Name: {company.name}, CNPJ: {company.cnpj or 'N/A'}"
            for company in companies
        )

    return [create_company, create_contact, find_companies]
