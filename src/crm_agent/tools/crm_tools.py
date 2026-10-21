import json
from uuid import UUID

from langchain_core.tools import BaseTool, tool

from crm_agent.crm.errors import CRMError
from crm_agent.crm.models import Company
from crm_agent.crm.repository import CRMRepository
from crm_agent.tools.schemas import (
    CreateCompanyInput,
    CreateContactInput,
    FindCompaniesInput,
)


def make_crm_tools(repo: CRMRepository) -> list[BaseTool]:
    @tool(args_schema=CreateCompanyInput)
    def create_company(name: str, cnpj: str | None = None) -> str:
        """
        Create a new company with the given name and optional CNPJ.

        Use this tool when the user asks to create a new company on the CRM.
        """
        try:
            company = repo.create_company(name=name, cnpj=cnpj)
        except CRMError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        else:
            return json.dumps(_get_company_serializable_dict(company), ensure_ascii=False)

    @tool(args_schema=CreateContactInput)
    def create_contact(
        name: str,
        phone: str | None = None,
        email: str | None = None,
        company_id: UUID | None = None,
    ) -> str:
        """
        Create a new contact with the given name.

        Optional phone, email, and associated company ID;
        use this tool when the user asks to create a new contact on the CRM;
        use `find_companies` before when the user mentions a company.
        """
        try:
            contact = repo.create_contact(name=name, phone=phone, email=email, company_id=company_id)
        except CRMError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        else:
            result = {
                "contact_id": str(contact.id),
                "name": contact.name,
                "phone": contact.phone,
                "email": contact.email,
                "company_id": str(contact.company_id) if contact.company_id is not None else None,
                "created_at": contact.created_at.isoformat(),
            }
            return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=FindCompaniesInput)
    def find_companies(query: str) -> str:
        """
        Find companies matching the given query string.

        Use this tool when the user asks to search for companies on the CRM;
        call `find_companies` first when the user mentions a company.
        """
        companies = repo.find_companies(query=query)
        result = {"companies": [_get_company_serializable_dict(company) for company in companies]}

        return json.dumps(result, ensure_ascii=False)

    return [create_company, create_contact, find_companies]


def _get_company_serializable_dict(company: Company) -> dict[str, object]:
    return {
        "company_id": str(company.id),
        "name": company.name,
        "cnpj": company.cnpj,
        "created_at": company.created_at.isoformat(),
    }
