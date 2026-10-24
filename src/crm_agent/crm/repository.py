"""CRM repository protocol definition."""

from typing import Protocol
from uuid import UUID

from crm_agent.crm.errors import CompanyNotFoundError, DuplicateCompanyError
from crm_agent.crm.models import Company, Contact


class CRMRepository(Protocol):
    """Protocol for CRM repository implementations."""

    def create_company(self, *, name: str, cnpj: str | None = None) -> Company: ...

    def create_contact(
        self,
        *,
        name: str,
        email: str | None = None,
        phone: str | None = None,
        company_id: UUID | None = None,
    ) -> Contact: ...

    def get_company(self, *, company_id: UUID) -> Company | None: ...

    def find_companies(self, *, query: str) -> list[Company]: ...

    def list_companies(self) -> list[Company]: ...

    def list_contacts(self) -> list[Contact]: ...


class InMemoryCRMRepository:
    def __init__(self) -> None:
        self._companies: dict[UUID, Company] = {}
        self._contacts: dict[UUID, Contact] = {}

    def create_company(self, *, name: str, cnpj: str | None = None) -> Company:
        if cnpj is not None and any(c.cnpj == cnpj for c in self._companies.values()):
            raise DuplicateCompanyError(f"Company with CNPJ {cnpj} already exists.")
        company = Company(name=name, cnpj=cnpj)
        self._companies[company.id] = company
        return company

    def create_contact(
        self,
        *,
        name: str,
        email: str | None = None,
        phone: str | None = None,
        company_id: UUID | None = None,
    ) -> Contact:
        if company_id is not None and company_id not in self._companies:
            raise CompanyNotFoundError(f"Company with ID {company_id} not found.")
        contact = Contact(name=name, email=email, phone=phone, company_id=company_id)
        self._contacts[contact.id] = contact
        return contact

    def get_company(self, *, company_id: UUID) -> Company | None:
        return self._companies.get(company_id)

    def find_companies(self, *, query: str) -> list[Company]:
        return [
            c
            for c in self._companies.values()
            if query.casefold() in c.name.casefold() or (c.cnpj is not None and query in c.cnpj)
        ]

    def list_companies(self) -> list[Company]:
        return list(self._companies.values())

    def list_contacts(self) -> list[Contact]:
        return list(self._contacts.values())
