import uuid

import pytest

from crm_agent.crm.errors import CompanyNotFoundError, DuplicateCompanyError
from crm_agent.crm.repository import CRMRepository, InMemoryCRMRepository


# Static checks to ensure the repository implementation satisfies the protocol
def test_in_memory_crm_repository_satisfies_protocol() -> None:
    repo: CRMRepository = InMemoryCRMRepository()
    assert repo is not None


def test_create_contact_with_invalid_company_id_raises_error(
    in_memory_crm_repository: InMemoryCRMRepository,
) -> None:
    with pytest.raises(CompanyNotFoundError):
        in_memory_crm_repository.create_contact(
            name="John Doe",
            email="john.doe@example.com",
            phone="1234567890",
            company_id=uuid.uuid4(),
        )


def test_create_company_with_duplicate_cnpj_raises_error(
    in_memory_crm_repository: InMemoryCRMRepository,
) -> None:
    in_memory_crm_repository.create_company(name="Company A", cnpj="123456789")
    with pytest.raises(DuplicateCompanyError):
        in_memory_crm_repository.create_company(name="Company B", cnpj="123456789")


def test_find_companies_by_name_and_cnpj(
    in_memory_crm_repository: InMemoryCRMRepository,
) -> None:
    company_a = in_memory_crm_repository.create_company(
        name="Company A", cnpj="12345678000192"
    )
    company_b = in_memory_crm_repository.create_company(
        name="Company B", cnpj="98765432000193"
    )

    # Find by name
    found_by_name = in_memory_crm_repository.find_companies(query="Company A")
    assert company_a in found_by_name
    assert company_b not in found_by_name

    # Find by cnpj
    found_by_cnpj = in_memory_crm_repository.find_companies(query="98765432000193")
    assert company_b in found_by_cnpj
    assert company_a not in found_by_cnpj


def test_find_companies_with_no_matches(
    in_memory_crm_repository: InMemoryCRMRepository,
) -> None:
    in_memory_crm_repository.create_company(name="Company A", cnpj="12345678000192")
    in_memory_crm_repository.create_company(name="Company B", cnpj="98765432000193")

    found_companies = in_memory_crm_repository.find_companies(
        query="Nonexistent Company"
    )
    assert found_companies == []


def test_find_companies_case_insensitive(
    in_memory_crm_repository: InMemoryCRMRepository,
) -> None:
    company_a = in_memory_crm_repository.create_company(
        name="Acme LTDA", cnpj="12345678000192"
    )
    company_b = in_memory_crm_repository.create_company(
        name="Diaz ME", cnpj="98765432000193"
    )

    # Find by name with different case
    found_by_name = in_memory_crm_repository.find_companies(query="acme")
    assert company_a in found_by_name
    assert company_b not in found_by_name


def test_create_company_with_cnpj_none_then_find_companies(
    in_memory_crm_repository: InMemoryCRMRepository,
) -> None:
    company = in_memory_crm_repository.create_company(name="Company C", cnpj=None)
    found_companies = in_memory_crm_repository.find_companies(query="Company C")
    assert company in found_companies


# Happy path tests for the InMemoryCRMRepository


def test_create_and_get_company(
    in_memory_crm_repository: InMemoryCRMRepository,
) -> None:
    company = in_memory_crm_repository.create_company(
        name="Company D", cnpj="12345678000194"
    )
    retrieved_company = in_memory_crm_repository.get_company(company_id=company.id)
    assert retrieved_company == company


def test_get_company_not_found(
    in_memory_crm_repository: InMemoryCRMRepository,
) -> None:
    retrieved_company = in_memory_crm_repository.get_company(company_id=uuid.uuid4())
    assert retrieved_company is None
