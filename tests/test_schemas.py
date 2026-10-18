from uuid import UUID

import pytest
from pydantic import ValidationError

from crm_agent.tools.schemas import CreateCompanyInput, CreateContactInput

# Valid cases tests


def test_create_company_input() -> None:
    # Arrange
    input_data = {"name": "Test Company", "cnpj": "12345678000195"}
    # Act
    company_input = CreateCompanyInput.model_validate(input_data)
    # Assert
    assert company_input.name == input_data["name"]
    assert company_input.cnpj == input_data["cnpj"]


def test_create_company_input_with_omitted_fields_returns_none() -> None:
    # Arrange
    input_data = {
        "name": "Test Company",
    }
    # Act
    company_input = CreateCompanyInput.model_validate(input_data)
    # Assert
    assert company_input.cnpj is None


def test_create_company_with_formatted_cnpj_returns_digits_only() -> None:
    # Arrange
    input_data = {"name": "Test Company", "cnpj": "12.345.678/0001-95"}
    # Act
    company_input = CreateCompanyInput.model_validate(input_data)
    # Assert
    assert company_input.cnpj == "12345678000195"


def test_create_contact_input() -> None:
    # Arrange
    input_data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "1234567890",
    }
    # Act
    contact_input = CreateContactInput.model_validate(input_data)
    # Assert
    assert contact_input.name == input_data["name"]
    assert contact_input.email == input_data["email"]
    assert contact_input.phone == input_data["phone"]


def test_create_contact_input_formatted_phone_returns_digits_only() -> None:
    # Arrange
    input_data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "+55 (12) 99456-7890",
    }
    # Act
    contact_input = CreateContactInput.model_validate(input_data)
    # Assert
    assert contact_input.phone == "5512994567890"


def test_create_contact_with_omitted_fields_returns_none() -> None:
    # Arrange
    input_data = {
        "name": "John Doe",
    }
    # Act
    contact_input = CreateContactInput.model_validate(input_data)
    # Assert
    assert contact_input.phone is None
    assert contact_input.email is None
    assert contact_input.company_id is None


def test_create_contact_input_with_str_company_id_becomes_uuid() -> None:
    # Arrange
    input_data = {
        "name": "John Doe",
        "company_id": "12345678-0001-95ab-cdef-1234567890ab",
    }
    # Act
    contact_input = CreateContactInput.model_validate(input_data)
    # Assert
    assert isinstance(contact_input.company_id, UUID)
    assert contact_input.company_id == UUID(input_data["company_id"])


# Invalid cases tests


# Parametrize for create contact, empty name, name with spaces
@pytest.mark.parametrize(
    "name",
    [
        "",  # empty name
        "   ",  # name with spaces
    ],
)
def test_create_contact_input_invalid_name_raises_error(name: str) -> None:
    # Arrange
    input_data = {"name": name, "email": "john.doe@example.com", "phone": "1234567890"}
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        CreateContactInput.model_validate(input_data)
    assert exc_info.value.errors()[0]["loc"] == ("name",)


# Parametrize empty name, name with spaces
@pytest.mark.parametrize(
    "name",
    [
        "",  # empty name
        "   ",  # name with spaces
    ],
)
def test_create_company_input_invalid_name_raises_error(name: str) -> None:
    # Arrange
    input_data = {
        "name": name,
    }
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        CreateCompanyInput.model_validate(input_data)
    assert exc_info.value.errors()[0]["loc"] == ("name",)


# Parametrize CNPJ with letters and CNPJ with 13 digits
@pytest.mark.parametrize(
    "cnpj",
    [
        "1234567890123",  # 13 digits
        "12A45678000195",  # contains letters
    ],
)
def test_create_company_input_invalid_cnpj_raises_error(cnpj: str) -> None:
    # Arrange
    input_data = {"name": "Test Company", "cnpj": cnpj}
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        CreateCompanyInput.model_validate(input_data)
    assert exc_info.value.errors()[0]["loc"] == ("cnpj",)


@pytest.mark.parametrize(
    "phone",
    [
        "invalid-phone",
        "+55 (12) 99456",  # incomplete phone number
        "1",  # 1 digit
        "+55129987987542",  # 14 digits
    ],
)
def test_create_contact_input_with_invalid_phone_raises_error(phone: str) -> None:
    # Arrange
    input_data = {"name": "John Doe", "email": "john.doe@example.com", "phone": phone}
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        CreateContactInput.model_validate(input_data)
    assert exc_info.value.errors()[0]["loc"] == ("phone",)


def test_create_contact_input_invalid_email_raises_error() -> None:
    # Arrange
    input_data = {"name": "John Doe", "email": "invalid-email", "phone": "1234567890"}
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        CreateContactInput.model_validate(input_data)
    assert exc_info.value.errors()[0]["loc"] == ("email",)
