from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class CreateCompanyInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    name: str = Field(
        min_length=1, max_length=200, description="The name of the company"
    )
    cnpj: str | None = Field(
        default=None,
        min_length=14,
        max_length=14,
        description="The CNPJ of the company (14 digits, numbers only). Omit if not provided",
    )

    @field_validator("cnpj", mode="before")
    @classmethod
    def normalize_cnpj(cls, v: object) -> object:
        if not isinstance(v, str):
            return v
        return _digits_only(v)


class CreateContactInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(
        min_length=1, max_length=200, description="The name of the contact"
    )
    phone: str | None = Field(
        default=None,
        min_length=10,
        max_length=13,
        description="The phone number of the contact (numbers only, optional country code). Omit if not provided.",
    )
    email: EmailStr | None = Field(
        default=None,
        min_length=5,
        max_length=200,
        description="The email address of the contact. Omit if not provided.",
    )
    company_id: UUID | None = Field(
        default=None,
        description="The ID of the company associated with the contact. Omit if not provided.",
    )

    @field_validator("phone", mode="before")
    @classmethod
    def normalize_phone(cls, v: object) -> object:
        if not isinstance(v, str):
            return v
        return _digits_only(v)


def _digits_only(value: str) -> str:
    return "".join(filter(str.isdigit, value))
