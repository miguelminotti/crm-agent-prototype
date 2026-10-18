class CRMError(Exception):
    """Base class for CRM-related errors."""


class CompanyNotFoundError(CRMError):
    """Raised when a company is not found."""


class DuplicateCompanyError(CRMError):
    """Raised when attempting to create a company that already exists."""
