from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4


@dataclass(frozen=True, kw_only=True)
class Company:
    id: UUID = field(default_factory=uuid4)
    name: str
    cnpj: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass(frozen=True, kw_only=True)
class Contact:
    id: UUID = field(default_factory=uuid4)
    name: str
    phone: str | None = None
    email: str | None = None
    company_id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
