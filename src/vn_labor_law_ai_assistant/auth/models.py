from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Role = Literal["user", "admin"]
MessageRole = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class AuthUser:
    id: str
    name: str
    email: str
    role: Role
    avatar_url: str | None
    is_active: bool
