"""Reusable permission profiles and helpers for PyIntell tools."""
from dataclasses import dataclass
from typing import Any, Optional

from .permissions import ToolPermissions


@dataclass(frozen=True)
class PermissionProfile:
    """A reusable baseline policy for tools."""
    allowed: bool = True
    require_confirmation: bool = False
    max_calls: Optional[int] = None
    timeout: Optional[float] = None
    output_limit: Optional[int] = None
    metadata: dict[str, Any] | None = None

    def build(self) -> ToolPermissions:
        return ToolPermissions(
            allowed=self.allowed,
            require_confirmation=self.require_confirmation,
            max_calls=self.max_calls,
            timeout=self.timeout,
            output_limit=self.output_limit,
            metadata=dict(self.metadata or {}),
        )


SAFE = PermissionProfile()
CONFIRM = PermissionProfile(require_confirmation=True)
DENY = PermissionProfile(allowed=False)
ONE_SHOT = PermissionProfile(max_calls=1)
RESTRICTED = PermissionProfile(require_confirmation=True, max_calls=10, timeout=30, output_limit=100_000)


def apply_profile(permissions: ToolPermissions, profile: PermissionProfile) -> ToolPermissions:
    """Apply a profile to an existing permission object."""
    permissions.allowed = profile.allowed
    permissions.require_confirmation = profile.require_confirmation
    permissions.max_calls = profile.max_calls
    permissions.timeout = profile.timeout
    permissions.output_limit = profile.output_limit
    if profile.metadata is not None:
        permissions.metadata = dict(profile.metadata)
    return permissions
