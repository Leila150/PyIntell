"""Fine-grained permission policies for PyIntell tools."""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class ToolPermissionError(PermissionError):
    """Raised when a tool call is denied by its permission policy."""


class ToolConfirmationRequired(ToolPermissionError):
    """Raised when a tool requires explicit confirmation before execution."""


@dataclass
class ToolPermissions:
    """Runtime policy controlling whether and how a tool may execute."""
    allowed: bool = True
    require_confirmation: bool = False
    max_calls: Optional[int] = None
    timeout: Optional[float] = None
    output_limit: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confirmation_callback: Optional[Callable[..., bool]] = field(default=None, repr=False, compare=False)
    calls: int = 0

    def check(self, tool_name: str, *, confirmed: bool = False) -> None:
        if not self.allowed:
            raise ToolPermissionError(f"Tool '{tool_name}' is denied by its permission policy")
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise ToolPermissionError(f"Tool '{tool_name}' has reached its call limit")
        if self.require_confirmation and not confirmed:
            if self.confirmation_callback is None:
                raise ToolConfirmationRequired(f"Tool '{tool_name}' requires confirmation")
            try:
                accepted = bool(self.confirmation_callback(tool_name))
            except TypeError:
                accepted = bool(self.confirmation_callback())
            if not accepted:
                raise ToolConfirmationRequired(f"Tool '{tool_name}' was not confirmed")

    def record_call(self) -> None:
        self.calls += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "require_confirmation": self.require_confirmation,
            "max_calls": self.max_calls,
            "timeout": self.timeout,
            "output_limit": self.output_limit,
            "calls": self.calls,
            "metadata": dict(self.metadata),
        }
