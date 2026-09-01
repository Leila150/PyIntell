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
    rate_limit: Optional[int] = None
    rate_window: Optional[float] = None
    require_trusted: bool = False
    require_explicit_enable: bool = False
    allowed_callers: Optional[tuple[str, ...]] = None
    blocked_callers: tuple[str, ...] = ()

    def check(self, tool_name: str, *, confirmed: bool = False, caller: Optional[str] = None,
              trusted: bool = True) -> None:
        if not self.allowed:
            raise ToolPermissionError(f"Tool '{tool_name}' is denied by its permission policy")
        if self.require_trusted and not trusted:
            raise ToolPermissionError(f"Tool '{tool_name}' requires a trusted caller")
        if self.allowed_callers is not None and caller not in self.allowed_callers:
            raise ToolPermissionError(f"Caller is not allowed to use tool '{tool_name}'")
        if caller is not None and caller in self.blocked_callers:
            raise ToolPermissionError(f"Caller is blocked from using tool '{tool_name}'")
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

    def reset_calls(self) -> None:
        self.calls = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "require_confirmation": self.require_confirmation,
            "max_calls": self.max_calls,
            "timeout": self.timeout,
            "output_limit": self.output_limit,
            "rate_limit": self.rate_limit,
            "rate_window": self.rate_window,
            "require_trusted": self.require_trusted,
            "require_explicit_enable": self.require_explicit_enable,
            "allowed_callers": list(self.allowed_callers) if self.allowed_callers is not None else None,
            "blocked_callers": list(self.blocked_callers),
            "calls": self.calls,
            "metadata": dict(self.metadata),
        }
