"""Dynamic, introspectable PyIntell tool registry."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from collections.abc import Mapping
import inspect
import threading

from .permissions import ToolPermissions, ToolPermissionError, ToolConfirmationRequired


class ToolDisabledError(RuntimeError):
    """Raised when a disabled tool is called."""


@dataclass
class Tool:
    function: Callable
    name: str
    description: str = ""
    enabled: bool = True
    trusted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    permissions: ToolPermissions = field(default_factory=ToolPermissions)
    _on_enable: Optional[Callable[[], Any]] = field(default=None, repr=False, compare=False)
    _on_disable: Optional[Callable[[], Any]] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not callable(self.function): raise TypeError("function must be callable")
        if not isinstance(self.name, str) or not self.name.strip(): raise ValueError("tool name must be a non-empty string")
        self.name = self.name.strip()
        self.aliases = tuple(dict.fromkeys(str(x).strip() for x in self.aliases if str(x).strip()))
        self.metadata = dict(self.metadata or {})
        if not isinstance(self.permissions, ToolPermissions):
            self.permissions = ToolPermissions(**dict(self.permissions or {}))

    def __call__(self, *args, **kwargs):
        if not self.enabled: raise ToolDisabledError(f"Tool '{self.name}' is disabled")
        confirmed = bool(kwargs.pop("_confirmed", False))
        caller = kwargs.pop("_caller", None)
        trusted = bool(kwargs.pop("_trusted", True))
        self.permissions.check(self.name, confirmed=confirmed, caller=caller, trusted=trusted)
        call_kwargs = dict(kwargs)
        if self.permissions.timeout is not None and "timeout" not in call_kwargs:
            call_kwargs["timeout"] = self.permissions.timeout
        result = self.function(*args, **call_kwargs)
        self.permissions.record_call()
        return result

    @property
    def signature(self):
        try: return str(inspect.signature(self.function))
        except (TypeError, ValueError): return "(*args, **kwargs)"

    @property
    def source(self):
        try: return inspect.getsource(self.function)
        except (OSError, TypeError): return None

    @property
    def module(self): return getattr(self.function, "__module__", None)
    @property
    def qualified_name(self): return getattr(self.function, "__qualname__", self.name)

    def schema(self):
        try:
            sig = inspect.signature(self.function)
            parameters = []
            for p in sig.parameters.values():
                required = p.default is inspect.Parameter.empty
                parameters.append({"name": p.name, "kind": p.kind.name, "required": required,
                                   "default": None if required else repr(p.default),
                                   "annotation": None if p.annotation is inspect.Parameter.empty else str(p.annotation)})
            return_type = None if sig.return_annotation is inspect.Signature.empty else str(sig.return_annotation)
        except (TypeError, ValueError): parameters, return_type = [], None
        return {"name": self.name, "description": self.description, "signature": self.signature,
                "parameters": parameters, "return": return_type, "enabled": self.enabled,
                "trusted": self.trusted, "aliases": list(self.aliases), "metadata": dict(self.metadata),
                "permissions": self.permissions.as_dict(), "module": self.module, "qualified_name": self.qualified_name}


class ToolRegistry:
    """Thread-safe registry for callable AI tools with fine-grained permissions."""
    def __init__(self):
        self._tools: Dict[str, Tool] = {}; self._aliases: Dict[str, str] = {}; self._lock = threading.RLock()

    @staticmethod
    def _key(value):
        key = value if isinstance(value, str) else getattr(value, "__name__", None)
        if not key: raise TypeError("tool must be a function, Tool, or name")
        return str(key).strip()

    def _resolve(self, value):
        key = value.name if isinstance(value, Tool) else self._key(value)
        return self._aliases.get(key, key)

    def add(self, function=None, name: Optional[str] = None, description: str = "", trusted: bool = False,
            replace: bool = False, aliases=(), enabled: bool = True, permissions: Optional[ToolPermissions] = None, **metadata):
        def register(fn):
            if isinstance(fn, Tool): raise TypeError("use registry.add(function=...) with a callable, not a Tool instance")
            if not callable(fn): raise TypeError("tool must be callable")
            tool_name = str(name or getattr(fn, "__name__", "")).strip()
            if not tool_name: raise ValueError("tool function needs a name")
            alias_values = tuple(dict.fromkeys(str(x).strip() for x in aliases if str(x).strip()))
            with self._lock:
                if tool_name in self._tools or tool_name in self._aliases:
                    if not replace: raise ValueError(f"Tool '{tool_name}' is already registered")
                    self.remove(tool_name)
                conflicts = [a for a in alias_values if a == tool_name or a in self._tools or a in self._aliases]
                if conflicts: raise ValueError(f"Tool alias is already registered: {conflicts[0]!r}")
                item = Tool(fn, tool_name, description or inspect.getdoc(fn) or "", bool(enabled), trusted, dict(metadata), alias_values,
                            permissions or ToolPermissions())
                self._tools[tool_name] = item
                for alias in alias_values: self._aliases[alias] = tool_name
                return fn
        return register(function) if function is not None else register

    def remove(self, function):
        with self._lock:
            name = self._resolve(function); item = self._tools.pop(name, None)
            if item is None: return False
            for alias in item.aliases: self._aliases.pop(alias, None)
            if item.enabled and item._on_disable: item._on_disable()
            return True

    def edit(self, function, contents=None, new_name=None, **changes):
        with self._lock:
            old = self._resolve(function)
            if old not in self._tools: raise KeyError(old)
            item = self._tools[old]
            if contents is not None and not callable(contents): raise TypeError("contents must be callable")
            target = old if new_name is None else str(new_name).strip()
            if not target: raise ValueError("new_name must be a non-empty string")
            if target != old and (target in self._tools or target in self._aliases): raise ValueError(f"Tool '{target}' is already registered")
            new_aliases = item.aliases if "aliases" not in changes else tuple(dict.fromkeys(str(x).strip() for x in changes["aliases"] if str(x).strip()))
            reserved = (set(self._tools) - {old}) | (set(self._aliases) - set(item.aliases))
            for alias in new_aliases:
                if alias == target or alias in reserved: raise ValueError(f"Tool alias is already registered: {alias!r}")
            for key in changes:
                if key in {"function", "name"}: raise ValueError(f"use contents/new_name to change '{key}'")
            new_function = contents if contents is not None else item.function
            new_metadata = dict(item.metadata)
            for key, value in changes.items():
                if key not in {"aliases", "description", "enabled", "trusted", "permissions"}: new_metadata[key] = value
            new_permissions = changes.get("permissions", item.permissions)
            if isinstance(new_permissions, dict): new_permissions = ToolPermissions(**new_permissions)
            new_item = Tool(new_function, target, changes.get("description", item.description), bool(changes.get("enabled", item.enabled)),
                            bool(changes.get("trusted", item.trusted)), new_metadata, new_aliases, new_permissions,
                            item._on_enable, item._on_disable)
            was_enabled = item.enabled
            self._tools.pop(old)
            for alias in item.aliases: self._aliases.pop(alias, None)
            self._tools[target] = new_item
            for alias in new_aliases: self._aliases[alias] = target
            if was_enabled and not new_item.enabled and new_item._on_disable: new_item._on_disable()
            elif not was_enabled and new_item.enabled and new_item._on_enable: new_item._on_enable()
            return new_item.function

    def get(self, function): return self.get_tool(function).function
    def get_tool(self, name):
        with self._lock:
            resolved = self._resolve(name)
            try: return self._tools[resolved]
            except KeyError: raise KeyError(f"Tool '{resolved}' is not registered") from None
    def schema(self, name): return self.get_tool(name).schema()

    def permissions(self, name): return self.get_tool(name).permissions

    def set_permissions(self, name, **changes):
        with self._lock:
            item = self.get_tool(name)
            current = item.permissions
            for key, value in changes.items():
                if not hasattr(current, key): raise ValueError(f"Unknown permission: {key}")
                if key == "max_calls" and value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                    raise ValueError("max_calls must be a non-negative integer or None")
                if key == "timeout" and value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0):
                    raise ValueError("timeout must be positive or None")
                if key == "output_limit" and value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                    raise ValueError("output_limit must be a non-negative integer or None")
                if key == "rate_limit" and value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                    raise ValueError("rate_limit must be a non-negative integer or None")
                if key == "rate_window" and value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0):
                    raise ValueError("rate_window must be positive or None")
                if key in {"allowed_callers", "blocked_callers"} and value is not None:
                    value = tuple(dict.fromkeys(str(x).strip() for x in value if str(x).strip()))
                setattr(current, key, value)
            return current

    def allow(self, name): return self.set_permissions(name, allowed=True)
    def deny(self, name): return self.set_permissions(name, allowed=False)
    def require_confirmation(self, name, required=True, callback=None):
        changes = {"require_confirmation": bool(required)}
        if callback is not None: changes["confirmation_callback"] = callback
        return self.set_permissions(name, **changes)
    def set_timeout(self, name, timeout): return self.set_permissions(name, timeout=timeout)
    def set_output_limit(self, name, limit): return self.set_permissions(name, output_limit=limit)
    def set_call_limit(self, name, limit): return self.set_permissions(name, max_calls=limit)
    def require_trusted(self, name, required=True): return self.set_permissions(name, require_trusted=bool(required))
    def require_explicit_enable(self, name, required=True): return self.set_permissions(name, require_explicit_enable=bool(required))
    def allow_callers(self, name, callers): return self.set_permissions(name, allowed_callers=callers)
    def block_callers(self, name, callers): return self.set_permissions(name, blocked_callers=callers)
    def set_rate_limit(self, name, calls, window): return self.set_permissions(name, rate_limit=calls, rate_window=window)
    def reset_call_count(self, name):
        permissions = self.permissions(name); permissions.reset_calls(); return permissions

    def enable(self, name):
        with self._lock:
            item = self.get_tool(name)
            if not item.enabled:
                if item._on_enable: item._on_enable()
                item.enabled = True
            return item
    def disable(self, name):
        with self._lock:
            item = self.get_tool(name)
            if item.enabled:
                if item._on_disable: item._on_disable()
                item.enabled = False
            return item
    def is_enabled(self, name): return self.get_tool(name).enabled
    def clear(self):
        with self._lock:
            items = list(self._tools.values()); self._tools.clear(); self._aliases.clear()
            for item in items:
                if item.enabled and item._on_disable: item._on_disable()
    def names(self, enabled_only=False):
        with self._lock: return [n for n, t in self._tools.items() if not enabled_only or t.enabled]
    def list(self, enabled_only=False):
        with self._lock: return [t.schema() for t in self._tools.values() if not enabled_only or t.enabled]
    def call(self, name, *args, **kwargs): return self.get_tool(name)(*args, **kwargs)
    def has(self, name):
        with self._lock:
            try: return self._resolve(name) in self._tools
            except TypeError: return False
    def __contains__(self, name): return self.has(name)
    def __len__(self):
        with self._lock: return len(self._tools)


tool = ToolRegistry()
add, remove, edit, get = tool.add, tool.remove, tool.edit, tool.get


def _code_execution_impl(code: str, language: str = "python", timeout=None, stdin=None, filename=None, cwd=None,
                         env: Optional[Mapping[str, str]] = None, args=(), keep_file=False, compile=True) -> dict[str, Any]:
    """Execute development code through PyIntell's opt-in execution engine."""
    from .execution import executor
    return executor.run(code, language, timeout=timeout, stdin=stdin, filename=filename, cwd=cwd,
                        env=env, args=args, keep_file=keep_file, compile=compile).as_dict()


def _enable_execution():
    from .execution import executor
    executor.enable()


def _disable_execution():
    from .execution import executor
    executor.disable()


tool.add(_code_execution_impl, name="code_execution",
         description="Execute code using an installed development runtime or compiler and return structured output.",
         trusted=True, aliases=("execute_code", "run_code"), category="development", dangerous=True,
         requires_explicit_enable=True)
_code_tool = tool.get_tool("code_execution")
_code_tool._on_enable = _enable_execution
_code_tool._on_disable = _disable_execution
tool.disable("code_execution")
code_execution = _code_tool
