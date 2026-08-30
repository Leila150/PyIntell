"""Dynamic, introspectable tool registry for PyIntell."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
import inspect

@dataclass
class Tool:
    function: Callable
    name: str
    description: str = ""
    enabled: bool = True
    trusted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not callable(self.function):
            raise TypeError("function must be callable")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool name must be a non-empty string")

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            raise RuntimeError(f"Tool '{self.name}' is disabled")
        return self.function(*args, **kwargs)

    @property
    def signature(self):
        try:
            return str(inspect.signature(self.function))
        except (TypeError, ValueError):
            return "(*args, **kwargs)"

    @property
    def source(self):
        try:
            return inspect.getsource(self.function)
        except (OSError, TypeError):
            return None

    def schema(self):
        return {"name": self.name, "description": self.description,
                "signature": self.signature, "enabled": self.enabled,
                "trusted": self.trusted, "metadata": dict(self.metadata)}

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def add(self, function=None, name: Optional[str] = None, description: str = "",
            trusted: bool = False, replace: bool = False, **metadata):
        """Register a function, directly or with ``@tool.add(...)``."""
        def register(fn):
            if not callable(fn):
                raise TypeError("tool must be callable")
            tool_name = name or getattr(fn, "__name__", None)
            if not tool_name:
                raise ValueError("tool function needs a name")
            if tool_name in self._tools and not replace:
                raise ValueError(f"Tool '{tool_name}' is already registered")
            self._tools[tool_name] = Tool(fn, tool_name, description or inspect.getdoc(fn) or "", True, trusted, metadata)
            return fn
        return register(function) if function is not None else register

    def remove(self, function):
        name = function if isinstance(function, str) else getattr(function, "__name__", None)
        if not name:
            raise TypeError("function must be a function or tool name")
        return self._tools.pop(name, None) is not None

    def edit(self, function, contents=None, new_name=None, **changes):
        """Edit a registered tool while preventing accidental name collisions."""
        old = function if isinstance(function, str) else getattr(function, "__name__", None)
        if old not in self._tools:
            raise KeyError(old)
        item = self._tools.pop(old)
        try:
            if contents is not None:
                if not callable(contents):
                    raise TypeError("contents must be callable")
                item.function = contents
            target = new_name or old
            if not isinstance(target, str) or not target.strip():
                raise ValueError("new_name must be a non-empty string")
            if target != old and target in self._tools:
                raise ValueError(f"Tool '{target}' is already registered")
            item.name = target
            for key, value in changes.items():
                if key in {"function", "name"}:
                    raise ValueError(f"use contents/new_name to change '{key}'")
                if hasattr(item, key):
                    setattr(item, key, value)
                else:
                    item.metadata[key] = value
            self._tools[target] = item
            return item.function
        except Exception:
            self._tools[old] = item
            raise

    def get(self, function):
        name = function if isinstance(function, str) else getattr(function, "__name__", None)
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name].function

    def get_tool(self, name): return self._tools[name]
    def schema(self, name): return self.get_tool(name).schema()
    def enable(self, name): self._tools[name].enabled = True; return self._tools[name]
    def disable(self, name): self._tools[name].enabled = False; return self._tools[name]
    def is_enabled(self, name): return self._tools[name].enabled
    def clear(self): self._tools.clear()
    def names(self, enabled_only=False):
        return [n for n, t in self._tools.items() if not enabled_only or t.enabled]
    def list(self, enabled_only=False):
        return [t.schema() for t in self._tools.values() if not enabled_only or t.enabled]
    def call(self, name, *args, **kwargs): return self._tools[name](*args, **kwargs)
    def __contains__(self, name): return name in self._tools
    def __len__(self): return len(self._tools)

tool = ToolRegistry()
add = tool.add
remove = tool.remove
edit = tool.edit
get = tool.get
