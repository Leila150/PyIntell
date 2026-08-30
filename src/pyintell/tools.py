"""Dynamic tool registry for PyIntell."""
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

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            raise RuntimeError(f"Tool '{self.name}' is disabled")
        return self.function(*args, **kwargs)

    @property
    def signature(self):
        return str(inspect.signature(self.function))

    def schema(self):
        return {"name": self.name, "description": self.description,
                "signature": self.signature, "enabled": self.enabled,
                "trusted": self.trusted, "metadata": dict(self.metadata)}

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def add(self, function=None, name: Optional[str] = None, description: str = "", trusted: bool = False, **metadata):
        def register(fn):
            tool_name = name or fn.__name__
            self._tools[tool_name] = Tool(fn, tool_name, description or inspect.getdoc(fn) or "", True, trusted, metadata)
            return fn
        return register(function) if function is not None else register

    def remove(self, function):
        name = function if isinstance(function, str) else function.__name__
        return self._tools.pop(name, None) is not None

    def edit(self, function, contents=None, new_name=None, **changes):
        old = function if isinstance(function, str) else function.__name__
        if old not in self._tools:
            raise KeyError(old)
        tool = self._tools.pop(old)
        if contents is not None:
            if not callable(contents):
                raise TypeError("contents must be callable")
            tool.function = contents
        tool.name = new_name or old
        for key, value in changes.items():
            if hasattr(tool, key): setattr(tool, key, value)
            else: tool.metadata[key] = value
        self._tools[tool.name] = tool
        return tool.function

    def get(self, function):
        name = function if isinstance(function, str) else function.__name__
        return self._tools[name].function

    def get_tool(self, name):
        return self._tools[name]

    def enable(self, name): self._tools[name].enabled = True
    def disable(self, name): self._tools[name].enabled = False
    def is_enabled(self, name): return self._tools[name].enabled
    def list(self, enabled_only=False):
        return [t.schema() for t in self._tools.values() if not enabled_only or t.enabled]
    def call(self, name, *args, **kwargs): return self._tools[name](*args, **kwargs)


tool = ToolRegistry()
add = tool.add
remove = tool.remove
edit = tool.edit
get = tool.get
