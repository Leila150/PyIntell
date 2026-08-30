"""Dynamic, introspectable user-tool registry.

PyIntell deliberately does not ship a fixed set of AI tools. Applications register
their own capabilities here. The registry provides metadata, validation,
permissions, enable/disable state, aliases, and safe introspection.
"""
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
    aliases: tuple[str, ...] = ()

    def __post_init__(self):
        if not callable(self.function): raise TypeError("function must be callable")
        if not isinstance(self.name, str) or not self.name.strip(): raise ValueError("tool name must be a non-empty string")
        self.name = self.name.strip()
        self.aliases = tuple(str(x).strip() for x in self.aliases if str(x).strip())

    def __call__(self, *args, **kwargs):
        if not self.enabled: raise RuntimeError(f"Tool '{self.name}' is disabled")
        return self.function(*args, **kwargs)

    @property
    def signature(self):
        try: return str(inspect.signature(self.function))
        except (TypeError, ValueError): return "(*args, **kwargs)"

    @property
    def source(self):
        try: return inspect.getsource(self.function)
        except (OSError, TypeError): return None

    def schema(self):
        try:
            sig = inspect.signature(self.function)
            parameters = []
            for p in sig.parameters.values():
                parameters.append({"name": p.name, "kind": p.kind.name,
                                   "required": p.default is inspect.Parameter.empty,
                                   "default": None if p.default is inspect.Parameter.empty else repr(p.default),
                                   "annotation": None if p.annotation is inspect.Parameter.empty else str(p.annotation)})
            return_type = None if sig.return_annotation is inspect.Signature.empty else str(sig.return_annotation)
        except (TypeError, ValueError):
            parameters, return_type = [], None
        return {"name": self.name, "description": self.description, "signature": self.signature,
                "parameters": parameters, "return": return_type, "enabled": self.enabled,
                "trusted": self.trusted, "aliases": list(self.aliases), "metadata": dict(self.metadata)}

class ToolRegistry:
    def __init__(self): self._tools: Dict[str, Tool] = {}; self._aliases: Dict[str, str] = {}

    def _resolve(self, name):
        key = name if isinstance(name, str) else getattr(name, "__name__", None)
        if not key: raise TypeError("tool must be a function or name")
        key = str(key)
        return self._aliases.get(key, key)

    def add(self, function=None, name: Optional[str] = None, description: str = "",
            trusted: bool = False, replace: bool = False, aliases=(), **metadata):
        def register(fn):
            if not callable(fn): raise TypeError("tool must be callable")
            tool_name = (name or getattr(fn, "__name__", None) or "").strip()
            if not tool_name: raise ValueError("tool function needs a name")
            if tool_name in self._tools and not replace: raise ValueError(f"Tool '{tool_name}' is already registered")
            if replace and tool_name in self._tools: self.remove(tool_name)
            item = Tool(fn, tool_name, description or inspect.getdoc(fn) or "", True, trusted, metadata, tuple(aliases))
            self._tools[tool_name] = item
            for alias in item.aliases:
                if alias in self._tools or alias in self._aliases: raise ValueError(f"Tool alias '{alias}' is already registered")
                self._aliases[alias] = tool_name
            return fn
        return register(function) if function is not None else register

    def remove(self, function):
        name = self._resolve(function)
        if name not in self._tools: return False
        item = self._tools.pop(name)
        for alias in item.aliases: self._aliases.pop(alias, None)
        return True

    def edit(self, function, contents=None, new_name=None, **changes):
        old = self._resolve(function)
        if old not in self._tools: raise KeyError(old)
        item = self._tools[old]
        old_aliases = item.aliases
        if contents is not None:
            if not callable(contents): raise TypeError("contents must be callable")
            item.function = contents
        target = (new_name or old).strip() if isinstance(new_name or old, str) else new_name
        if not target: raise ValueError("new_name must be a non-empty string")
        if target != old and target in self._tools: raise ValueError(f"Tool '{target}' is already registered")
        for key, value in changes.items():
            if key in {"function", "name"}: raise ValueError(f"use contents/new_name to change '{key}'")
            if key == "aliases":
                for alias in old_aliases: self._aliases.pop(alias, None)
                item.aliases = tuple(value)
                for alias in item.aliases: self._aliases[alias] = target
            elif hasattr(item, key): setattr(item, key, value)
            else: item.metadata[key] = value
        if target != old:
            self._tools.pop(old); item.name = target; self._tools[target] = item
            for alias in item.aliases: self._aliases[alias] = target
        return item.function

    def get(self, function): return self._tools[self._resolve(function)].function
    def get_tool(self, name): return self._tools[self._resolve(name)]
    def schema(self, name): return self.get_tool(name).schema()
    def enable(self, name): self.get_tool(name).enabled = True; return self.get_tool(name)
    def disable(self, name): self.get_tool(name).enabled = False; return self.get_tool(name)
    def is_enabled(self, name): return self.get_tool(name).enabled
    def clear(self): self._tools.clear(); self._aliases.clear()
    def names(self, enabled_only=False): return [n for n,t in self._tools.items() if not enabled_only or t.enabled]
    def list(self, enabled_only=False): return [t.schema() for t in self._tools.values() if not enabled_only or t.enabled]
    def call(self, name, *args, **kwargs): return self.get_tool(name)(*args, **kwargs)
    def __contains__(self, name): return self._resolve(name) in self._tools
    def __len__(self): return len(self._tools)

tool = ToolRegistry()
add, remove, edit, get = tool.add, tool.remove, tool.edit, tool.get
