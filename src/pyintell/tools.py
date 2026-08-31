"""Dynamic, introspectable PyIntell tool registry."""
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
    _on_enable: Optional[Callable[[], Any]] = field(default=None, repr=False, compare=False)
    _on_disable: Optional[Callable[[], Any]] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not callable(self.function): raise TypeError("function must be callable")
        if not isinstance(self.name, str) or not self.name.strip(): raise ValueError("tool name must be a non-empty string")
        self.name = self.name.strip()
        self.aliases = tuple(dict.fromkeys(str(x).strip() for x in self.aliases if str(x).strip()))

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
            parameters = [{"name": p.name, "kind": p.kind.name, "required": p.default is inspect.Parameter.empty,
                           "default": None if p.default is inspect.Parameter.empty else repr(p.default),
                           "annotation": None if p.annotation is inspect.Parameter.empty else str(p.annotation)}
                          for p in sig.parameters.values()]
            return_type = None if sig.return_annotation is inspect.Signature.empty else str(sig.return_annotation)
        except (TypeError, ValueError): parameters, return_type = [], None
        return {"name": self.name, "description": self.description, "signature": self.signature,
                "parameters": parameters, "return": return_type, "enabled": self.enabled,
                "trusted": self.trusted, "aliases": list(self.aliases), "metadata": dict(self.metadata)}

class ToolRegistry:
    def __init__(self): self._tools: Dict[str, Tool] = {}; self._aliases: Dict[str, str] = {}
    def _resolve(self, value):
        key = value if isinstance(value, str) else getattr(value, "__name__", None)
        if not key: raise TypeError("tool must be a function or name")
        return self._aliases.get(str(key).strip(), str(key).strip())
    def add(self, function=None, name: Optional[str] = None, description: str = "", trusted: bool = False,
            replace: bool = False, aliases=(), **metadata):
        def register(fn):
            if not callable(fn): raise TypeError("tool must be callable")
            tool_name = str(name or getattr(fn, "__name__", "")).strip()
            if not tool_name: raise ValueError("tool function needs a name")
            alias_values = tuple(dict.fromkeys(str(x).strip() for x in aliases if str(x).strip()))
            if tool_name in self._tools or tool_name in self._aliases:
                if not replace: raise ValueError(f"Tool '{tool_name}' is already registered")
                self.remove(tool_name)
            conflicts = [a for a in alias_values if a == tool_name or a in self._tools or a in self._aliases]
            if conflicts: raise ValueError(f"Tool alias is already registered: {conflicts[0]!r}")
            item = Tool(fn, tool_name, description or inspect.getdoc(fn) or "", True, trusted, dict(metadata), alias_values)
            self._tools[tool_name] = item
            for alias in alias_values: self._aliases[alias] = tool_name
            return fn
        return register(function) if function is not None else register
    def remove(self, function):
        name = self._resolve(function); item = self._tools.pop(name, None)
        if item is None: return False
        for alias in item.aliases: self._aliases.pop(alias, None)
        if item.enabled and item._on_disable: item._on_disable()
        return True
    def edit(self, function, contents=None, new_name=None, **changes):
        old = self._resolve(function)
        if old not in self._tools: raise KeyError(old)
        item = self._tools[old]
        if contents is not None and not callable(contents): raise TypeError("contents must be callable")
        target = old if new_name is None else str(new_name).strip()
        if not target: raise ValueError("new_name must be a non-empty string")
        if target != old and (target in self._tools or target in self._aliases): raise ValueError(f"Tool '{target}' is already registered")
        new_aliases = item.aliases if "aliases" not in changes else tuple(dict.fromkeys(str(x).strip() for x in changes["aliases"] if str(x).strip()))
        reserved = set(self._tools) | set(self._aliases); reserved.discard(old)
        for alias in new_aliases:
            if alias == target or alias in reserved: raise ValueError(f"Tool alias is already registered: {alias!r}")
        for key in changes:
            if key in {"function", "name"}: raise ValueError(f"use contents/new_name to change '{key}'")
        new_function = contents if contents is not None else item.function
        new_metadata = dict(item.metadata)
        for key, value in changes.items():
            if key not in {"aliases", "description", "enabled", "trusted"}: new_metadata[key] = value
        new_item = Tool(new_function, target, changes.get("description", item.description),
                        bool(changes.get("enabled", item.enabled)), bool(changes.get("trusted", item.trusted)),
                        new_metadata, new_aliases, item._on_enable, item._on_disable)
        was_enabled = item.enabled
        self._tools.pop(old)
        for alias in item.aliases: self._aliases.pop(alias, None)
        self._tools[target] = new_item
        for alias in new_aliases: self._aliases[alias] = target
        if was_enabled and not new_item.enabled and new_item._on_disable: new_item._on_disable()
        elif not was_enabled and new_item.enabled and new_item._on_enable: new_item._on_enable()
        return new_item.function
    def get(self, function): return self.get_tool(function).function
    def get_tool(self, name): return self._tools[self._resolve(name)]
    def schema(self, name): return self.get_tool(name).schema()
    def enable(self, name):
        item = self.get_tool(name)
        if not item.enabled:
            item.enabled = True
            if item._on_enable: item._on_enable()
        return item
    def disable(self, name):
        item = self.get_tool(name)
        if item.enabled:
            item.enabled = False
            if item._on_disable: item._on_disable()
        return item
    def is_enabled(self, name): return self.get_tool(name).enabled
    def clear(self): self._tools.clear(); self._aliases.clear()
    def names(self, enabled_only=False): return [n for n, t in self._tools.items() if not enabled_only or t.enabled]
    def list(self, enabled_only=False): return [t.schema() for t in self._tools.values() if not enabled_only or t.enabled]
    def call(self, name, *args, **kwargs): return self.get_tool(name)(*args, **kwargs)
    def __contains__(self, name):
        try: return self._resolve(name) in self._tools
        except TypeError: return False
    def __len__(self): return len(self._tools)

tool = ToolRegistry()
add, remove, edit, get = tool.add, tool.remove, tool.edit, tool.get


def _code_execution_impl(code: str, language: str = "python", timeout=None, stdin=None, filename=None, cwd=None,
                         env=None, args=(), keep_file=False, compile=True):
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
tool.get_tool("code_execution")._on_enable = _enable_execution
tool.get_tool("code_execution")._on_disable = _disable_execution
tool.disable("code_execution")
code_execution = tool.get_tool("code_execution")
