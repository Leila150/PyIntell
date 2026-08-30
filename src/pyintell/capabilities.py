"""High-level runtime capabilities used by PyIntell models and applications.

These are core infrastructure, not built-in AI tools. User-defined tools can use
these primitives, while code execution remains a separate explicit capability.
"""
from dataclasses import dataclass, field
from collections import OrderedDict
from contextlib import contextmanager
import hashlib, json, threading, time
from typing import Any, Callable, Dict, Iterable, Optional

class Config:
    def __init__(self, defaults=None):
        self._data = dict(defaults or {})
        self._lock = threading.RLock()
    def set(self, key, value):
        with self._lock: self._data[str(key)] = value
        return value
    def get(self, key, default=None):
        with self._lock: return self._data.get(str(key), default)
    def delete(self, key):
        with self._lock: return self._data.pop(str(key), None)
    def update(self, values):
        with self._lock: self._data.update(values)
        return self
    def as_dict(self):
        with self._lock: return dict(self._data)
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f: json.dump(self.as_dict(), f, indent=2, default=str)
    def load(self, path):
        with open(path, encoding="utf-8") as f: self._data.update(json.load(f))
        return self

class EventBus:
    def __init__(self): self._handlers = {}; self._lock = threading.RLock()
    def on(self, event, handler):
        if not callable(handler): raise TypeError("handler must be callable")
        with self._lock: self._handlers.setdefault(event, []).append(handler)
        return handler
    def off(self, event, handler):
        with self._lock:
            if event in self._handlers and handler in self._handlers[event]: self._handlers[event].remove(handler); return True
        return False
    def emit(self, event, **payload):
        with self._lock: handlers = list(self._handlers.get(event, ())) + list(self._handlers.get("*", ()))
        results=[]
        for handler in handlers:
            try: results.append(handler(**payload))
            except Exception as exc: results.append(exc)
        return results
    def clear(self): self._handlers.clear()

class LRUCache:
    def __init__(self, maxsize=256, ttl=None): self.maxsize=int(maxsize); self.ttl=ttl; self._data=OrderedDict(); self._lock=threading.RLock()
    def _expired(self, item): return self.ttl is not None and time.monotonic()-item[1] > self.ttl
    def get(self, key, default=None):
        with self._lock:
            item=self._data.get(key)
            if item is None or self._expired(item): self._data.pop(key,None); return default
            self._data.move_to_end(key); return item[0]
    def set(self, key, value):
        with self._lock:
            self._data[key]=(value,time.monotonic()); self._data.move_to_end(key)
            while len(self._data)>self.maxsize: self._data.popitem(last=False)
        return value
    def delete(self,key):
        with self._lock: return self._data.pop(key,None) is not None
    def clear(self): self._data.clear()
    def __len__(self): return len(self._data)

class ScopedMemory:
    def __init__(self): self._scopes={"default":{}}; self._current="default"; self._lock=threading.RLock()
    def use(self, scope):
        with self._lock: self._current=str(scope); self._scopes.setdefault(self._current,{})
        return self._current
    @property
    def scope(self): return self._current
    def set(self,key,value): self._scopes[self._current][key]=value; return value
    def get(self,key,default=None): return self._scopes[self._current].get(key,default)
    def delete(self,key): return self._scopes[self._current].pop(key,None)
    def clear(self, scope=None): self._scopes[scope or self._current].clear()
    def snapshot(self, scope=None): return dict(self._scopes[scope or self._current])

class Context:
    def __init__(self): self._values={}; self._lock=threading.RLock()
    def set(self, **values):
        with self._lock: self._values.update(values)
        return self
    def get(self,key,default=None): return self._values.get(key,default)
    def remove(self,key): return self._values.pop(key,None)
    def clear(self): self._values.clear()
    def snapshot(self): return dict(self._values)
    @contextmanager
    def push(self, **values):
        old=self.snapshot(); self.set(**values)
        try: yield self
        finally: self._values=old

@dataclass
class TaskStep:
    description: str
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class TaskPlan:
    goal: str
    steps: list[TaskStep] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    def add(self, description, **metadata):
        step=TaskStep(description, metadata=metadata); self.steps.append(step); return step
    def complete(self, step, result=None): step.status="completed"; step.result=result; return step
    def fail(self, step, error): step.status="failed"; step.error=str(error); return step
    def pending(self): return [s for s in self.steps if s.status=="pending"]
    def as_dict(self): return {"goal":self.goal,"steps":[s.__dict__.copy() for s in self.steps],"metadata":dict(self.metadata)}

class RetryPolicy:
    def __init__(self, attempts=3, delay=0.2, backoff=2.0): self.attempts=max(1,int(attempts)); self.delay=float(delay); self.backoff=float(backoff)
    def run(self, fn, *args, **kwargs):
        last=None; delay=self.delay
        for i in range(self.attempts):
            try: return fn(*args, **kwargs)
            except Exception as exc:
                last=exc
                if i+1<self.attempts: time.sleep(delay); delay*=self.backoff
        raise last

def fingerprint(value):
    raw=json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()

config=Config({"execution.enabled": True, "execution.default_timeout": 30, "generation.streaming": True})
events=EventBus(); cache=LRUCache(); memory=ScopedMemory(); context=Context()
