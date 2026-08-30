"""Core runtime capabilities; these are infrastructure, not built-in AI tools."""
from dataclasses import dataclass, field
from collections import OrderedDict
from contextlib import contextmanager
import hashlib, json, threading, time
from typing import Any, Callable, Optional

class Config:
    def __init__(self, defaults=None): self._data=dict(defaults or {}); self._lock=threading.RLock()
    def set(self,key,value):
        with self._lock:
            self._data[str(key)]=value
            return value
    def get(self,key,default=None):
        with self._lock: return self._data.get(str(key),default)
    def delete(self,key):
        with self._lock: return self._data.pop(str(key),None)
    def update(self,values):
        with self._lock: self._data.update(dict(values))
        return self
    def as_dict(self):
        with self._lock: return dict(self._data)
    def save(self,path):
        with open(path,"w",encoding="utf-8") as f: json.dump(self.as_dict(),f,indent=2,default=str)
    def load(self,path):
        with open(path,encoding="utf-8") as f: self.update(json.load(f))
        return self

class EventBus:
    def __init__(self): self._handlers={}; self._lock=threading.RLock()
    def on(self,event,handler):
        if not callable(handler): raise TypeError("handler must be callable")
        with self._lock: self._handlers.setdefault(str(event),[]).append(handler)
        return handler
    def off(self,event,handler):
        with self._lock:
            handlers=self._handlers.get(str(event),[])
            if handler in handlers: handlers.remove(handler); return True
        return False
    def emit(self,event,**payload):
        with self._lock: handlers=list(self._handlers.get(str(event),()))+list(self._handlers.get("*",()))
        results=[]
        for handler in handlers:
            try: results.append(handler(**payload))
            except Exception as exc: results.append(exc)
        return results
    def clear(self,event=None):
        with self._lock:
            if event is None: self._handlers.clear()
            else: self._handlers.pop(str(event),None)

class LRUCache:
    def __init__(self,maxsize=256,ttl=None): self.maxsize=max(1,int(maxsize)); self.ttl=ttl; self._data=OrderedDict(); self._lock=threading.RLock()
    def get(self,key,default=None):
        with self._lock:
            item=self._data.get(key)
            if item is None: return default
            if self.ttl is not None and time.monotonic()-item[1]>self.ttl: self._data.pop(key,None); return default
            self._data.move_to_end(key); return item[0]
    def set(self,key,value):
        with self._lock:
            self._data[key]=(value,time.monotonic()); self._data.move_to_end(key)
            while len(self._data)>self.maxsize: self._data.popitem(last=False)
        return value
    def delete(self,key):
        with self._lock: return self._data.pop(key,None) is not None
    def clear(self):
        with self._lock: self._data.clear()
    def __len__(self): return len(self._data)

class ScopedMemory:
    def __init__(self): self._scopes={"default":{}}; self._current="default"; self._lock=threading.RLock()
    def use(self,scope):
        with self._lock: self._current=str(scope); self._scopes.setdefault(self._current,{})
        return self._current
    @property
    def scope(self): return self._current
    def set(self,key,value):
        with self._lock: self._scopes[self._current][key]=value
        return value
    def get(self,key,default=None):
        with self._lock: return self._scopes[self._current].get(key,default)
    def delete(self,key):
        with self._lock: return self._scopes[self._current].pop(key,None)
    def clear(self,scope=None):
        with self._lock: self._scopes[scope or self._current].clear()
    def snapshot(self,scope=None):
        with self._lock: return dict(self._scopes[scope or self._current])

class Context:
    def __init__(self): self._values={}; self._lock=threading.RLock()
    def set(self,**values):
        with self._lock: self._values.update(values)
        return self
    def get(self,key,default=None):
        with self._lock: return self._values.get(key,default)
    def remove(self,key):
        with self._lock: return self._values.pop(key,None)
    def clear(self):
        with self._lock: self._values.clear()
    def snapshot(self):
        with self._lock: return dict(self._values)
    @contextmanager
    def push(self,**values):
        with self._lock: old=dict(self._values); self._values.update(values)
        try: yield self
        finally:
            with self._lock: self._values=old

@dataclass
class TaskStep:
    description:str
    status:str="pending"
    result:Any=None
    error:Optional[str]=None
    metadata:dict=field(default_factory=dict)

@dataclass
class TaskPlan:
    goal:str
    steps:list[TaskStep]=field(default_factory=list)
    metadata:dict=field(default_factory=dict)
    def add(self,description,**metadata):
        step=TaskStep(description,metadata=metadata); self.steps.append(step); return step
    def complete(self,step,result=None): step.status="completed"; step.result=result; return step
    def fail(self,step,error): step.status="failed"; step.error=str(error); return step
    def pending(self): return [s for s in self.steps if s.status=="pending"]
    def done(self): return bool(self.steps) and not self.pending()
    def as_dict(self): return {"goal":self.goal,"steps":[s.__dict__.copy() for s in self.steps],"metadata":dict(self.metadata)}

class RetryPolicy:
    def __init__(self,attempts=3,delay=0.2,backoff=2.0): self.attempts=max(1,int(attempts)); self.delay=max(0,float(delay)); self.backoff=max(1,float(backoff))
    def run(self,fn,*args,**kwargs):
        last=None; delay=self.delay
        for i in range(self.attempts):
            try: return fn(*args,**kwargs)
            except Exception as exc:
                last=exc
                if i+1<self.attempts: time.sleep(delay); delay*=self.backoff
        raise last

def fingerprint(value): return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()

config=Config({"execution.enabled":False,"execution.default_timeout":30,"execution.max_output":1_000_000,"generation.streaming":True})
events=EventBus(); cache=LRUCache(); memory=ScopedMemory(); context=Context()
