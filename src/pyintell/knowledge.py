"""Lightweight local knowledge and long-term memory store."""
from dataclasses import dataclass, asdict
import json, os, re, time

@dataclass
class Memory:
    text: str
    source: str = "user"
    timestamp: float = 0.0
    tags: tuple = ()

class KnowledgeStore:
    def __init__(self, path=None):
        self.path = path
        self.items = []
        if path and os.path.exists(path): self.load(path)

    def add(self, text, source="user", tags=()):
        if not text or not str(text).strip(): raise ValueError("text must be non-empty")
        item = Memory(str(text), str(source), time.time(), tuple(tags))
        self.items.append(item)
        return item

    def add_many(self, texts, source="dataset"):
        return [self.add(x, source) for x in texts]

    def search(self, query, top_k=5):
        terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for item in self.items:
            words = set(re.findall(r"\w+", item.text.lower()))
            score = len(terms & words) / max(1, len(terms))
            if score: scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**asdict(item), "score": score} for score, item in scored[:top_k]]

    def save(self, path=None):
        target = path or self.path
        if not target: raise ValueError("No storage path supplied")
        with open(target, "w", encoding="utf-8") as f:
            json.dump([asdict(x) | {"tags": list(x.tags)} for x in self.items], f, ensure_ascii=False, indent=2)
        self.path = target

    def load(self, path=None):
        target = path or self.path
        with open(target, encoding="utf-8") as f: data = json.load(f)
        self.items = [Memory(x["text"], x.get("source", "unknown"), x.get("timestamp", 0), tuple(x.get("tags", []))) for x in data]
        self.path = target
        return self

MemoryStore = KnowledgeStore
