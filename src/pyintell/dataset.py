"""JSON-first dataset utilities for PyIntell training and knowledge ingestion."""
import json, random

class Dataset:
    def __init__(self, records=None): self.records = list(records or [])
    def add(self, record): self.records.append(record); return record
    def extend(self, records): self.records.extend(records); return self
    def __len__(self): return len(self.records)
    def __getitem__(self, index): return self.records[index]
    def shuffle(self, seed=None):
        rng = random.Random(seed); rng.shuffle(self.records); return self
    def split(self, validation=0.1, seed=None):
        if not 0 <= validation < 1: raise ValueError("validation must be in [0, 1)")
        self.shuffle(seed)
        cut = int(len(self.records) * (1 - validation))
        return Dataset(self.records[:cut]), Dataset(self.records[cut:])
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f: json.dump(self.records, f, ensure_ascii=False, indent=2)
    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f: data = json.load(f)
        if isinstance(data, dict): data = data.get("data", data.get("records", [data]))
        if not isinstance(data, list): raise ValueError("Dataset JSON must contain a list or data/records list")
        return cls(data)

def load_dataset(path): return Dataset.load(path)
def save_dataset(dataset, path): dataset.save(path)
