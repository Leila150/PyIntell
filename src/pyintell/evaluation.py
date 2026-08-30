"""Simple, extensible evaluation framework for PyIntell models."""
from dataclasses import dataclass, asdict
import time

@dataclass
class EvaluationResult:
    name: str
    score: float
    passed: bool
    details: dict
    duration: float

class Evaluator:
    def __init__(self, threshold=0.5): self.threshold = threshold
    def evaluate(self, model, cases, metric=None):
        metric = metric or self.exact_match
        results=[]; start_all=time.perf_counter()
        for case in cases:
            prompt = case.get("prompt", "") if isinstance(case, dict) else case[0]
            expected = case.get("expected", "") if isinstance(case, dict) else case[1]
            start=time.perf_counter()
            try:
                actual = model.generate(prompt) if hasattr(model, "generate") else model(prompt)
                score = float(metric(actual, expected))
                results.append(EvaluationResult(str(prompt)[:80], score, score >= self.threshold, {"expected": expected, "actual": actual}, time.perf_counter()-start))
            except Exception as exc:
                results.append(EvaluationResult(str(prompt)[:80], 0.0, False, {"error": str(exc)}, time.perf_counter()-start))
        return {"score": sum(r.score for r in results)/max(1,len(results)), "passed": all(r.passed for r in results), "duration": time.perf_counter()-start_all, "results": [asdict(r) for r in results]}
    @staticmethod
    def exact_match(actual, expected): return 1.0 if str(actual).strip() == str(expected).strip() else 0.0
    @staticmethod
    def contains(actual, expected): return 1.0 if str(expected).lower() in str(actual).lower() else 0.0

def evaluate(model, cases, metric=None, threshold=0.5):
    return Evaluator(threshold).evaluate(model, cases, metric)
