"""Alpha focus/capability utilities for pyintell.

Focus is intentionally lightweight in 0.1.x. It describes intended model
specialization and provides metadata/priorities for future training systems.
It does not claim that a focus alone makes a model capable.
"""

SUPPORTED_FOCUSES = (
    "intelligence", "natural", "coding", "reasoning", "math", "knowledge",
    "creativity", "conversation", "instruction", "accuracy", "speed", "memory",
    "context", "language", "translation", "summarization", "classification", "roleplay",
)

FOCUS_PROFILES = {
    "intelligence": {"reasoning": 1.0, "knowledge": 0.9, "accuracy": 0.9},
    "natural": {"language": 1.0, "conversation": 0.8, "creativity": 0.6},
    "coding": {"reasoning": 1.0, "instruction": 0.9, "accuracy": 0.9},
    "reasoning": {"reasoning": 1.0, "accuracy": 0.9, "math": 0.8},
    "math": {"math": 1.0, "reasoning": 0.9, "accuracy": 0.9},
    "knowledge": {"knowledge": 1.0, "memory": 0.8, "accuracy": 0.8},
    "creativity": {"creativity": 1.0, "language": 0.8, "natural": 0.7},
    "conversation": {"conversation": 1.0, "natural": 0.9, "instruction": 0.7},
    "instruction": {"instruction": 1.0, "accuracy": 0.9, "conversation": 0.6},
    "accuracy": {"accuracy": 1.0, "reasoning": 0.8, "knowledge": 0.7},
    "speed": {"speed": 1.0, "efficiency": 0.9},
    "memory": {"memory": 1.0, "knowledge": 0.8, "context": 0.7},
    "context": {"context": 1.0, "memory": 0.8, "language": 0.6},
    "language": {"language": 1.0, "natural": 0.8, "translation": 0.7},
    "translation": {"translation": 1.0, "language": 0.9, "accuracy": 0.8},
    "summarization": {"summarization": 1.0, "language": 0.8, "accuracy": 0.7},
    "classification": {"classification": 1.0, "accuracy": 0.9, "instruction": 0.6},
    "roleplay": {"roleplay": 1.0, "conversation": 0.9, "creativity": 0.8},
}


def normalize_focus(focus):
    """Normalize and validate one or more focus names."""
    if isinstance(focus, str):
        values = [focus]
    else:
        try:
            values = list(focus)
        except TypeError as exc:
            raise TypeError("focus must be a string or iterable of strings") from exc
    if not values:
        raise ValueError("focus must contain at least one focus")

    result = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("focus values must be strings")
        value = value.strip().lower()
        if value not in SUPPORTED_FOCUSES:
            raise ValueError(f"unsupported focus: {value}")
        if value not in result:
            result.append(value)
    return result


def build_focus_config(focus):
    """Build a serializable alpha focus configuration."""
    focuses = normalize_focus(focus)
    priorities = {}
    for name in focuses:
        for capability, weight in FOCUS_PROFILES[name].items():
            priorities[capability] = priorities.get(capability, 0.0) + weight

    maximum = max(priorities.values(), default=1.0)
    priorities = {name: round(value / maximum, 4) for name, value in priorities.items()}
    return {"focuses": focuses, "priorities": priorities}


def focus_description(focus):
    """Return a small human-readable description of a focus selection."""
    config = build_focus_config(focus)
    ranked = sorted(config["priorities"].items(), key=lambda item: (-item[1], item[0]))
    return {
        "focuses": config["focuses"],
        "top_capabilities": [name for name, _ in ranked[:5]],
        "priorities": config["priorities"],
    }
