"""Generation and sampling utilities."""

import numpy as np

from .serialization import get_model


def sample(logits, temperature=1.0, top_k=None, rng=None):
    """Sample one token ID from a logits vector."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits must be a non-empty one-dimensional array")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    scaled = logits / temperature
    if top_k is not None:
        if isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")
        top_k = int(top_k)
        if top_k <= 0: raise ValueError("top_k must be greater than zero")
        top_k = min(top_k, scaled.size)
        indices = np.argpartition(scaled, -top_k)[-top_k:]
        filtered = np.full_like(scaled, -np.inf)
        filtered[indices] = scaled[indices]
        scaled = filtered
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("logits produced invalid sampling probabilities")
    probabilities /= total
    generator = rng or np.random.default_rng()
    return int(generator.choice(len(probabilities), p=probabilities))


def generate(model_name, prompt, max_tokens=50, temperature=1.0, top_k=None, **kwargs):
    """Generate a response using a saved model name."""
    model = get_model(model_name)
    if not hasattr(model, "generate"):
        raise TypeError("saved object is not a valid pymodel model")
    return model.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_k=top_k, **kwargs)


def model_run(model_name, max_tokens=50, temperature=1.0, top_k=None, **kwargs):
    """Run an interactive terminal chat session with a saved model.

    Invalid prompts or generation errors are reported without terminating the
    entire chat session. Enter ``exit``, ``quit``, ``/exit``, or ``/quit`` to
    stop the session.
    """
    # Validate/load once before opening the interactive session.
    get_model(model_name)
    print(f"pyintell model '{model_name}' is running.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            prompt = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in {"exit", "quit", "/exit", "/quit"}:
            break
        if not prompt.strip():
            continue
        try:
            response = generate(model_name, prompt, max_tokens=max_tokens,
                                temperature=temperature, top_k=top_k, **kwargs)
            print(f"AI: {response}")
        except (TypeError, ValueError, RuntimeError, FileNotFoundError) as error:
            print(f"AI error: {error}")

    return None
