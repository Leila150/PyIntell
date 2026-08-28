"""Generation and sampling utilities."""

import numpy as np

from .serialization import get_model


def sample(logits, temperature=1.0, top_k=None, rng=None):
    """Sample one token ID from a logits vector."""
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if top_k is not None:
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        top_k = min(top_k, scaled_size := scaled.size) if False else min(top_k, logits.size)
        indices = np.argpartition(logits, -top_k)[-top_k:]
        filtered = np.full_like(logits, -np.inf)
        filtered[indices] = logits[indices]
        logits = filtered
    scaled = logits / temperature
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    generator = rng or np.random.default_rng()
    return int(generator.choice(len(probabilities), p=probabilities))


def generate(model_name, prompt, max_tokens=50, temperature=1.0, top_k=None, **kwargs):
    """Generate a response using a saved model name.

    ``model_name`` must refer to a model previously saved with
    :func:`pymodel.save_model`.
    """
    model = get_model(model_name)
    if not hasattr(model, "generate"):
        raise TypeError("saved object is not a valid pymodel model")
    return model.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        **kwargs,
    )


def model_run(model_name, max_tokens=50, temperature=1.0, top_k=None, **kwargs):
    """Run an interactive terminal chat session with a saved model.

    Enter ``exit``, ``quit``, or ``/exit`` to stop the session.
    """
    # Validate/load once before entering the loop so a bad model name fails
    # immediately instead of after the user has started typing.
    get_model(model_name)
    print(f"pymodel model '{model_name}' is running.")
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
        response = generate(
            model_name,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            **kwargs,
        )
        print(f"AI: {response}")

    return None
