"""Training helpers."""


def train(model, dataset, epochs=1, **kwargs):
    """Run a model's training method when available."""
    if hasattr(model, "train"):
        return model.train(dataset, epochs=epochs, **kwargs)
    raise TypeError("model must provide a train() method")


def evaluate(model, dataset, **kwargs):
    """Evaluate a model using its evaluate() method."""
    if hasattr(model, "evaluate"):
        return model.evaluate(dataset, **kwargs)
    raise TypeError("model must provide an evaluate() method")
