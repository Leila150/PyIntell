# pymodel

**pymodel** is a modular Python framework for building AI models. It is designed to expose the important building blocks of language models through a simple Python API instead of hiding everything behind one large abstraction.

> **Status:** early development (`0.1.0`). The current release provides the core architecture and numerical primitives. Full automatic-differentiation training, a complete trainable language-model output head, and production inference are not implemented yet.

## Goals

pymodel aims to provide a progression from basic AI components to complete custom models:

```text
Text
 ↓
Tokenizer
 ↓
Vocabulary / token IDs
 ↓
Embeddings
 ↓
Positional information
 ↓
Attention
 ↓
Transformer blocks
 ↓
Loss / optimization
 ↓
Training
 ↓
Generation
```

The package is intentionally modular so individual components can be used directly and the high-level builder can compose them.

## Installation

From the repository:

```bash
pip install .
```

The package currently uses NumPy for numerical operations and psutil for system-resource inspection.

## Quick API

### Tokenization

```python
import pymodel

text = "hello world hello"
tokens = pymodel.tokenizer(text)
vocabulary = pymodel.vocab(tokens)
reverse = pymodel.reverse_vocab(vocabulary)

print(tokens)
print(vocabulary)
print(reverse)
```

### Build a model

The main high-level function is:

```python
pymodel.build(
    vocab,
    reverse_vocab,
    dataset,
    parameters,
    focus,
    dtype=None,
    settings=None,
)
```

Arguments:

- `vocab` — token-to-ID dictionary.
- `reverse_vocab` — ID-to-token dictionary.
- `dataset` — training data supplied to the model.
- `parameters` — requested approximate parameter count.
- `focus` — one focus or a list of focuses describing the intended model specialization.
- `dtype` — optional numerical type. Defaults to `float32`.
- `settings` — optional dictionary for additional architecture and training settings.

Example:

```python
model = pymodel.build(
    vocabulary,
    reverse,
    dataset,
    10_000_000,
    ["natural", "conversation", "roleplay"],
    dtype="float32",
    settings={
        "context_length": 512,
        "batch_size": 4,
        "learning_rate": 0.0003,
        "epochs": 3,
    },
)
```

`build()` composes pymodel's existing primitives rather than requiring a separate custom builder for every specialization.

## Focus

Focus describes the intended capability of a model. Supported values currently include:

```text
intelligence
natural
coding
reasoning
math
knowledge
creativity
conversation
instruction
accuracy
speed
memory
context
language
translation
summarization
classification
roleplay
```

Multiple focuses can be supplied:

```python
focus = ["natural", "reasoning", "roleplay"]
```

Focus is currently stored as model metadata. Future training systems can use it to influence data weighting, objectives, and architecture choices without creating separate model-building APIs.

## Parameter count

`parameters` expresses the desired approximate model size:

```python
model = pymodel.build(vocabulary, reverse, dataset, 100_000_000, "natural")
```

The builder estimates a Transformer architecture from the requested size and vocabulary size. The resulting architecture is exposed through `model.summary()`.

Parameter count is not the same thing as memory usage. Training normally requires additional memory for gradients, optimizer state, activations, and batches.

## Data types

`dtype` is optional. Supported values are:

```text
float64
float32
float16
bfloat16
int8
int4
```

Lower-precision formats can reduce weight storage, but they do not automatically make every part of training compatible with that precision. In the current early implementation, `bfloat16` and `int4` are represented using compatible NumPy storage while the requested type remains part of the model configuration.

Approximate raw weight storage is:

| Type | Bytes per parameter |
|---|---:|
| `float64` | 8 |
| `float32` | 4 |
| `float16` | 2 |
| `bfloat16` | 2 |
| `int8` | 1 |
| `int4` | 0.5 |

These are weight-storage estimates only, not full training-memory requirements.

## Settings

`settings` is an optional dictionary. It keeps the public `build()` signature compact while allowing additional configuration.

Supported settings currently include:

```python
{
    "layers": 6,
    "heads": 8,
    "embedding_size": 512,
    "hidden_size": 2048,
    "context_length": 512,
    "batch_size": 4,
    "learning_rate": 0.0003,
    "epochs": 3,
}
```

When architecture values are omitted, `build()` derives reasonable defaults from the requested parameter count and vocabulary size.

## Resource protection

Before constructing a requested model, `build()` checks available system RAM and free storage using pymodel's resource functions:

```python
pymodel.ram()
pymodel.storage_info()
pymodel.memory_info()
pymodel.system_info()
```

The builder estimates weight storage and a conservative RAM requirement. If the request is clearly larger than the available resources, it raises an error instead of attempting the allocation.

This is a protection against accidentally requesting a model that is too large for the current machine. Hardware-specific VRAM detection and more precise training-memory estimates are planned for future versions.

## Core modules

### `tokenization`

```python
pymodel.tokenizer()
pymodel.vocab()
pymodel.reverse_vocab()
pymodel.encode()
pymodel.decode()
```

### `embeddings`

```python
pymodel.embedding()
pymodel.positional_embedding()
```

### `attention`

```python
pymodel.attention()
pymodel.self_attention()
pymodel.multihead_attention()
pymodel.causal_attention()
```

### `layers`

```python
pymodel.linear()
pymodel.activation()
pymodel.relu()
pymodel.gelu()
pymodel.layer_norm()
pymodel.dropout()
pymodel.feedforward()
```

### `transformer`

```python
pymodel.transformer_block()
pymodel.transformer()
```

### `loss`

```python
pymodel.loss()
pymodel.cross_entropy()
pymodel.mse()
```

### `optim`

```python
pymodel.optimizer()
pymodel.sgd()
pymodel.adam()
pymodel.adamw()
```

### `training` and `generation`

```python
pymodel.train()
pymodel.evaluate()
pymodel.generate()
pymodel.sample()
```

These APIs are present as the framework surface, while complete automatic-differentiation training and a finished generative language-model head are still under development.

## Transformer flow

The current architecture follows the basic Transformer structure:

```text
Token IDs
   ↓
Token embedding + positional embedding
   ↓
Causal multi-head self-attention
   ↓
Residual connection + normalization
   ↓
Feed-forward network
   ↓
Residual connection + normalization
   ↓
Repeat Transformer blocks
```

The low-level functions are intentionally public so users can experiment with individual pieces before using `build()`.

## Project structure

```text
pymodel/
├── __init__.py
├── tokenization.py
├── embeddings.py
├── attention.py
├── layers.py
├── transformer.py
├── loss.py
├── optim.py
├── training.py
├── generation.py
├── system.py
├── model.py
└── builder.py
```

There are deliberately no example or test directories in the initial repository layout. The README serves as the primary project documentation while the implementation is being built.

## Development direction

The intended roadmap is:

1. Robust tensor abstraction.
2. Automatic differentiation and backpropagation.
3. Trainable parameters and optimizer updates.
4. Complete Transformer language-model output head.
5. Real training loops and dataset batching.
6. Efficient sampling and text generation.
7. Better memory estimation and hardware/VRAM detection.
8. Mixed precision and quantization.
9. Checkpointing and model serialization.
10. Performance improvements for larger models.

## License

pymodel is released under the MIT License. See `LICENSE` for details.
