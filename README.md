# pymodel

**pymodel** is a modular Python framework for building, experimenting with, and eventually training AI models. It exposes the building blocks of modern language models through a small Python API, from tokenization and embeddings to attention, Transformers, losses, optimizers, generation, and system-resource inspection.

> **Status:** `0.1.0` — Alpha. The architecture and numerical primitives are available now. Full reverse-mode automatic differentiation, large-scale optimizer training, and production-grade inference are still under development.

## Install

```bash
pip install pymodel
```

Or from source:

```bash
pip install .
```

## Main builder

The central API is:

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

### Arguments

| Argument | Meaning |
|---|---|
| `vocab` | Token → integer ID dictionary. |
| `reverse_vocab` | Integer ID → token dictionary. |
| `dataset` | Data associated with the model. |
| `parameters` | Requested approximate parameter count. |
| `focus` | One focus or a list of model goals. |
| `dtype` | Optional numerical type; defaults to `float32`. |
| `settings` | Optional configuration dictionary. |

`build()` is intentionally an orchestrator: it uses pymodel's existing embedding, positional embedding, Transformer, linear, generation, and system functions rather than creating separate hidden implementations for each focus.

## Focus

Supported focus values:

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

Multiple values are allowed:

```python
focus = ["intelligence", "natural", "reasoning", "roleplay"]
```

In `0.1.0`, focus is validated and stored as model metadata. It is designed so future training/data systems can use the same API to influence objectives and data weighting without introducing a different builder for every specialization.

## Parameters and resource protection

`parameters` is the requested model size, not a byte count. A model with 100 million parameters is not automatically a 100 MB model.

For raw weights:

| dtype | Bytes / parameter |
|---|---:|
| `float64` | 8 |
| `float32` | 4 |
| `float16` | 2 |
| `bfloat16` | 2 |
| `int8` | 1 |
| `int4` | 0.5 |

Training usually needs substantially more memory than the raw weights because of gradients, optimizer state, activations, and batches.

Before constructing a model, `build()` checks available RAM and free storage through pymodel's system functions and raises an error when the requested model is clearly too large for the machine.

Available system APIs include:

```python
pymodel.system_info()
pymodel.ram()
pymodel.cpu_info()
pymodel.gpu_info()
pymodel.storage_info()
pymodel.memory_info()
pymodel.device_info()
pymodel.is_gpu_available()
```

GPU detection is optional and does not add a GPU framework as a package dependency.

## Settings

`settings` is optional and is always treated as a dictionary. It keeps the main function signature small while allowing more configuration.

Current architecture/training settings include:

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

Additional keys are preserved on the model so the API can grow without repeatedly changing `build()`.

## Dtypes

Supported names are:

```text
float64
float32
float16
bfloat16
int8
int4
```

`bfloat16` and `int4` are represented with compatible NumPy storage in this alpha release while the requested dtype remains part of the model configuration. They should not be interpreted as a complete production quantization implementation yet.

## Tokenization and vocabulary

```python
pymodel.tokenizer()
pymodel.tokenize()
pymodel.detokenize()
pymodel.vocab()
pymodel.build_vocab()
pymodel.update_vocab()
pymodel.merge_vocab()
pymodel.reverse_vocab()
pymodel.encode()
pymodel.decode()
pymodel.encode_batch()
pymodel.decode_batch()
pymodel.add_token()
pymodel.remove_token()
pymodel.token_id()
pymodel.id_token()
pymodel.token_exists()
pymodel.special_tokens()
pymodel.add_special_token()
pymodel.vocab_size()
pymodel.normalize_text()
pymodel.clean_text()
pymodel.truncate()
pymodel.pad_sequence()
```

## Embeddings

```python
pymodel.embedding()
pymodel.positional_embedding()
pymodel.position_embedding()
pymodel.sinusoidal_embedding()
pymodel.rotary_embedding()
pymodel.embedding_similarity()
```

## Attention

```python
pymodel.attention()
pymodel.scaled_dot_product_attention()
pymodel.self_attention()
pymodel.cross_attention()
pymodel.multihead_attention()
pymodel.multi_query_attention()
pymodel.grouped_query_attention()
pymodel.causal_attention()
pymodel.local_attention()
pymodel.global_attention()
pymodel.sparse_attention()
pymodel.sliding_window_attention()
pymodel.block_attention()
pymodel.flash_attention()
pymodel.rotary_attention()
pymodel.alibi_attention()
pymodel.attention_mask()
pymodel.causal_mask()
pymodel.padding_mask()
```

## Neural-network layers

```python
pymodel.linear()
pymodel.activation()
pymodel.relu()
pymodel.gelu()
pymodel.sigmoid()
pymodel.tanh()
pymodel.leaky_relu()
pymodel.softplus()
pymodel.silu()
pymodel.swish()
pymodel.mish()
pymodel.softmax()
pymodel.log_softmax()
pymodel.layer_norm()
pymodel.batch_norm()
pymodel.rms_norm()
pymodel.dropout()
pymodel.flatten_layer()
pymodel.feedforward()
pymodel.mlp()
pymodel.residual()
pymodel.residual_block()
```

## Transformer

```python
pymodel.transformer_block()
pymodel.transformer()
```

The current forward path is:

```text
Token IDs
  ↓
Token embedding + positional embedding
  ↓
Causal multi-head self-attention
  ↓
Residual + normalization
  ↓
Feed-forward network
  ↓
Residual + normalization
  ↓
Repeat blocks
  ↓
Language-model logits
```

## Tensor and math utilities

```python
pymodel.tensor()
pymodel.zeros()
pymodel.ones()
pymodel.full()
pymodel.random()
pymodel.randn()
pymodel.randint()
pymodel.arange()
pymodel.reshape()
pymodel.flatten()
pymodel.squeeze()
pymodel.unsqueeze()
pymodel.transpose()
pymodel.permute()
pymodel.matmul()
pymodel.dot()
pymodel.sum()
pymodel.mean()
pymodel.max()
pymodel.min()
pymodel.argmax()
pymodel.argmin()
pymodel.clip()
pymodel.sqrt()
pymodel.exp()
pymodel.log()
pymodel.abs()
pymodel.power()
pymodel.cat()
pymodel.stack()
pymodel.split()
pymodel.chunk()
pymodel.repeat()
pymodel.expand()
pymodel.pad()
pymodel.roll()
pymodel.gather()
pymodel.scatter()
pymodel.where()
pymodel.masked_fill()
pymodel.einsum()
pymodel.norm()
pymodel.normalize()
```

## Losses

```python
pymodel.loss()
pymodel.cross_entropy()
pymodel.binary_cross_entropy()
pymodel.mse()
pymodel.mae()
pymodel.huber_loss()
pymodel.kl_divergence()
pymodel.contrastive_loss()
pymodel.label_smoothing()
```

## Autograd helpers

The alpha release includes numerical gradient helpers for experimentation:

```python
pymodel.gradient()
pymodel.compute_gradients()
pymodel.numerical_gradient()
pymodel.backward()
pymodel.requires_grad()
pymodel.detach()
pymodel.no_grad()
pymodel.zero_grad()
```

These are not yet a replacement for a production reverse-mode autograd engine.

## Optimizers

```python
pymodel.optimizer()
pymodel.sgd()
pymodel.adam()
pymodel.adamw()
pymodel.rmsprop()
pymodel.adagrad()
pymodel.adadelta()
pymodel.update_weights()
pymodel.step()
pymodel.learning_rate()
pymodel.zero_grad()
```

## Datasets and training utilities

```python
pymodel.dataset()
pymodel.load_dataset()
pymodel.save_dataset()
pymodel.split_dataset()
pymodel.shuffle()
pymodel.batch()
pymodel.map_dataset()
pymodel.filter_dataset()
pymodel.cache()
pymodel.train()
pymodel.evaluate()
```

The high-level model training loop is intentionally not advertised as production-ready in `0.1.0`; the framework is still building its complete autograd and trainable-parameter system.

## Generation and sampling

```python
pymodel.generate()
pymodel.sample()
pymodel.temperature()
pymodel.top_k()
pymodel.top_p()
pymodel.repetition_penalty()
pymodel.frequency_penalty()
pymodel.presence_penalty()
pymodel.stop_at_token()
```

A constructed model also exposes:

```python
model.forward(token_ids)
model.logits(token_ids)
model.generate(prompt)
model.summary()
```

## Model inspection and utilities

```python
pymodel.model_info()
pymodel.summary()
pymodel.count_parameters()
pymodel.trainable_parameters()
pymodel.parameter_shapes()
pymodel.model_size()
pymodel.memory_usage()
pymodel.save_config()
pymodel.load_config()
```

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
├── autograd.py
├── utilities.py
├── model.py
└── builder.py
```

There are intentionally **no test or example directories** in the repository, as requested. The README is the primary user-facing documentation.

## Packaging and PyPI

The project uses modern `pyproject.toml` metadata and setuptools. The standard Python packaging flow builds both a source distribution and a wheel. The PyPA documentation recommends `python -m build` for building distributions and `twine check` for validating the generated metadata/README before upload. citeturn0search1turn0search4

Build locally:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

For manual upload:

```bash
python -m twine upload dist/*
```

For automated releases, the repository includes a GitHub Actions workflow using PyPI Trusted Publishing. PyPA recommends Trusted Publishing for supported CI/CD platforms because it avoids storing a long-lived PyPI API token in the workflow. citeturn0search2

Before the first automated release, configure a PyPI Trusted Publisher for this GitHub repository/workflow. Then publish a GitHub Release and the workflow will build, validate, and publish the distributions.

## Development roadmap

The next major milestones are:

1. Trainable parameter objects.
2. Full reverse-mode automatic differentiation.
3. Real optimizer state and update loops.
4. Complete trainable Transformer output head.
5. Efficient dataset batching and streaming.
6. Better checkpointing and serialization.
7. Mixed precision and real quantization.
8. Hardware-aware VRAM and memory estimation.
9. Faster attention and inference.
10. Larger-model performance work.

## License

MIT License. See `LICENSE`.
