# pymodel-ai

**pymodel-ai** is a modular NumPy-based Python framework for building, experimenting with, evaluating, training, and generating from AI/Transformer-style models.

> **Status:** `0.1.0` — Alpha. The public API is available for experimentation. Some advanced training, autograd, quantization, and hardware features are lightweight/experimental rather than production-grade.

## Package name vs. import name

The PyPI project is **`pymodel-ai`**, while the Python import remains **`pymodel`**.

```bash
pip install pymodel-ai
```

```python
import pymodel
```

## Quick start

```python
import pymodel

model = pymodel.build(
    vocab={"hello": 0, "world": 1},
    reverse_vocab={0: "hello", 1: "world"},
    dataset=[[0, 1]],
    parameters=100_000,
    focus=["coding", "reasoning", "math"],
)

print(model.summary())
print(model.generate("hello"))
```

## Main builder

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

`build()` constructs a `Model` from a vocabulary, reverse vocabulary, dataset, requested parameter count, focus, numerical dtype, and optional settings.

## Focus system

Supported focus profiles are exposed through `SUPPORTED_FOCUSES` and `FOCUS_PROFILES`. Focus utilities include:

- `normalize_focus()`
- `build_focus_config()`
- `focus_description()`

Multiple focuses are supported:

```python
focus=["coding", "reasoning", "math"]
```

## Model API

The core model API includes:

- `Model`
- `build()`
- `forward()`
- `logits()`
- `train()`
- `evaluate()`
- `generate()`
- `summary()`
- `named_parameters()`
- `parameter_count()`
- `parameters_info()`
- `get_focus()`
- `set_focus()`
- `focus_scores()`
- `focus_info()`

Models support configurable parameter counts, layers, heads, embedding size, hidden size, context length, dtype, platform, optimizer, learning rate, training steps, and focus settings.

## Tokenization

The public tokenization API includes:

`tokenizer`, `tokenize`, `tokenize_batch`, `detokenize`, `detokenize_batch`, `vocab`, `build_vocab`, `update_vocab`, `merge_vocab`, `reverse_vocab`, `vocab_size`, `token_id`, `id_token`, `token_exists`, `add_token`, `remove_token`, `special_tokens`, `add_special_token`, `encode`, `encode_batch`, `decode`, `decode_batch`, `normalize_text`, `clean_text`, `split_text`, `truncate`, and `pad_sequence`.

Example:

```python
vocabulary = pymodel.build_vocab(["hello", "world"])
reverse = pymodel.reverse_vocab(vocabulary)
ids = pymodel.encode("hello world", vocabulary)
text = pymodel.decode(ids, reverse)
```

## Generation

Generation helpers include:

- `generate()`
- `sample()`
- `temperature()`
- `top_k()`
- `top_p()`
- `repetition_penalty()`
- `frequency_penalty()`
- `presence_penalty()`
- `stop_at_token()`
- `get_model()`
- `model_run()`

`model_run()` provides an interactive model session.

## Embeddings and Transformer operations

The package provides embedding helpers such as `embedding()`, `embeddings()`, `positional_embedding()`, `position_embedding()`, `sinusoidal_embedding()`, `rotary_embedding()`, and `embedding_similarity()`.

Attention and Transformer primitives include `attention()`, `scaled_dot_product_attention()`, `self_attention()`, `cross_attention()`, `multihead_attention()`, `multi_query_attention()`, `grouped_query_attention()`, `causal_attention()`, `local_attention()`, `global_attention()`, `sparse_attention()`, `sliding_window_attention()`, `block_attention()`, `flash_attention()`, `rotary_attention()`, `alibi_attention()`, `attention_mask()`, `causal_mask()`, and `padding_mask()`.

Layer primitives include `linear()`, `activation()`, `relu()`, `gelu()`, `sigmoid()`, `tanh()`, `leaky_relu()`, `softplus()`, `silu()`, `swish()`, `mish()`, `softmax()`, `log_softmax()`, `layer_norm()`, `batch_norm()`, `rms_norm()`, `dropout()`, `flatten_layer()`, `feedforward()`, `mlp()`, `residual()`, `residual_block()`, `transformer_block()`, and `transformer()`.

## Losses and metrics

Available loss and evaluation helpers include `loss()`, `cross_entropy()`, `binary_cross_entropy()`, `mse()`, `mae()`, `huber_loss()`, `kl_divergence()`, `contrastive_loss()`, `label_smoothing()`, `perplexity()`, `accuracy()`, `f1_score()`, and `recall()`.

## Tensor and numerical operations

pymodel exposes NumPy-oriented tensor and numerical helpers including:

`tensor`, `zeros`, `ones`, `full`, `random`, `randn`, `randint`, `arange`, `reshape`, `flatten`, `squeeze`, `unsqueeze`, `transpose`, `permute`, `repeat`, `expand`, `pad`, `roll`, `cat`, `stack`, `split`, `chunk`, `matmul`, `dot`, `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `clip`, `sqrt`, `exp`, `log`, `abs`, `power`, `norm`, `normalize`, `einsum`, `where`, `masked_fill`, `gather`, and `scatter`.

## Autograd

Experimental gradient/autograd helpers include `gradient()`, `compute_gradients()`, `numerical_gradient()`, `backward()`, `requires_grad()`, `detach()`, `no_grad()`, and `zero_grad()`.

## Optimizers and scheduling

Optimizer helpers include `Optimizer`, `optimizer()`, `sgd()`, `adam()`, `adamw()`, `rmsprop()`, `adagrad()`, `update_weights()`, `step()`, `learning_rate()`, `weight_decay()`, `gradient_clipping()`, `clip()`, and scheduler utilities such as `lr_scheduler()`, `constant_lr()`, `linear_decay()`, `cosine_decay()`, and `warmup()`.

## Dataset and training

Dataset and training helpers include `dataset()`, `load_dataset()`, `save_dataset()`, `split_dataset()`, `shuffle()`, `batch()`, `map_dataset()`, `filter_dataset()`, `cache()`, `train()`, and `evaluate()`.

## Serialization and model storage

Model and data storage functionality includes `save()`, `load()`, `save_model()`, `load_model()`, `delete_model()`, `get_model()`, `edit_model()`, `model_info()`, `model_size()`, `count_parameters()`, `parameter_shapes()`, `save_weights()`, `load_weights()`, `save_state_dict()`/`load_state_dict()` where exposed, `serialize()`, `deserialize()`, `save_config()`, `load_config()`, `save_checkpoint()`, `load_checkpoint()`, `checkpoint()`, `export_model()`, `import_model()`, `save_text()`, `load_text()`, `read_text()`, and `write_text()`.

## Quantization and parameter-efficient fine-tuning

The public API includes quantization helpers such as `quantization()`, `quantize()`, `dequantize()`, `int4`, `int8`, `float16`, and `bfloat16`.

Parameter-efficient fine-tuning helpers include `finetune()`, `finetuning`, `parameter_efficient_finetuning`, `lora()`, `qlora()`, `adapter()`, `low_rank()`, `prefix_tuning()`, `prompt_tuning()`, `freeze()`, `unfreeze()`, `freeze_layers()`, `unfreeze_layers()`, `trainable()`, `trainable_parameters()`, `prune()`, `sparsify()`, and `factorize()`.

## Devices and system information

Device/system helpers include `device()`, `to_device()`, `cpu()`, `gpu()`, `cuda()`, `device_count()`, `device_info()`, `is_gpu_available()`, `cpu_info()`, `gpu_info()`, `cpu_memory()`, `gpu_memory()`, `memory_info()`, `memory_usage()`, `free_memory()`, `clear_cache()`, `ram()`, `storage_info()`, and `system_info()`.

## Reproducibility

Use `seed()`, `set_seed()`, and `random_seed()` to control random initialization and reproducible model construction.

```python
pymodel.set_seed(42)
```

## Package layout

The project uses a `src` layout so the import package remains `pymodel` while the distributable PyPI project is named `pymodel-ai`.

```text
pymodel/
├── src/
│   └── pymodel/
├── tests/
├── .github/
│   └── workflows/
│       └── release.yml
├── pyproject.toml
├── README.md
└── LICENSE
```

## PyPI publishing

Releases are built and published through GitHub Actions using PyPI Trusted Publishing. The workflow is:

```text
.github/workflows/release.yml
```

Create a GitHub Release to trigger publication. The workflow builds both source and wheel distributions, validates them with Twine, and publishes them to the **pymodel-ai** PyPI project.

The repository currently remains `Leila150/pymodel`; the PyPI project/package name is `pymodel-ai`, and the Python import name intentionally remains `pymodel`.

## Requirements

- Python `>=3.9`
- NumPy `>=1.24`
- Optional: `psutil>=5.9` for system-related functionality

## Status

**pymodel-ai 0.1.0 Alpha**

The project is intended for experimentation and development. The Alpha API may evolve before a stable release.

## License

MIT License. See `LICENSE` for the full license text.

## Repository

urlGitHub repositoryhttps://github.com/Leila150/pymodel
