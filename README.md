# pymodel-ai

**pymodel-ai** is a modular NumPy-based Python framework for building, experimenting with, evaluating, training, and generating from AI/Transformer-style models.

> **Status:** `0.1.0` — Alpha. The public API is available for experimentation. Some advanced training, autograd, quantization, and hardware features are lightweight/experimental rather than production-grade.

## Installation

The **PyPI project name is `pymodel-ai`**.

```bash
pip install pymodel-ai
```

The Python import name intentionally remains **`pymodel`** for compatibility:

```python
import pymodel
```

This distinction is important: **install `pymodel-ai`, import `pymodel`.**

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

Focus profiles are exposed through `SUPPORTED_FOCUSES` and `FOCUS_PROFILES`. Focus utilities include:

- `normalize_focus()`
- `build_focus_config()`
- `focus_description()`

Multiple focuses are supported:

```python
focus=["coding", "reasoning", "math"]
```

Models can also expose and modify focus through `get_focus()`, `set_focus()`, `focus_scores()`, and `focus_info()`.

## Public API

The `pymodel-ai` distribution exposes the Python package `pymodel`. Its public API includes the following functional areas.

### Model and building

`Model`, `build`, `named_parameters`, `parameter_count`, `parameters_info`, `summary`, `model_info`, `model_size`, `count_parameters`, `parameter_shapes`, `set_current_model`, and model-management helpers.

### Tokenization

`tokenizer`, `tokenize`, `tokenize_batch`, `detokenize`, `detokenize_batch`, `vocab`, `build_vocab`, `update_vocab`, `merge_vocab`, `reverse_vocab`, `vocab_size`, `token_id`, `id_token`, `token_exists`, `add_token`, `remove_token`, `special_tokens`, `add_special_token`, `encode`, `encode_batch`, `decode`, `decode_batch`, `normalize_text`, `clean_text`, `split_text`, `split_words`, `split_sentences`, `chunk_text`, `truncate`, `truncate_text`, `lower_text`, `uppercase`, and `pad_sequence`.

### Embeddings

`embedding`, `embeddings`, `positional_embedding`, `position_embedding`, `sinusoidal_embedding`, `rotary_embedding`, and `embedding_similarity`.

### Attention and Transformer

`attention`, `scaled_dot_product_attention`, `self_attention`, `cross_attention`, `multihead_attention`, `multi_query_attention`, `grouped_query_attention`, `causal_attention`, `local_attention`, `global_attention`, `sparse_attention`, `sliding_window_attention`, `block_attention`, `flash_attention`, `rotary_attention`, `alibi_attention`, `attention_mask`, `causal_mask`, `padding_mask`, `transformer_block`, and `transformer`.

### Layers and activations

`linear`, `activation`, `relu`, `gelu`, `sigmoid`, `tanh`, `leaky_relu`, `softplus`, `silu`, `swish`, `mish`, `softmax`, `log_softmax`, `layer_norm`, `batch_norm`, `rms_norm`, `dropout`, `flatten_layer`, `feedforward`, `mlp`, `residual`, and `residual_block`.

### Losses and metrics

`loss`, `cross_entropy`, `binary_cross_entropy`, `mse`, `mae`, `huber_loss`, `kl_divergence`, `contrastive_loss`, `label_smoothing`, `perplexity`, `accuracy`, `f1_score`, and `recall`.

### Tensor and numerical operations

`tensor`, `zeros`, `ones`, `full`, `random`, `randn`, `randint`, `arange`, `reshape`, `flatten`, `squeeze`, `unsqueeze`, `transpose`, `permute`, `repeat`, `expand`, `pad`, `roll`, `cat`, `stack`, `split`, `chunk`, `matmul`, `dot`, `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `clip`, `sqrt`, `exp`, `log`, `abs`, `power`, `norm`, `normalize`, `einsum`, `where`, `masked_fill`, `gather`, and `scatter`.

### Autograd

`gradient`, `compute_gradients`, `numerical_gradient`, `backward`, `requires_grad`, `detach`, `no_grad`, and `zero_grad`.

### Optimizers and scheduling

`Optimizer`, `optimizer`, `sgd`, `adam`, `adamw`, `rmsprop`, `adagrad`, `update_weights`, `step`, `learning_rate`, `weight_decay`, `gradient_clipping`, `lr_scheduler`, `constant_lr`, `linear_decay`, `cosine_decay`, and `warmup`.

### Dataset and training

`dataset`, `load_dataset`, `save_dataset`, `split_dataset`, `shuffle`, `batch`, `map_dataset`, `filter_dataset`, `cache`, `train`, and `evaluate`.

### Generation

`generate`, `sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `stop_at_token`, `get_model`, and `model_run`.

### Serialization and storage

`save`, `load`, `save_model`, `load_model`, `delete_model`, `edit_model`, `save_weights`, `load_weights`, `state_dict`, `load_state_dict`, `serialize`, `deserialize`, `save_config`, `load_config`, `save_checkpoint`, `load_checkpoint`, `checkpoint`, `export_model`, `import_model`, `save_text`, `load_text`, `read_text`, and `write_text`.

### Quantization and fine-tuning

`quantization`, `quantize`, `dequantize`, `int4`, `int8`, `float16`, `bfloat16`, `finetune`, `finetuning`, `parameter_efficient_finetuning`, `lora`, `qlora`, `adapter`, `low_rank`, `prefix_tuning`, `prompt_tuning`, `freeze`, `unfreeze`, `freeze_layers`, `unfreeze_layers`, `trainable`, `trainable_parameters`, `prune`, `sparsify`, and `factorize`.

### System and devices

`device`, `to_device`, `cpu`, `gpu`, `cuda`, `device_count`, `device_info`, `is_gpu_available`, `cpu_info`, `gpu_info`, `cpu_memory`, `gpu_memory`, `memory_info`, `memory_usage`, `free_memory`, `clear_cache`, `ram`, `storage_info`, and `system_info`.

### Reproducibility

`seed`, `set_seed`, and `random_seed` control random initialization and reproducibility.

```python
pymodel.set_seed(42)
```

## Package layout

The project uses a `src` layout. The **distribution name is `pymodel-ai`**, while the **Python package/import name remains `pymodel`**.

```text
pymodel-ai/
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

PyPI publication is automated through GitHub Actions and PyPI Trusted Publishing.

Workflow:

```text
.github/workflows/release.yml
```

The release workflow:

1. Checks out the repository.
2. Sets up Python.
3. Installs `build` and `twine`.
4. Builds source and wheel distributions.
5. Runs `twine check` against the distributions.
6. Publishes the distributions to the `pymodel-ai` PyPI project using Trusted Publishing.

A GitHub Release with the `published` event triggers the workflow. It can also be started manually with `workflow_dispatch`.

## Requirements

- Python `>=3.9`
- NumPy `>=1.24`
- Optional: `psutil>=5.9` for system-related functionality

## Status

**pymodel-ai `0.1.0` Alpha**

The project is intended for experimentation and development. The Alpha API may evolve before a stable release.

## Repository

GitHub: https://github.com/Leila150/pymodel-ai

## License

MIT License. See `LICENSE` for the full license text.
