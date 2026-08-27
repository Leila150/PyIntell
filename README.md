# pymodel

**pymodel** is a modular Python framework for building, experimenting with, and eventually training AI models. It exposes tokenization, embeddings, attention, Transformer blocks, losses, optimizers, generation, numerical utilities, serialization, quantization, fine-tuning helpers, and system-resource inspection through one package.

> **Status:** `0.1.0` — Alpha. Core architecture and numerical primitives are available. Full reverse-mode automatic differentiation, large-scale optimizer training, and production-grade inference are still under development.

## Install

```bash
pip install pymodel
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

- `vocab`: token → integer ID dictionary.
- `reverse_vocab`: integer ID → token dictionary.
- `dataset`: model/training data.
- `parameters`: requested approximate parameter count.
- `focus`: one focus or a list of focuses.
- `dtype`: optional numerical type; defaults to `float32`.
- `settings`: optional dictionary for extra configuration.

`build()` is an orchestrator and uses pymodel's existing embedding, positional embedding, Transformer, linear, generation, and system functions.

## Focus

```text
intelligence, natural, coding, reasoning, math, knowledge,
creativity, conversation, instruction, accuracy, speed, memory,
context, language, translation, summarization, classification, roleplay
```

Multiple focuses are supported. In `0.1.0`, focus is validated and stored as model metadata so future training/data systems can use the same API for specialization.

## Parameters, dtype, RAM, and storage

`parameters` is model size, not byte size. Raw weight storage depends on `dtype`:

| dtype | Bytes / parameter |
|---|---:|
| `float64` | 8 |
| `float32` | 4 |
| `float16` | 2 |
| `bfloat16` | 2 |
| `int8` | 1 |
| `int4` | 0.5 |

Training normally requires additional memory for gradients, optimizer state, activations, and batches. `build()` checks available RAM and free storage before construction and refuses requests that are clearly too large.

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

GPU detection is optional and does not add a GPU framework as a required dependency.

## Settings

`settings` is optional and is a dictionary. Architecture values include:

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

Extra settings are preserved on the model. `build()` fills missing architecture values automatically from the requested parameter count and vocabulary size.

## Public API

### Tokenization

```text
tokenizer() tokenize() detokenize() vocab() build_vocab() update_vocab()
merge_vocab() reverse_vocab() encode() decode() encode_batch() decode_batch()
add_token() remove_token() token_id() id_token() token_exists()
special_tokens() add_special_token() vocab_size() normalize_text() clean_text()
truncate() pad_sequence()
```

### Embeddings

```text
embedding() positional_embedding() position_embedding()
sinusoidal_embedding() rotary_embedding() embedding_similarity()
```

### Attention

```text
attention() scaled_dot_product_attention() self_attention() cross_attention()
multihead_attention() multi_query_attention() grouped_query_attention()
causal_attention() local_attention() global_attention() sparse_attention()
sliding_window_attention() block_attention() flash_attention()
rotary_attention() alibi_attention() attention_mask() causal_mask() padding_mask()
```

### Layers and activations

```text
linear() activation() relu() gelu() sigmoid() tanh() leaky_relu()
softplus() silu() swish() mish() softmax() log_softmax() layer_norm()
batch_norm() rms_norm() dropout() flatten_layer() feedforward() mlp()
residual() residual_block()
```

### Transformer

```text
transformer_block() transformer()
```

Current forward flow:

```text
Token IDs → token embedding + positional embedding
→ causal multi-head attention → residual/normalization
→ feed-forward → residual/normalization → repeated blocks → logits
```

### Tensor/math utilities

```text
tensor() zeros() ones() full() random() randn() randint() arange()
reshape() flatten() squeeze() unsqueeze() transpose() permute()
matmul() dot() sum() mean() max() min() argmax() argmin() clip()
sqrt() exp() log() abs() power() cat() stack() split() chunk()
repeat() expand() pad() roll() gather() scatter() where() masked_fill()
einsum() norm() normalize()
```

### Losses

```text
loss() cross_entropy() binary_cross_entropy() mse() mae() huber_loss()
kl_divergence() contrastive_loss() label_smoothing()
```

### Autograd helpers

```text
gradient() compute_gradients() numerical_gradient() backward()
requires_grad() detach() no_grad() zero_grad()
```

These are numerical/experimental helpers in `0.1.0`, not a production reverse-mode engine.

### Optimizers

```text
optimizer() sgd() adam() adamw() rmsprop() adagrad() adadelta()
update_weights() step() learning_rate() zero_grad()
```

### Dataset/training helpers

```text
dataset() load_dataset() save_dataset() split_dataset() shuffle() batch()
map_dataset() filter_dataset() cache() train() evaluate()
```

### Generation and sampling

```text
generate() sample() temperature() top_k() top_p()
repetition_penalty() frequency_penalty() presence_penalty() stop_at_token()
```

A model exposes:

```python
model.forward(token_ids)
model.logits(token_ids)
model.generate(prompt)
model.summary()
```

### Serialization and checkpoints

```text
save() load() save_model() load_model() save_weights() load_weights()
state_dict() load_state_dict() checkpoint() save_checkpoint() load_checkpoint()
save_vocab() load_vocab() serialize() deserialize() export_model() import_model()
```

### Quantization/compression helpers

```text
quantize() dequantize() int8() int4() float16() bfloat16()
prune() sparsify() compress() decompress() low_rank() factorize()
```

These are lightweight NumPy helpers; production quantization kernels are not part of `0.1.0`.

### Fine-tuning helpers

```text
finetune() freeze() unfreeze() freeze_layers() unfreeze_layers()
trainable() parameter_efficient_finetuning() lora() qlora() adapter()
prompt_tuning() prefix_tuning() gradient_clipping() weight_decay() ema()
```

### Learning-rate scheduling

```text
learning_rate() lr_scheduler() constant_lr() linear_decay()
cosine_decay() warmup()
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
├── serialization.py
├── quantization.py
├── scheduling.py
├── finetuning.py
├── model.py
└── builder.py
```

There are intentionally **no test or example directories** in the repository. This README is the primary user-facing documentation.

## PyPI release

The project uses modern `pyproject.toml` metadata and setuptools. Build distributions with:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Manual upload:

```bash
python -m twine upload dist/*
```

The repository also contains a GitHub Actions workflow that builds and validates the package when a GitHub Release is published, then uses PyPI Trusted Publishing. Configure the repository as a Trusted Publisher on PyPI before the first automated release.

## Roadmap

1. Trainable parameter objects.
2. Full reverse-mode automatic differentiation.
3. Real optimizer state and update loops.
4. Complete trainable Transformer output head.
5. Efficient dataset batching and streaming.
6. Better checkpointing and serialization.
7. Mixed precision and production quantization.
8. Hardware-aware VRAM and memory estimation.
9. Faster attention and inference.
10. Larger-model performance improvements.

## License

MIT License. See `LICENSE`.
