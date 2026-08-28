# pymodel

**pymodel** is a modular NumPy-based Python framework for building, experimenting with, evaluating, training, and generating from small AI/Transformer-style models.

> **Status:** `0.1.0` — Alpha. The public API is available for experimentation. Some advanced training, autograd, quantization, and hardware features are lightweight/experimental rather than production-grade.

## Installation

```bash
pip install pymodel
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

### Focus system

Supported focus names include:

```text
intelligence, natural, coding, reasoning, math, knowledge,
creativity, conversation, instruction, accuracy, speed, memory,
context, language, translation, summarization, classification, roleplay
```

Focus helpers are available as `SUPPORTED_FOCUSES`, `FOCUS_PROFILES`, `normalize_focus()`, `build_focus_config()`, and `focus_description()`.

Multiple focuses are supported, for example `focus=["coding", "reasoning", "math"]`.

## Complete public function reference

The following sections document the public functions currently exported by `pymodel` in version `0.1.0`. This list is intentionally kept aligned with the package's actual public API; functions that do not exist are not documented here.

---

## Tokenization

### `tokenizer(text)`
Creates a basic whitespace tokenizer and returns a list of tokens.

### `tokenize(text)`
Alias-style convenience wrapper around `tokenizer()`.

### `detokenize(tokens)`
Joins a sequence of tokens into a space-separated string.

### `vocab(tokens)`
Builds a token-to-integer vocabulary from tokens or text.

### `build_vocab(tokens)`
Builds a vocabulary using `vocab()`.

### `update_vocab(vocabulary, tokens)`
Returns an updated copy of a vocabulary containing newly encountered tokens.

### `merge_vocab(*vocabularies)`
Combines multiple vocabularies into one vocabulary while preserving unique token IDs.

### `reverse_vocab(vocabulary)`
Converts a token-to-ID vocabulary into an ID-to-token vocabulary.

### `vocab_size(vocabulary)`
Returns the number of entries in a vocabulary.

### `token_id(token, vocabulary, default=None)`
Returns the ID for a token, or `default` when it is absent.

### `id_token(index, reverse)`
Returns the token associated with an ID in a reverse vocabulary.

### `token_exists(token, vocabulary)`
Checks whether a token exists in a vocabulary.

### `add_token(vocabulary, token)`
Returns a vocabulary containing the token, adding it if necessary.

### `remove_token(vocabulary, token)`
Returns a vocabulary with the requested token removed.

### `special_tokens(pad="<PAD>", unk="<UNK>", bos="<BOS>", eos="<EOS>")`
Returns the standard special-token mapping.

### `add_special_token(vocabulary, token)`
Adds a special token using the normal vocabulary insertion behavior.

### `encode(text, vocabulary, unknown_token=None)`
Tokenizes text and converts its tokens into vocabulary IDs.

### `decode(ids, vocabulary)`
Converts token IDs into text. Accepts either a reverse vocabulary or a normal vocabulary.

### `encode_batch(texts, vocabulary, unknown_token=None)`
Encodes multiple text strings.

### `decode_batch(ids, reverse)`
Decodes multiple ID sequences.

### `tokenize_batch(texts)`
Tokenizes multiple strings.

### `detokenize_batch(batch)`
Detokenizes multiple token sequences.

### `normalize_text(text)`
Normalizes whitespace in text.

### `clean_text(text)`
Cleans text using the package's normalization behavior.

### `split_text(text, separator=None)`
Splits text using the supplied separator.

### `truncate(text, length)`
Truncates text to a maximum character length.

### `pad_sequence(sequence, length, value=0, pad_id=None)`
Pads or truncates a sequence to exactly `length`. `pad_id` is an API-compatible alias for `value` and takes precedence when supplied.

---

## Embeddings

### `embedding(tokens, embedding_size, seed=None)`
Creates token embedding vectors for token IDs.

### `embeddings(tokens, embedding_size, seed=None)`
Convenience embedding operation for token sequences.

### `positional_embedding(sequence_length, embedding_size)`
Creates positional embeddings for a sequence.

### `position_embedding(sequence_length, embedding_size)`
Alias/convenience form of positional embedding creation.

### `sinusoidal_embedding(sequence_length, embedding_size)`
Creates sinusoidal positional representations.

### `rotary_embedding(sequence_length, embedding_size)`
Creates rotary-position representations.

### `embedding_similarity(a, b)`
Computes a similarity measure between embedding vectors or matrices.

---

## Attention

### `attention(query, key, value, mask=None)`
Applies an attention operation to query, key, and value arrays.

### `scaled_dot_product_attention(query, key, value, mask=None)`
Computes scaled dot-product attention.

### `self_attention(x, mask=None)`
Applies self-attention to an input sequence.

### `cross_attention(query, key, value, mask=None)`
Applies attention where the query sequence differs from the key/value sequence.

### `multihead_attention(x, heads=1, mask=None)`
Applies multi-head attention.

### `multi_query_attention(x, heads=1, mask=None)`
Provides multi-query attention behavior.

### `grouped_query_attention(x, heads=1, groups=1, mask=None)`
Provides grouped-query attention behavior.

### `causal_attention(x)`
Applies attention with causal masking.

### `local_attention(x, window_size=1)`
Restricts attention to a local window.

### `global_attention(x)`
Applies global attention over the sequence.

### `sparse_attention(x)`
Provides sparse attention behavior.

### `sliding_window_attention(x, window_size=1)`
Applies sliding-window attention.

### `block_attention(x, block_size=1)`
Applies block-oriented attention.

### `flash_attention(query, key, value, mask=None)`
Provides the package's lightweight flash-attention-compatible operation.

### `rotary_attention(x, positions=None)`
Applies rotary positional information to attention inputs.

### `alibi_attention(x, slopes=None)`
Applies ALiBi-style positional attention biasing.

### `attention_mask(length, dtype=None)`
Creates an attention mask for a sequence length.

### `causal_mask(length)`
Creates a causal mask preventing access to future positions.

### `padding_mask(sequence, pad_id=0)`
Creates a mask for padded sequence positions.

---

## Layers and activations

### `linear(x, weight, bias=None)`
Applies a linear transformation.

### `activation(x, kind="relu")`
Applies an activation selected by name.

### `relu(x)`
Rectified linear activation.

### `gelu(x)`
Gaussian Error Linear Unit activation.

### `sigmoid(x)`
Sigmoid activation.

### `tanh(x)`
Hyperbolic tangent activation.

### `leaky_relu(x, negative_slope=0.01)`
Leaky ReLU activation.

### `softplus(x)`
Softplus activation.

### `silu(x)`
SiLU/Swish activation.

### `swish(x)`
Swish activation.

### `mish(x)`
Mish activation.

### `softmax(x, axis=-1)`
Converts logits into normalized probabilities.

### `log_softmax(x, axis=-1)`
Computes log-softmax values.

### `layer_norm(x, eps=1e-5)`
Applies layer normalization.

### `batch_norm(x, eps=1e-5)`
Applies batch normalization.

### `rms_norm(x, eps=1e-5)`
Applies RMS normalization.

### `dropout(x, rate=0.0, training=True)`
Applies dropout when training is enabled.

### `flatten_layer(x)`
Flattens an array for layer-style processing.

### `feedforward(x, hidden_size=None)`
Applies a feed-forward transformation.

### `mlp(x, hidden_size=None)`
Applies a multi-layer-perceptron-style transformation.

### `residual(x, update)`
Adds a residual update to an input.

### `residual_block(x, fn)`
Applies a function and adds its result residually to the input.

---

## Transformer

### `transformer_block(x, ...)`
Constructs or applies a Transformer block using the package's attention, normalization, and feed-forward primitives.

### `transformer(x, ...)`
Applies the package's Transformer stack.

---

## Loss functions

### `loss(predictions, targets)`
General loss entry point.

### `cross_entropy(predictions, targets)`
Computes cross-entropy loss.

### `binary_cross_entropy(predictions, targets)`
Computes binary cross-entropy.

### `mse(predictions, targets)`
Computes mean squared error.

### `mae(predictions, targets)`
Computes mean absolute error.

### `huber_loss(predictions, targets, delta=1.0)`
Computes Huber loss.

### `kl_divergence(p, q)`
Computes Kullback-Leibler divergence.

### `contrastive_loss(...)`
Computes the package's contrastive loss.

### `label_smoothing(labels, smoothing=0.1)`
Applies label smoothing.

### `perplexity(loss_value)`
Converts a language-model loss value into perplexity.

### `accuracy(predictions, targets)`
Computes prediction accuracy.

### `f1_score(predictions, targets)`
Computes an F1 score.

### `recall(predictions, targets)`
Computes recall.

---

## Tensor and NumPy math utilities

### Creation

- `tensor(x)` — Converts input to the package tensor/NumPy representation.
- `zeros(shape)` — Creates an array of zeros.
- `ones(shape)` — Creates an array of ones.
- `full(shape, value)` — Creates an array filled with a value.
- `random(shape)` — Creates random values.
- `randn(shape)` — Creates normally distributed random values.
- `randint(low, high, size)` — Creates random integer values.
- `arange(...)` — Creates evenly spaced integer/floating values.

### Shape operations

- `reshape(x, shape)` — Reshapes an array.
- `flatten(x)` — Flattens an array.
- `squeeze(x, axis=None)` — Removes dimensions of size one.
- `unsqueeze(x, axis)` — Adds a dimension.
- `transpose(x, axes=None)` — Transposes an array.
- `permute(x, axes)` — Reorders dimensions.
- `repeat(x, repeats, axis=None)` — Repeats array values.
- `expand(x, shape)` — Expands an array to a requested shape where supported.
- `pad(x, ...)` — Pads an array.
- `roll(x, shift, axis=None)` — Rolls values along an axis.
- `cat(arrays, axis=0)` — Concatenates arrays.
- `stack(arrays, axis=0)` — Stacks arrays.
- `split(x, indices_or_sections, axis=0)` — Splits an array.
- `chunk(x, chunks, axis=0)` — Splits an array into chunks.

### Mathematical operations

- `matmul(a, b)` — Matrix multiplication.
- `dot(a, b)` — Dot product.
- `sum(x, axis=None)` — Sum reduction.
- `mean(x, axis=None)` — Mean reduction.
- `max(x, axis=None)` — Maximum reduction.
- `min(x, axis=None)` — Minimum reduction.
- `argmax(x, axis=None)` — Index of the maximum value.
- `argmin(x, axis=None)` — Index of the minimum value.
- `clip(x, min_value, max_value)` — Clips values to a range.
- `sqrt(x)` — Square root.
- `exp(x)` — Exponential.
- `log(x)` — Natural logarithm.
- `abs(x)` — Absolute value.
- `power(x, exponent)` — Power operation.
- `norm(x, ...)` — Vector/matrix norm.
- `normalize(x, ...)` — Normalizes values.
- `einsum(subscripts, *operands)` — Einstein summation.
- `where(condition, x, y)` — Selects values according to a condition.
- `masked_fill(x, mask, value)` — Replaces values at masked positions.
- `gather(x, indices, axis)` — Gathers values by index.
- `scatter(x, indices, values, axis)` — Scatters values by index.

---

## Autograd

### `gradient(...)`
Numerical/experimental gradient helper.

### `compute_gradients(...)`
Computes gradients using the package's available numerical/autograd helpers.

### `numerical_gradient(...)`
Computes a numerical gradient.

### `backward(...)`
Runs the package's backward/gradient helper.

### `requires_grad(...)`
Controls or queries whether an object participates in gradient computation.

### `detach(...)`
Returns a detached numerical value/array where supported.

### `no_grad(...)`
Provides a no-gradient context/helper.

### `zero_grad(...)`
Clears gradients where supported.

---

## Optimizers

### `optimizer(...)`
Creates/configures an optimizer.

### `sgd(...)`
Stochastic Gradient Descent optimizer helper.

### `adam(...)`
Adam optimizer helper.

### `adamw(...)`
AdamW optimizer helper.

### `rmsprop(...)`
RMSProp optimizer helper.

### `adagrad(...)`
AdaGrad optimizer helper.

### `update_weights(...)`
Updates weights using the available optimizer/update behavior.

### `step(...)`
Performs an optimizer step where supported.

### `learning_rate(...)`
Reads or configures learning-rate behavior.

### `zero_grad(...)`
Clears optimizer/model gradients where supported.

---

## Dataset and training

### `dataset(...)`
Creates or prepares dataset structures used by pymodel.

### `load_dataset(...)`
Loads a dataset from storage.

### `save_dataset(...)`
Saves a dataset to storage.

### `split_dataset(...)`
Splits a dataset into partitions.

### `shuffle(...)`
Shuffles dataset items.

### `batch(...)`
Creates batches from dataset data.

### `map_dataset(...)`
Maps a transformation across dataset items.

### `filter_dataset(...)`
Filters dataset items using a predicate.

### `cache(...)`
Caches dataset/computation data where supported.

### `train(...)`
Trains a model and returns training metrics/history.

### `evaluate(...)`
Evaluates a model and returns loss, accuracy, sample count, and perplexity metrics.

---

## Generation and sampling

### `generate(...)`
Generates text or token IDs from a model/prompt.

### `sample(...)`
Samples from a probability/logit distribution.

### `temperature(...)`
Applies/configures temperature-based sampling behavior.

### `top_k(...)`
Applies/configures top-k sampling.

### `top_p(...)`
Applies/configures nucleus/top-p sampling.

### `repetition_penalty(...)`
Applies a repetition penalty to generation scores.

### `frequency_penalty(...)`
Applies a frequency penalty to generation scores.

### `presence_penalty(...)`
Applies a presence penalty to generation scores.

### `stop_at_token(...)`
Stops generation when a specified token is encountered.

### `get_model(model_name)`
Loads/retrieves a saved model by name for generation or inference.

### `model_run(...)`
Runs an interactive model session.

---

## Serialization and model storage

### `save(...)`
Generic save helper.

### `load(...)`
Generic load helper.

### `save_model(model, ...)`
Saves a complete model.

### `load_model(...)`
Loads a complete model.

### `save_weights(...)`
Saves model weights.

### `load_weights(...)`
Loads model weights.

### `state_dict(model)`
Returns a model state dictionary.

### `load_state_dict(model, state)`
Loads a state dictionary into a model.

### `checkpoint(...)`
Creates or manages checkpoint data.

### `save_checkpoint(...)`
Saves a checkpoint.

### `load_checkpoint(...)`
Loads a checkpoint.

### `save_vocab(...)`
Saves a vocabulary.

### `load_vocab(...)`
Loads a vocabulary.

### `serialize(...)`
Serializes an object/model representation.

### `deserialize(...)`
Deserializes a serialized representation.

### `export_model(...)`
Exports a model representation.

### `import_model(...)`
Imports an exported model.

### `delete_model(...)`
Deletes a saved model.

### `edit_model(...)`
Edits model metadata/configuration where supported.

### `load_config(...)`
Loads model configuration.

### `save_config(...)`
Saves model configuration.

---

## Quantization and compression

### `quantize(...)`
Quantizes numerical data/model weights.

### `dequantize(...)`
Restores quantized data to a floating representation.

### `int8(...)`
Provides int8 numerical/quantization behavior.

### `int4(...)`
Provides int4 numerical/quantization behavior.

### `float16(...)`
Provides float16 numerical behavior.

### `bfloat16(...)`
Provides bfloat16-compatible numerical behavior.

### `prune(...)`
Prunes values/weights according to the available pruning behavior.

### `sparsify(...)`
Converts data toward a sparse representation.

### `compress(...)`
Compresses a model/data representation.

### `decompress(...)`
Decompresses a compressed representation.

### `low_rank(...)`
Creates/applies a low-rank representation.

### `factorize(...)`
Factorizes a matrix/representation.

---

## Fine-tuning

### `finetune(...)`
Runs/configures model fine-tuning.

### `freeze(...)`
Freezes parameters or a model component.

### `unfreeze(...)`
Unfreezes parameters or a model component.

### `freeze_layers(...)`
Freezes selected layers.

### `unfreeze_layers(...)`
Unfreezes selected layers.

### `trainable(...)`
Queries/configures trainable parameters.

### `parameter_efficient_finetuning(...)`
Provides parameter-efficient fine-tuning helpers.

### `lora(...)`
Provides LoRA-style adaptation helpers.

### `qlora(...)`
Provides QLoRA-style adaptation helpers.

### `adapter(...)`
Provides adapter-based fine-tuning helpers.

### `prompt_tuning(...)`
Provides prompt-tuning helpers.

### `prefix_tuning(...)`
Provides prefix-tuning helpers.

### `gradient_clipping(...)`
Clips gradients to a configured range/norm.

### `weight_decay(...)`
Applies/configures weight decay.

### `ema(...)`
Provides exponential moving-average weight behavior.

---

## Learning-rate scheduling

### `lr_scheduler(...)`
Creates/configures a learning-rate scheduler.

### `constant_lr(...)`
Provides a constant learning-rate schedule.

### `linear_decay(...)`
Provides a linearly decaying learning-rate schedule.

### `cosine_decay(...)`
Provides a cosine-decay learning-rate schedule.

### `warmup(...)`
Provides learning-rate warmup behavior.

### `scheduling(...)`
General scheduling helper.

---

## Focus API

### `normalize_focus(...)`
Normalizes one or more focus names into the package's canonical focus representation.

### `build_focus_config(...)`
Builds a configuration dictionary for selected focuses.

### `focus_description(...)`
Returns a description of a focus.

### `focus(...)`
Provides the public focus helper exposed by the focus module.

---

## Model management helpers

### `model_info(...)`
Returns model metadata/information.

### `model_size(...)`
Estimates or returns model size information.

### `count_parameters(...)`
Counts model parameters.

### `parameter_shapes(...)`
Returns parameter shapes.

### `trainable_parameters(...)`
Returns/counts trainable parameters.

### `summary(...)`
Provides summary information for a model/object where supported.

### `set_current_model(...)`
Sets the package's current/default model.

### `build(...)`
High-level model builder described above.

---

## System and device information

### `system_info()`
Returns a combined platform, CPU, RAM, storage, and GPU information dictionary.

### `ram()`
Returns RAM totals and availability.

### `cpu_info()`
Returns CPU information.

### `cpu_memory()`
Returns RAM information associated with CPU memory reporting.

### `gpu_info()`
Returns GPU availability/count/name information.

### `gpu_memory()`
Returns GPU memory information when available.

### `storage_info()`
Returns storage capacity, free space, and used space.

### `memory_info()`
Returns system RAM and process memory information.

### `device_info()`
Returns combined CPU/GPU device information.

### `device_count()`
Returns the available CPU/device count reported by pymodel.

### `is_gpu_available()`
Returns whether a supported GPU is available.

### `cuda()`
Reports whether CUDA is available to pymodel.

### `cpu()`
Reports CPU availability.

### `device(...)`
Selects/configures a device representation.

### `to_device(...)`
Moves/converts numerical data toward the selected device representation where supported.

### `clear_cache()`
Clears available package/system caches.

### `free_memory()`
Reports currently available system memory.

---

## Utility and text helpers

### `read_text(path)`
Reads text from a file.

### `write_text(path, text)`
Writes text to a file.

### `load_text(path)`
Loads text data.

### `save_text(path, text)`
Saves text data.

### `chunk_text(text, ...)`
Splits text into chunks.

### `split_sentences(text)`
Splits text into sentence-like units.

### `split_words(text)`
Splits text into words.

### `lower_text(text)`
Converts text to lowercase.

### `uppercase(text)`
Converts text to uppercase.

### `normalize_text(text)`
Normalizes text whitespace; see Tokenization above.

### `clean_text(text)`
Cleans text; see Tokenization above.

### `truncate_text(text, ...)`
Truncates text to the requested size.

### `shuffle(...)`
Shuffles a sequence/dataset.

### `random_seed(...)`
Creates/configures a random seed.

### `set_seed(...)`
Sets the package random seed.

### `seed(...)`
Seed helper.

---

## Additional public numerical helpers

The package also exposes the following NumPy/math-compatible helpers directly:

- `power()` — power operation.
- `norm()` — norm operation.
- `normalize()` — normalization operation.
- `split()` — array splitting.
- `chunk()` — array chunking.
- `cat()` — concatenation.
- `stack()` — stacking.
- `gather()` — indexed gathering.
- `scatter()` — indexed scattering.
- `masked_fill()` — masked replacement.
- `where()` — conditional selection.
- `einsum()` — Einstein summation.
- `reshape()` — reshape.
- `flatten()` — flatten.
- `squeeze()` — squeeze dimensions.
- `unsqueeze()` — add dimensions.
- `transpose()` — transpose dimensions.
- `permute()` — permute dimensions.
- `repeat()` — repeat values.
- `expand()` — expand dimensions.
- `pad()` — pad values.
- `roll()` — roll values.

---

## Model object API

`build()` returns a `Model` object. The high-level model interface includes:

```python
model.forward(token_ids)
model.logits(token_ids)
model.generate(prompt)
model.train(...)
model.evaluate(...)
model.summary()
model.named_parameters()
model.parameter_count()
model.parameters_info()
model.get_focus()
model.focus_scores()
model.focus_info()
model.set_focus(...)
```

### Core model properties

Models expose metadata such as:

- `model_name`
- `platform`
- `settings`
- `vocab`
- `reverse_vocab`
- `dataset`
- `focus`
- `dtype`
- `layers`
- `heads`
- `embedding_size`
- `hidden_size`
- `context_length`
- `training_steps`

### `forward(token_ids)`
Runs the model's forward pass and returns a NumPy array containing the sequence representation.

### `logits(token_ids)`
Returns model output logits over the vocabulary.

### `generate(prompt, ...)`
Generates text from a text prompt or token-ID input.

### `train(...)`
Runs model training and returns metrics including loss history, epochs, samples, steps, and before/after evaluation data.

### `evaluate(...)`
Evaluates the model and returns metrics including `loss`, `accuracy`, `samples`, and `perplexity`.

### `summary()`
Returns a dictionary containing model metadata, parameter counts, architecture information, focus, settings, and training state.

### `named_parameters()`
Returns model parameters keyed by their parameter names.

### `parameter_count()`
Returns the number of model parameters.

### `parameters_info()`
Returns detailed parameter information.

### `get_focus()`
Returns the model's current focus configuration.

### `focus_scores()`
Returns focus scores as a dictionary.

### `focus_info()`
Returns detailed focus information.

### `set_focus(...)`
Changes the model's active focus configuration.

---

## Optimizer object

`Optimizer` is the package's public optimizer class. It supports the optimizer-step and gradient-clearing interface used by the model/training system.

```python
optimizer = pymodel.Optimizer(...)
optimizer.step()
optimizer.zero_grad()
```

---

## Dtypes

The public API exposes numerical dtype helpers/names including:

```text
float16
bfloat16
int8
int4
```

These should be treated as package-level numerical/quantization interfaces in Alpha rather than promises of hardware-accelerated kernels.

## Parameter sizing and memory

`parameters` in `build()` refers to the requested number of model parameters, not the number of bytes.

Approximate raw weight storage is:

| dtype | Bytes per parameter |
|---|---:|
| float64 | 8 |
| float32 | 4 |
| float16 | 2 |
| bfloat16 | 2 |
| int8 | 1 |
| int4 | 0.5 |

Training can require substantially more memory because of gradients, optimizer state, activations, and batches.

## Settings

Example settings:

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

Additional settings are preserved by the model. Architecture values can be derived from the requested parameter count when not explicitly supplied.

## Project structure

```text
pymodel/
├── __init__.py
├── attention.py
├── autograd.py
├── builder.py
├── embeddings.py
├── finetuning.py
├── focus.py
├── generation.py
├── layers.py
├── loss.py
├── model.py
├── optim.py
├── quantization.py
├── scheduling.py
├── serialization.py
├── system.py
├── tokenization.py
├── training.py
├── transformer.py
└── utilities.py
```

## Alpha status

The package currently provides a broad experimental API. Numerical primitives and the tested high-level model workflow are functional, while advanced features such as full reverse-mode autograd, large-scale training, production quantization kernels, and hardware-specific acceleration remain areas for future development.

## Development roadmap

1. Expand trainable parameter support.
2. Improve reverse-mode automatic differentiation.
3. Improve optimizer state and large-scale training.
4. Expand trainable Transformer components.
5. Add efficient streaming/batched datasets.
6. Improve checkpointing and serialization.
7. Improve mixed precision and quantization.
8. Improve hardware-aware memory/VRAM estimation.
9. Improve attention and inference performance.
10. Improve large-model scalability.

## License

MIT License. See `LICENSE`.
