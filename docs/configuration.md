# Configuration

Configuration contains settings shared across all dataset examples. The dataset contains the prompts, target answers, optional split assignments, and optional semantic annotations.

This page records the configuration design. **TODO** indicates an unresolved specification.

**TODO:** Define the configuration file format and how it is supplied to the toolkit.

## Model

- **`model_name_or_path`** — A Hugging Face model ID or a local checkpoint directory. This setting is required. The toolkit loads pretrained weights from the local directory or Hugging Face cache, downloading them if needed. It does not train the language model from scratch.
- **Tokenizer** — Loaded from the same model ID or checkpoint directory by default.

The toolkit targets text-only, decoder-only language models loaded through Hugging Face Transformers using `AutoModelForCausalLM`. Models must expose the internal activations needed for capture. Compatibility with every model is not assumed.

**TODO:** List verified models and supported Transformers versions; define model revision selection and tokenizer overrides.

## Data and results

- **Dataset path** — Where the toolkit reads your `.jsonl` dataset. See [Dataset format](dataset-format.md).
- **Output directory** — Where the toolkit saves model responses, captured activations, trained probes, and evaluation results.

**TODO:** Define setting names, the default output location, and rules for resuming runs or reusing saved outputs.

## Generation

- **`allow_abstention` (optional; default: `true`)** — Allows the model to decline to answer with `FINAL: UNKNOWN`. The toolkit's appended output instruction reflects this setting. When enabled, UNKNOWN predictions are reported separately and excluded from probe training and gate TPR/FPR.

The toolkit appends the instruction requiring `FINAL: <answer>` to each dataset prompt. Users do not need to add it themselves.

**TODO:** Define reasoning controls for supported models, generation budgets, sampling settings and defaults, and handling of responses that reach the generation limit.

**TODO:** Finalize the appended instruction, chat-template handling, response-validation rules, and handling of UNKNOWN when abstention is disabled or supplied as a target answer.

## Activation capture

By default, the toolkit captures residual-stream activations across all layers at:

- The last prompt token, before response generation.
- The token containing the colon in the final `FINAL:` marker.
- The answer token following that marker.

Additional input locations are supplied through the dataset's optional [`semantic_positions`](dataset-format.md#semantic-positions) field.

**TODO:** Define exact layer conventions, answer-token selection for multi-token answers, token-boundary handling, how multi-token spans become probe features, and any capture overrides.

## Training and evaluation

The toolkit uses dataset split assignments when provided. When assignments are omitted across the dataset, it creates random splits.

- **Training data** — Fits the probes.
- **Validation data** — Selects probe settings and the acceptance threshold.
- **Test data** — Evaluates the frozen probe and threshold against output-probability confidence.

The threshold is selected using a desired **correct-answer retention**: the fraction of correct validation predictions the gate should accept. Retention on test data is measured separately and may differ.

**TODO:** Define automatic split proportions, seed configuration, and handling of partially assigned datasets.

**TODO:** Define probe types, feature preprocessing, hyperparameter selection, the retention setting and its default, and output-probability scoring.

## Execution

The proposed default is to check CUDA availability, use a GPU when available, and otherwise use the CPU. The toolkit reports the selected device and allows an explicit override.

**TODO:** Finalize device selection, precision, batch size, and handling of insufficient memory or incompatible hardware.

## Example configuration

**TODO:** Add a complete example once setting names, defaults, and the configuration file format are settled.
