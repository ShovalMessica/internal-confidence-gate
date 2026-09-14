# Configuration

Configuration contains settings shared across all dataset examples. The dataset contains the prompts, target answers, optional split assignments, and optional semantic annotations.

This page specifies planned behavior; the generation protocol below is not yet implemented in this repository. **TODO** indicates an unresolved specification.

Open decisions and dependencies are tracked in the [design checklist](design-checklist.md).

**TODO:** Define the configuration file format and how it is supplied to the toolkit.

## Model

- **`model_name_or_path`** — A Hugging Face model ID or a local checkpoint directory. This setting is required. The toolkit loads pretrained weights from the local directory or Hugging Face cache, downloading them if needed. It does not train the language model from scratch.
- **Tokenizer** — Loaded from the same model ID or checkpoint directory by default.

The selected backend is Hugging Face Transformers, using `AutoModelForCausalLM`. Users choose the model; the toolkit targets text-only, decoder-only language models with access to the activations needed for capture. Loading successfully does not by itself establish compatibility with the full pipeline.

**TODO:** List verified models and supported Transformers versions; define model revision selection and tokenizer overrides.

## Data and results

- **Dataset path** — Where the toolkit reads your `.jsonl` dataset. See [Dataset format](dataset-format.md).
- **Output directory** — Where the toolkit saves model responses, captured activations, trained probes, and evaluation results.

**TODO:** Define setting names, the default output location, and rules for resuming runs or reusing saved outputs.

## Generation

**`reasoning_mode`** — Choose whether the model answers directly or reasons first. Proposed values: `direct` and `reasoning`. **Default: TODO.**

- **Direct:** Append the answer instruction below to the prompt, supply `FINAL:` as the start of the model's reply, then generate the answer.
- **Reasoning first:** Append:
  > Reason about the task first. A separate final-answer instruction will follow.

  Generate reasoning until its end boundary or token limit. Close reasoning if needed, then append the answer instruction and `FINAL:` before generating the answer.

**Answer instruction:**

> Return only the final answer on one line, without reasoning or explanation. The prefix FINAL: is already supplied; do not repeat it.

**The toolkit supplies `FINAL:` in both modes; the model generates the answer after it.**

**TODO:** Finalize the field name and values, default mode, and model-specific reasoning controls and chat-template handling.

**Generation settings** — Use the model's generation defaults with optional user overrides; record effective settings for reproducibility.

**TODO:** Define override fields and behavior when the model provides no generation configuration.

**Token limits** — Separate limits for reasoning and answer generation. Direct mode uses only the answer limit.

**TODO:** Define field names, numerical defaults, and handling of incomplete answers.

**`allow_abstention` (optional; default: `true`)** — Allows `UNKNOWN`. Add the applicable sentence to the answer instruction, before the supplied `FINAL:`:

- Enabled: “If you cannot determine the answer, return UNKNOWN.”
- Disabled: “Provide your best answer. Do not return UNKNOWN.”

When enabled, UNKNOWN predictions are reported separately and excluded from probe training and gate TPR/FPR.

**TODO:** Finalize UNKNOWN target-answer handling and unexpected UNKNOWN responses when disabled.

Answers are extracted after the supplied marker; formatting failures are reported.

**TODO:** Finalize answer-validation and failure-handling rules.

## Activation capture

By default, the toolkit captures residual-stream activations across all layers at:

- The last token supplied before the first generation stage: before reasoning in reasoning mode, or the injected `FINAL:` colon in direct mode.
- The token containing the colon in the final `FINAL:` marker.
- The answer token following that marker.

Additional input locations are supplied through the dataset's optional [`semantic_positions`](dataset-format.md#semantic-positions) field.

In direct mode, the last prompt token and final-marker token coincide under this protocol. They refer to one capture, not two distinct locations. In reasoning mode they occur at different stages.

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
