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

- **Reasoning mode** — Select direct-answer generation or reasoning followed by answer generation, as specified below.
- **Generation settings** — Use the selected model's generation defaults, with optional user overrides. The toolkit records the effective settings for reproducibility. It does not impose greedy decoding as a toolkit-wide default.
- **`allow_abstention` (optional; default: `true`)** — Allows the model to decline to answer with `FINAL: UNKNOWN`. The toolkit's appended output instruction reflects this setting. When enabled, UNKNOWN predictions are reported separately and excluded from probe training and gate TPR/FPR.

**Who supplies `FINAL:`?** The toolkit does, in both modes. It supplies the beginning of the model's reply and asks the model to continue from there. This is called **prefilling**. The model generates only the answer after the supplied marker during the answer stage. The marker is part of the assistant continuation, not a bare suffix inside the user's message.

**Shared answer instruction**

Use this exact text wherever the steps below refer to the answer instruction:

```text
Return only the final answer on one line. Do not include reasoning, explanation, quotation marks, or markdown. The answer prefix FINAL: is already supplied; do not repeat it.
```

When `allow_abstention` is `true`, add this sentence on the next line:

```text
If you cannot determine the answer, return UNKNOWN.
```

When it is `false`, use this sentence instead:

```text
Provide your best answer. Do not return UNKNOWN.
```

Users supply their task prompt in `input`; the toolkit adds these instructions without changing the dataset text or its annotation offsets.

**Direct answers**

1. Append two newlines and the answer instruction, including the applicable UNKNOWN sentence, to `input`.
2. Format the prompt for the selected model, disabling native thinking where supported.
3. Begin the assistant reply with the literal text `FINAL:` (no trailing space), supplied by the toolkit.
4. Generate the answer continuation under the answer-token limit.

```text
Toolkit supplies:  [formatted task and instructions] [assistant start] FINAL:
Model generates:   London
```

The bracketed items illustrate placement; they are not literal text to inject. This adds a common answer boundary to the original non-reasoning NER flow, which generated the answer directly without a marker.

**Reasoning followed by an answer**

1. Append two newlines and this instruction to `input`: `Reason about the task first. A separate final-answer instruction will follow the reasoning stage.`
2. Format the prompt with native reasoning enabled and generate until the model's reasoning-end boundary, an end-of-response token, or the reasoning-token limit.
3. Preserve the generated reasoning. If its closing boundary was not emitted, append the model's required closing boundary and record that closure was forced.
4. Append two newlines, the shared answer instruction including the applicable UNKNOWN sentence, one newline, and the literal `FINAL:`. These are supplied continuation tokens after the reasoning, not another generated response or a new user turn.
5. Resume generation under a separate answer-token limit.

```text
Model generates:   [reasoning]
Toolkit supplies:  [closing boundary if needed] [answer instruction] FINAL:
Model generates:   London
```

This follows the original bounded-reasoning flow. The model may finish reasoning before its limit; reaching that limit triggers the answer stage rather than consuming the answer budget. Reasoning and answer limits count generated tokens in their respective stages, excluding injected control text.

**Answer extraction and validation**

Record the injected marker's token boundary. Extract the answer from the generated continuation after that boundary; do not search the prompt or reasoning for the last occurrence of `FINAL:`. Injected text is not model-generated output and contributes no generated-answer probabilities.

After removing terminal control tokens and surrounding whitespace, require a nonempty single-line answer with no repeated `FINAL:` marker. Report empty, multiline, repeated-marker, and answer-budget-exhausted outputs as invalid; do not silently truncate, repair, or retry them. A normal end-of-response token at the limit counts as completion. Invalid outputs are counted separately and excluded from probe training and gate metrics. An exact normalized UNKNOWN answer is handled separately when abstention is enabled and is invalid when disabled.

Prefilling guarantees the supplied marker and its location. It does not guarantee that the continuation obeys the instruction or is correct. A single-line explanation may still pass structural validation; answer comparison and the remaining normalization rules determine correctness.

**TODO:** Define the mode setting's name and default, and numerical defaults and field names for the separate reasoning and answer limits.

**TODO:** Implement and verify chat-template placement, assistant continuation, native-thinking controls, and reasoning boundaries for supported models. `</think>` is used by the original implementation, not assumed for every model. Reject unsupported model/mode combinations before processing data; do not silently change the protocol. Define formatting for models without a chat template.

**TODO:** Finalize terminal-control-token handling, context-length overflow handling, and UNKNOWN target-answer rules. Record injected tokens, stage boundaries, effective settings, and forced reasoning closure in saved outputs.

**TODO:** Define exposed generation overrides and behavior when the model supplies no generation configuration. Distinguish saved generation defaults from recommendations that appear only in the model's documentation.

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
