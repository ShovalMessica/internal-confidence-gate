# Configuration

Edit [configs/task.yaml](../configs/task.yaml) to supply shared settings. Replace its required `null` placeholders with your values; optional settings already contain the agreed defaults. Example-specific data belongs in the [dataset](dataset-format.md).

This is a design draft. New field names are proposed; **TODO** marks unresolved details.

## Model and paths

- **`model_name_or_path`** — Required Hugging Face model ID or absolute path to a local pretrained checkpoint. Uses Transformers’ `AutoModelForCausalLM` for text-only, decoder-only models. The tokenizer comes from the same location by default.
- **`dataset_path`** — Required absolute path to the JSONL dataset.
- **`output_dir` (optional)** — Absolute path to the results directory. If omitted, use an `outputs` folder beside the YAML file. Matching runs reuse existing results as described in the [design checklist](design-checklist.md#run-management-and-reuse).

User-supplied filesystem paths must be absolute; relative paths are rejected. Hugging Face model IDs are identifiers, not filesystem paths.

## Generation

- **`reasoning_mode`** — Required: `direct` or `reasoning`.

  - **Direct:** Append the answer instruction below to the prompt, supply `FINAL:` as the start of the model’s reply, then generate the answer.
  - **Reasoning:** Append:

    > Reason about the task first. A separate final-answer instruction will follow.

    Generate reasoning until its end boundary or token limit. Close reasoning if needed, then append the answer instruction and `FINAL:` before generating the answer.

  **Answer instruction:**

  > Return only the final answer on one line, without reasoning or explanation. The prefix FINAL: is already supplied; do not repeat it.

  **The toolkit supplies `FINAL:`; the model generates the answer after it.** Extraction uses this known boundary.

- **`reasoning_max_new_tokens` (optional; default: `1024`)** — Reasoning-token limit. Reaching it triggers the answer stage. Applies only in reasoning mode.

- **`answer_max_new_tokens` (optional; default: `64`)** — Separate answer-token limit, used in both modes.

- **`allow_abstention` (optional; default: `true`)** — Add the applicable sentence to the answer instruction:

  - Enabled: “If you cannot determine the answer, return UNKNOWN.”
  - Disabled: “Provide your best answer. Do not return UNKNOWN.”

  Enabled UNKNOWN predictions are reported separately and excluded from probe training and gate TPR/FPR.

Sampling uses the model’s defaults, without user overrides. Effective settings are recorded.

Invalid answers are saved with their failure reasons, excluded from probe training and gate metrics, and reported separately. No automatic retries.

## Automatic splitting

Applies only when the dataset omits `split`.

- **`split_ratios` (optional)** — Defaults to `train: 0.70`, `validation: 0.15`, `test: 0.15`. Values must sum to 1.
- **`split_seed` (optional; default: `42`)** — Seed for reproducible random splitting.

## Example

```yaml
# Required
model_name_or_path: "C:/models/my-model"
dataset_path: "C:/data/dataset.jsonl"
reasoning_mode: reasoning

# Optional — example output path; remaining values show defaults
output_dir: "C:/results/confidence-gate"
reasoning_max_new_tokens: 1024
answer_max_new_tokens: 64
allow_abstention: true
split_ratios:
  train: 0.70
  validation: 0.15
  test: 0.15
split_seed: 42
```

## Remaining design

- **Model/runtime:** compatibility, versions, chat templates, reasoning controls, device, precision, and batch size.
- **Generation:** missing model defaults, exact validation rules, completion at token limits, and unexpected UNKNOWN when disabled.
- **Capture:** layer conventions, multi-token features, and configurable options.
- **Training/evaluation:** probe settings, threshold selection, probability baseline, and reporting denominators.
- **Files:** run naming, saved artifacts, and reuse/resume implementation.

These details remain **TODO**; the example is not yet a complete configuration for the full pipeline.
