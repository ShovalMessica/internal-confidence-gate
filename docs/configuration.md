# Configuration

Edit [configs/task.yaml](../configs/task.yaml) to supply shared settings. Replace its required `null` placeholders with your values; optional settings already contain the agreed defaults. Example-specific data belongs in the [dataset](dataset-format.md).

The fields below are accepted by the configuration loader. Later pipeline behavior remains under design; **TODO** marks unresolved details.

## Model and paths

- **`model_name_or_path`** — Required Hugging Face model ID or absolute path to a local pretrained checkpoint. Uses Transformers’ `AutoModelForCausalLM` for text-only, decoder-only models. The tokenizer comes from the same location by default.
- **`dataset_path`** — Required absolute path to the JSONL dataset.
- **`output_dir` (optional)** — Absolute path to the results directory. If omitted, use an `outputs` folder beside the YAML file. Reuse of saved results for matching runs is planned but not implemented yet.

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

## Loading and validation

From Python, with the repository root as the working directory:

```python
from src.config import load_config

config = load_config("configs/task.yaml")
```

The function returns immutable `TaskConfig` settings with defaults filled in. It raises `ConfigurationError` with the discovered errors together; the future runner will display and log them.

- Required `null` placeholders must be replaced. Optional defaults apply only when fields are omitted; explicit `null` values are invalid.
- Unknown or duplicate YAML fields are errors. Token limits must be positive integers, `split_seed` a nonnegative integer, and `allow_abstention` a Boolean.
- Split ratios must contain exactly `train`, `validation`, and `test`, each strictly between 0 and 1, summing to 1 within floating-point tolerance.
- Dataset paths must point to existing `.jsonl` files; local checkpoint paths must point to existing directories. Hugging Face IDs are checked syntactically, without accessing the Hub.
- The path locating the YAML may be relative or absolute. Filesystem values inside it must be absolute. The default output path is computed beside the YAML, without creating it.

This step does not read dataset records, inspect model weights, or run generation. Actual model compatibility and dataset validation belong to later stages.

**TODO:** Complete the configuration for later pipeline stages. The example above is not yet a complete configuration for the full pipeline.
