# Configuration

Edit [configs/task.yaml](../configs/task.yaml) to supply shared settings. Replace its required `null` placeholders with your values; optional settings already contain the agreed defaults. Example-specific data belongs in the [dataset](dataset-format.md).

The fields below are accepted by the configuration loader. Later pipeline behavior remains under design; **TODO** marks unresolved details.

## Model and paths

- **`model_name_or_path`** — Required Hugging Face model ID or absolute path to a local pretrained checkpoint. Uses Transformers’ `AutoModelForCausalLM` for text-only, decoder-only models. The tokenizer comes from the same location by default.
- **`model_revision` (optional)** — Hub commit, tag, or branch. When omitted, the loader resolves the current Hub version and returns its exact commit for later provenance recording. Local checkpoint paths do not accept this setting.
- **`device` (optional; default: `auto`)** — `auto`, `cpu`, `cuda`, or a numbered CUDA device such as `cuda:1`.
- **`dtype` (optional; default: `auto`)** — `auto`, `float16`, `bfloat16`, or `float32`. `auto` uses the dtype stored with the checkpoint.
- **`dataset_path`** — Required absolute path to the JSONL dataset.
- **`output_dir` (optional)** — Absolute path to the results directory. If omitted, use an `outputs` folder beside the YAML file. Each validated run is stored under `<output_dir>/<run_id>`.

User-supplied filesystem paths must be absolute; relative paths are rejected. Hugging Face model IDs are identifiers, not filesystem paths.

V1 requires a standard Transformers text-only, decoder-only causal language model with a tokenizer chat template. Models requiring `trust_remote_code=True` are not supported.

## Generation

- **`reasoning_mode`** — Required: `direct` or `reasoning`.

  - **Direct:** Disable thinking when the model's chat template supports that option, append the answer instruction below, supply `FINAL:` as the start of the model’s reply, then generate the answer.
  - **Reasoning:** Enable thinking when supported and append:

    > Reason about the task first. A separate final-answer instruction will follow.

    Generate reasoning until its end boundary or token limit. Close reasoning if needed, then append the answer instruction and `FINAL:` before generating the answer.

  **Answer instruction:**

  > Complete the FINAL: line with only the answer, without reasoning or explanation.

  **The toolkit supplies `FINAL:`; the model generates the answer after it.** Extraction uses this known boundary.

- **`reasoning_max_new_tokens` (optional; default: `1024`)** — Reasoning-token limit. Reaching it triggers the answer stage. Applies only in reasoning mode.

- **`answer_max_new_tokens` (optional; default: `64`)** — Separate answer-token limit, used in both modes.

- **`allow_abstention` (optional; default: `true`)** — Add the applicable sentence to the answer instruction:

  - Enabled: “If you cannot determine the answer, return UNKNOWN.”
  - Disabled: “Provide your best answer. Do not return UNKNOWN.”

  Enabled UNKNOWN predictions are reported separately and excluded from probe training and gate TPR/FPR.

- **`generation_seed` (optional; default: `42`)** — Base seed used to derive reproducible seeds from fixed dataset positions.

- **`direct_batch_size` (optional; default: `8`)** — Batch size in direct mode. Reasoning mode always processes one example at a time.

Sampling uses the model’s defaults, without user overrides. Effective settings are recorded.

Context-length and prompt-rendering failures are saved per example while generation continues. Unexpected model or runtime failures stop the run.

## Answer evaluation

After generation, the runner compares each saved answer with its dataset target. Matching ignores case, surrounding whitespace, and repeated internal whitespace. It does not remove punctuation or apply task-specific rules.

Generation failures, empty answers, repeated `FINAL:` markers, and multiple nonempty answer lines are invalid. Exact `UNKNOWN` responses are abstentions when abstention is enabled. Other structurally valid nonmatching answers are incorrect. Answers that reach the token limit remain valid and are reported separately.

Evaluation is saved to `evaluations.jsonl` and summarized by split in `run.json`. Insufficient correct or incorrect counts produce exit code `1` after saving the results.

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
device: auto
dtype: auto
reasoning_max_new_tokens: 1024
answer_max_new_tokens: 64
allow_abstention: true
generation_seed: 42
direct_batch_size: 8
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

The function returns immutable `TaskConfig` settings with defaults filled in. It raises `ConfigurationError` with the discovered errors together; the runner displays them.

- Required `null` placeholders must be replaced. Optional defaults apply only when fields are omitted; explicit `null` values are invalid.
- Unknown or duplicate YAML fields are errors. Token limits and `direct_batch_size` must be positive integers; generation and split seeds must be nonnegative integers; `allow_abstention` must be a Boolean.
- `model_revision`, when supplied, must be a nonempty string and may be used only with a Hub model ID. Device and dtype values must use the supported choices above.
- Split ratios must contain exactly `train`, `validation`, and `test`, each strictly between 0 and 1, summing to 1 within floating-point tolerance.
- Dataset paths must point to existing `.jsonl` files; local checkpoint paths must point to existing directories. Hugging Face IDs are checked syntactically, without accessing the Hub.
- The path locating the YAML may be relative or absolute. Filesystem values inside it must be absolute. The default output path is computed beside the YAML, without creating it.

The configuration loader itself does not read dataset records, inspect model weights, or run generation. The runner coordinates those stages. Use `--prepare-only` to stop before model loading.

## Run identity and reuse

After validation, the runner creates a small `run.json` record. The run ID is derived from the effective configuration and the SHA-256 hash of the exact dataset bytes. YAML and dataset paths do not affect the ID; `output_dir` only controls where the run is stored.

Model revision, device, dtype, generation seed, and active generation settings affect run identity. Inactive settings are excluded. For example, `reasoning_max_new_tokens` does not affect a direct-mode run, and split settings do not affect a dataset with supplied splits. Folder existence alone is not treated as completed work.

When `model_revision` is omitted, the model loader resolves one exact Hub commit and uses it for the tokenizer and weights. That commit is recorded at the first model load and reused by interrupted and forced runs rather than silently switching weights. Local checkpoint paths are treated as immutable for now; stronger local-checkpoint identity remains **TODO**.

Generation progress is appended after each reasoning example or completed direct batch. Complete, hash-validated generation and evaluation artifacts are reused without loading the model. Use `--force-recompute` to restart generation while keeping the recorded model revision; its downstream evaluation is cleared as well.

Two YAML files in different folders use different default output roots. Set the same absolute `output_dir` when they should share stored runs.

**TODO:** Complete the configuration for later pipeline stages. The example above is not yet a complete configuration for the full pipeline.
