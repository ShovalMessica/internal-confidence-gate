# Configuration

Edit [configs/task.yaml](../configs/task.yaml), or your own copy of it.
All configuration fields below are top-level YAML keys in that file.
Fill in required `""` values, edit optional defaults as needed, and remove the
leading `#` to enable a commented setting. Example-specific fields belong in
your [JSONL dataset](dataset-format.md).

## Model and paths

- **`model_name_or_path`** - Required Hugging Face model ID or absolute path to a local pretrained checkpoint. Uses Transformers’ `AutoModelForCausalLM` for text-only, decoder-only models. The tokenizer comes from the same location by default.
- **`model_revision`** - Optional Hub commit, tag, or branch. When omitted for a Hub model, the loader resolves and records the exact commit. For a local checkpoint, this field is required as a stable version string; change it whenever any checkpoint file changes. It identifies caches but is not passed to Transformers.
- **`device` (optional; default: `auto`)** - `auto`, `cpu`, `cuda`, or a numbered CUDA device such as `cuda:1`.
- **`dtype` (optional; default: `auto`)** - `auto`, `float16`, `bfloat16`, or `float32`. `auto` uses the checkpoint's precision, not a hardware-based choice. Before generation, the toolkit warns if BF16 is loaded on a GPU without native BF16 support. If your model supports FP16, consider setting `dtype: "float16"` in your task YAML. The warning does not change precision automatically.
- **`dataset_path`** - Required absolute path to the JSONL dataset.
- **`system_prompt_path` (optional)** - Absolute path to a nonempty UTF-8 text
  file used as the fixed system message for every example. It must not contain
  placeholders. When omitted, generation uses only the dataset `input` as the
  user message. The file's SHA-256 hash is recorded in the run identity, so
  changing its contents creates a different generation run.
- **`output_dir` (optional)** - Absolute path to the results directory. If omitted, use `<repository>/outputs`, regardless of the YAML location. Behavior and full runs are stored under `<output_dir>/<run_name>`. Preparation checks write no files.

User-supplied filesystem paths must be absolute; relative paths are rejected. Hugging Face model IDs are identifiers, not filesystem paths.

The current version requires a standard Transformers text-only, decoder-only causal language model with a tokenizer chat template. Models requiring `trust_remote_code=True` are not supported.

## Generation

The toolkit builds standard chat messages and lets the model tokenizer's
`apply_chat_template` render its model-specific format. When
`system_prompt_path` is configured, the fixed file becomes the `system`
message and each dataset `input` becomes the `user` message. Otherwise, the
dataset `input` is the only user message. The tokenizer then renders the
assistant generation boundary.

- **`reasoning_mode`** - Required: `direct` or `reasoning`.

  **`direct`**

  The toolkit appends this instruction to the user message:

  ```text
  Answer immediately after FINAL: with only the final answer. Do not repeat the prefix or add reasoning or explanation.
  ```

  It renders the messages through the chat template, opens the assistant's
  response, supplies `FINAL:`, then generates the answer.

  **`reasoning`**

  The toolkit appends this instruction to the user message:

  ```text
  Reason about the task first. A separate final-answer instruction will follow.
  ```

  It generates reasoning until `</think>`, the model's end-of-response token,
  or the reasoning-token limit. It supplies `</think>` if missing, then appends
  the final-answer instruction above and `FINAL:` inside the same assistant
  response and resumes generation to produce the answer.

  The toolkit requests thinking enabled for reasoning mode and disabled for
  direct mode through the chat template when supported.

- **`reasoning_max_new_tokens` (optional; default: `1024`)** - Maximum reasoning
  tokens before starting the answer step. Used only in reasoning mode.

- **`answer_max_new_tokens` (optional; default: `64`)** - Maximum generated
  answer tokens. Used in both modes.

- **`allow_abstention` (optional; default: `true`)** - Adds one sentence to the
  final-answer instruction.

  When enabled:

  ```text
  If you cannot determine the answer, return UNKNOWN.
  ```

  When disabled:

  ```text
  Provide your best answer. Do not return UNKNOWN.
  ```

  Enabled `UNKNOWN` responses are reported as abstentions and excluded from
  probe training and gate evaluation. When disabled, any `UNKNOWN` response
  is compared with the target as an ordinary answer. This setting does not
  block the model from generating it.

- **`generation_seed` (optional; default: `42`)** - Base seed combined with the first example ID in each direct-mode batch, or each example ID in reasoning mode. It controls generation only. Activation capture replays saved tokens without sampling.

- **`direct_batch_size` (optional; default: `8`)** - Batch size in direct mode,
  for both sampling and greedy decoding. After reusing cached predictions, the
  toolkit saves the remaining batch membership and order in `run.json`.
  Resuming skips completed batches and retries an unfinished batch with its
  original members and seed, saving only missing predictions. Extended datasets
  reuse cached predictions and batch the new examples. Changing the batch size
  can change sampled answers. Reasoning mode processes one example at a time.

- **`decoding_strategy` (optional; default: `model_default`)** -
  `model_default` preserves the checkpoint's decoding behavior; `greedy`
  always selects the highest-scoring next token. Low-level sampling parameters
  are not exposed. Effective settings are recorded.

Context-length and prompt-rendering failures are saved per example while generation continues. Unexpected model or runtime failures stop the run.

## Answer evaluation

After generation, the runner compares each saved answer with its dataset target. Matching ignores case, surrounding whitespace, and repeated internal whitespace. It does not remove punctuation or apply task-specific rules.

- **`answer_matcher_path` (optional; default: omitted)** - Absolute path to a trusted Python file that defines:

  ```python
  def answer_match(prediction: str, target_answer: str) -> bool:
      return prediction.casefold().strip() == target_answer.casefold().strip()
  ```

  Save the function in a file such as `answer_matcher.py`. In your YAML, uncomment
  `answer_matcher_path` and set it to that file's absolute path.

  The function receives the raw prediction and target and must return a Boolean.
  Use it for aliases, punctuation rules, numeric tolerance, or other task-specific
  matching. The toolkit still handles invalid outputs and `UNKNOWN`.

Generation failures, empty answers, repeated `FINAL:` markers, and multiple nonempty answer lines are invalid. Exact `UNKNOWN` responses are abstentions when abstention is enabled. Other structurally valid nonmatching answers are incorrect. Answers that reach the token limit remain valid and are reported separately.

Probe training and gate coverage include only correct and incorrect predictions,
excluding invalid outputs and enabled abstentions.

The runner prints and stores a Model Behavior summary overall and by split. For `N` examples in the reported scope:

- **Correct prediction:** a concrete answer accepted by `answer_match`; rate `N_correct / N`.
- **Wrong prediction:** a concrete answer rejected by `answer_match`; rate `N_wrong / N`.
- **Missed prediction:** the model returned `UNKNOWN`; rate `N_missed / N`.
- **Invalid output:** the response could not be evaluated structurally; rate `N_invalid / N`.
- **Token-limit output:** answer generation reached its token limit; rate `N_token_limit / N`. This is an independent diagnostic, so the same example also appears in one of the four outcomes above.

Changing the matcher reevaluates saved answers without rerunning the model.
`--behavior-only` reports these results without enforcing probe-training minimums.

## Custom task metrics

- **`custom_metrics_path` (optional; default: omitted)** - Absolute path to a
  trusted Python file defining `compute_metrics(records)`. Generic Model
  Behavior metrics are always reported first; custom metrics are additional
  summaries and do not change correctness labels, probe readiness, or probes.

Save the function below in a file such as `custom_metrics.py`. In your YAML,
uncomment `custom_metrics_path` and set it to that file's absolute path.
Any metadata it uses belongs in each JSONL record's `metadata` object.

The toolkit calls the function once for all examples and once for each split.
Each record is a dictionary with this structure:

```python
{
    "id": int,
    "split": str,
    "prediction": str | None,
    "normalized_prediction": str | None,
    "target_answer": str,
    "outcome": "correct" | "incorrect" | "abstained" | "invalid",
    "is_correct": bool | None,
    "invalid_reason": str | None,
    "answer_token_limit": bool,
    "metadata": dict,
}
```

Return a nonempty mapping from metric names to integer counts:

```python
def compute_metrics(records):
    relevant = [
        record for record in records
        if record["metadata"].get("group") == "priority"
    ]
    correct = sum(record["is_correct"] is True for record in relevant)
    return {
        "priority_accuracy": {
            "numerator": correct,
            "denominator": len(relevant),
        }
    }
```

Each metric must contain only nonnegative integer `numerator` and `denominator`
values, with `numerator <= denominator`. The toolkit calculates the rate and
uses `null` when the denominator is zero. Return the same metric names overall
and for every split. Changing this file recalculates custom metrics without
rerunning the model.

## Probe training and validation selection

- **`probe_seed` (optional; default: `42`)** - Seed for reproducible linear-probe fitting.
- **`probe_positions` (optional)** - Capture positions to use for probe training,
  such as `[prompt_end]` or `[span_1]`. When omitted, train on every default
  capture position and semantic span.
- **`probe_layers` (optional)** - Model-state numbers to use, such as `[35]`.
  Layer `0` is the embedding output; positive numbers identify returned
  transformer hidden states. When omitted, use every state.
- **`probe_regularization_c` (optional; default: `1.0`)** - Positive logistic-
  regression `C` value.
- **`probe_class_weight` (optional; default: `balanced`)** - `balanced` or
  `none`.
- **`target_tpr` (optional; default: `0.90`)** - Minimum fraction of correct validation predictions the selected gate must accept. Must be greater than `0` and at most `1`.

The toolkit trains an L2 logistic-regression probe for each selected position and
layer. By default, it uses every captured position and state. Multi-token spans
are mean-pooled, and feature standardization is fitted on training data only.

For each probe, validation chooses the highest observed threshold that meets
`target_tpr`. It then selects the probe with the lowest FPR. AUROC is reported
but does not affect selection. Test data is not read during this process.
Candidates and selections are stored in `<run_dir>/probes.h5`.

## Gate evaluation

The toolkit applies the validation-selected probe and threshold unchanged to
test predictions. It reports TPR, FPR, balanced accuracy, AUROC, coverage, and
accepted-error rate. Coverage is the accepted fraction of eligible predictions;
accepted-error rate is the incorrect fraction among accepted predictions.
Invalid outputs and enabled abstentions are ineligible.

The output-probability baseline uses the same examples and receives its own threshold selected on validation at the same `target_tpr`. Its answer-level score is the geometric mean of generated answer-token probabilities, which avoids penalizing longer answers merely for containing more tokens. One-token answers keep their original token probability. Token probabilities are calculated from the model's raw next-token logits, before temperature, top-k, top-p, or other sampling filters are applied.

Neither method uses test data to select a representation or threshold.

## Reporting

Reporting runs automatically after gate evaluation and uses only saved scores. It reports TPR, FPR, balanced accuracy, and AUROC.

- `validation_layers_<position>.png` shows validation balanced accuracy across Layer 0 (the token embedding) and every transformer layer.
- `validation_tpr_fpr.png` shows one validation-selected representative per captured position plus output probability. Each representative is the position's layer with the lowest FPR while meeting `target_tpr`.
- `test_tpr_fpr.png` compares only the validation-selected overall winner and output-probability baseline. Test data does not choose either method or threshold.
- `metrics.json` and CSV files store the exact values behind the figures.

The TPR-FPR plots reverse conventional ROC axes: TPR is on the horizontal axis and FPR is on the vertical axis. Lower-right is better. Reports are stored under `<run_dir>/reports/<report_id>` with a hash-validated manifest and are reused when their saved inputs match.

## Automatic splitting

Applies only when the dataset omits `split`.

- **`split_ratios` (optional)** - Defaults to `train: 0.70`, `validation: 0.15`, `test: 0.15`. Values must sum to 1.
- **`split_seed` (optional; default: `42`)** - Seed for reproducible random splitting.

## Example

These settings belong in `configs/task.yaml` or your own configuration file.
Replace the example paths with your actual paths.

```yaml
# Required
model_name_or_path: "C:/models/my-model"
model_revision: "my-model-v1"
dataset_path: "C:/data/dataset.jsonl"
reasoning_mode: "reasoning"

# Optional - example output path; remaining values show defaults
output_dir: "C:/results/confidence-gate"
device: "auto"
dtype: "auto"
reasoning_max_new_tokens: 1024
answer_max_new_tokens: 64
allow_abstention: true
# answer_matcher_path: "C:/tasks/my-task/answer_matcher.py"
# custom_metrics_path: "C:/tasks/my-task/custom_metrics.py"
generation_seed: 42
direct_batch_size: 8
decoding_strategy: "model_default"
probe_seed: 42
probe_regularization_c: 1.0
probe_class_weight: "balanced"
target_tpr: 0.90
split_ratios:
  train: 0.70
  validation: 0.15
  test: 0.15
split_seed: 42
```

## Validation

Run preparation from the repository root:

```sh
python -m src.run configs/task.yaml --prepare-only
```

Preparation checks the configuration and dataset without loading the model or
writing files. Results are printed in the terminal.
Unknown fields, invalid values, duplicate YAML keys, and bad paths are reported
together. In the command above, replace `--prepare-only` with `--behavior-only`
to run through answer evaluation without capturing activations or training probes.
Add `--show-examples 10` to that command to inspect the first 10 inputs,
predictions, targets, and outcomes. These flags go in the command, not the YAML.

## Run identity and reuse

Behavior and full runs are stored under `<output_dir>/<run_name>`.
The folder name comes from the YAML filename: `dev-task.yaml` creates
`dev-task/`. Changed inputs create `dev-task_2/`, `dev-task_3/`, and so on.
Matching setups reuse their existing folder, even if the YAML is renamed or
you return to earlier settings. Existing hash-named folders remain reusable.

Fingerprints stay in `run.json` and depend on the effective configuration,
prompt contents, and exact dataset contents, not their file paths.
`--prepare-only` creates no run folder or log.
The resolved model revision is pinned so resumed runs use the same weights.

Running the same setup again reuses completed work. Dataset extensions can reuse
generations and activations for unchanged examples. Changes limited to matching,
metrics, probe settings, or threshold selection reuse compatible earlier stages.

Add `--force-recompute` to your `python -m src.run configs/task.yaml` command
to regenerate the current run. This clears its downstream
evaluation, activation, probe, and report artifacts. Behavior and full invocations are also
recorded in `<run_dir>/execution.log`.
Progress updates one terminal line; the log keeps the final update or the last
update before interruption.
Elapsed time and ETA use readable units, such as `4h 30m` or `30m 8s`.
