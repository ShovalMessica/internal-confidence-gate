# Configuration

Edit [configs/task.yaml](../configs/task.yaml) to supply shared settings. Replace its required `null` placeholders with your values; optional settings already contain the agreed defaults. Example-specific data belongs in the [dataset](dataset-format.md).

The fields below are accepted by the configuration loader. Later pipeline behavior remains under design; **TODO** marks unresolved details.

## Model and paths

- **`model_name_or_path`** — Required Hugging Face model ID or absolute path to a local pretrained checkpoint. Uses Transformers’ `AutoModelForCausalLM` for text-only, decoder-only models. The tokenizer comes from the same location by default.
- **`model_revision` (optional)** — Hub commit, tag, or branch. When omitted, the loader resolves the current Hub version and returns its exact commit for later provenance recording. Local checkpoint paths do not accept this setting.
- **`device` (optional; default: `auto`)** — `auto`, `cpu`, `cuda`, or a numbered CUDA device such as `cuda:1`.
- **`dtype` (optional; default: `auto`)** — `auto`, `float16`, `bfloat16`, or `float32`. `auto` uses the dtype stored with the checkpoint.
- **`dataset_path`** — Required absolute path to the JSONL dataset.
- **`system_prompt_path` (optional)** — Absolute path to a nonempty UTF-8 text
  file used as the fixed system message for every example. It must not contain
  placeholders. When omitted, generation uses only the dataset `input` as the
  user message. The file's SHA-256 hash is recorded in the run identity, so
  changing its contents creates a different generation run.
- **`output_dir` (optional)** — Absolute path to the results directory. If omitted, use an `outputs` folder beside the YAML file. Each validated run is stored under `<output_dir>/<run_id>`.

User-supplied filesystem paths must be absolute; relative paths are rejected. Hugging Face model IDs are identifiers, not filesystem paths.

The current version requires a standard Transformers text-only, decoder-only causal language model with a tokenizer chat template. Models requiring `trust_remote_code=True` are not supported.

## Generation

The toolkit builds standard chat messages and lets the model tokenizer's
`apply_chat_template` render its model-specific format. When
`system_prompt_path` is configured, the fixed file becomes the `system`
message and each dataset `input` becomes the `user` message. Otherwise, the
dataset `input` is the only user message. The tokenizer then renders the
assistant generation boundary.

- **`reasoning_mode`** — Required: `direct` or `reasoning`.

  - **Direct:** Disable thinking when the model's chat template supports that option, then follow `direct_output_format`.
  - **Reasoning:** Enable thinking when supported and append:

    > Reason about the task first. A separate final-answer instruction will follow.

    Generate reasoning until its end boundary or token limit. Close reasoning if needed, then append the answer instruction and `FINAL:` before generating the answer.

  **Answer instruction:**

  > Complete the FINAL: line with only the answer, without reasoning or explanation.

  **The toolkit supplies `FINAL:`; the model generates the answer after it.** Extraction uses this known boundary.

- **`direct_output_format` (optional; default: `final_prefix`)** — Direct-mode
  answer protocol:

  - `final_prefix`: append the toolkit answer instruction and inject `FINAL:`
    before generating the answer.
  - `raw_answer`: send the configured system and user messages unchanged and
    treat the complete generated response as the answer. Use this only when
    your own prompt reliably requests one answer in the required format.

  Reasoning mode always uses `final_prefix`. In `raw_answer` mode,
  `final_prompt_end` is unavailable because no final marker is injected.

- **`reasoning_max_new_tokens` (optional; default: `1024`)** — Reasoning-token limit. Reaching it triggers the answer stage. Applies only in reasoning mode.

- **`answer_max_new_tokens` (optional; default: `64`)** — Separate answer-token limit, used in both modes.

- **`allow_abstention` (optional; default: `true`)** — In `final_prefix`
  mode, add the applicable sentence to the answer instruction:

  - Enabled: “If you cannot determine the answer, return UNKNOWN.”
  - Disabled: “Provide your best answer. Do not return UNKNOWN.”

  In `raw_answer` mode, your prompt controls whether the model may return
  `UNKNOWN`. Enabled UNKNOWN predictions are reported separately and excluded
  from probe training and gate TPR/FPR in either format.

- **`generation_seed` (optional; default: `42`)** — Base seed used to derive a reproducible seed from each example ID. It controls generation only. Activation capture replays saved tokens without sampling.

- **`direct_batch_size` (optional; default: `8`)** — Batch size in direct mode. Reasoning mode always processes one example at a time.

- **`decoding_strategy` (optional; default: `model_default`)** —
  `model_default` preserves the checkpoint's decoding behavior; `greedy`
  always selects the highest-scoring next token. Low-level sampling parameters
  are not exposed. Effective settings are recorded.

Context-length and prompt-rendering failures are saved per example while generation continues. Unexpected model or runtime failures stop the run.

## Answer evaluation

After generation, the runner compares each saved answer with its dataset target. Matching ignores case, surrounding whitespace, and repeated internal whitespace. It does not remove punctuation or apply task-specific rules.

- **`answer_matcher_path` (optional; default: omitted)** — Absolute path to a trusted Python file that defines:

  ```python
  def answer_match(prediction: str, target_answer: str) -> bool:
      return prediction.casefold().strip() == target_answer.casefold().strip()
  ```

  The function receives the raw extracted answer and raw dataset target and must return `True` or `False`. Use it for task-specific rules such as aliases, punctuation handling, or numeric tolerance. When omitted, the toolkit uses its stricter built-in complete-answer comparison described above. Keep the task-specific matching logic in this file because its exact bytes define the matcher hash.

  The matcher is loaded during preparation, so its path, function, and signature are checked before model generation. Exceptions raised for an example or non-Boolean return values stop evaluation with a clear error. The toolkit still decides whether an output is structurally valid and whether it is exactly `UNKNOWN`; the custom function only compares concrete answers.

Generation failures, empty answers, repeated `FINAL:` markers, and multiple nonempty answer lines are invalid. Exact `UNKNOWN` responses are abstentions when abstention is enabled. Other structurally valid nonmatching answers are incorrect. Answers that reach the token limit remain valid and are reported separately.

The runner prints and stores a Model Behavior summary overall and by split. For `N` examples in the reported scope:

- **Correct prediction:** a concrete answer accepted by `answer_match`; rate `N_correct / N`.
- **Wrong prediction:** a concrete answer rejected by `answer_match`; rate `N_wrong / N`.
- **Missed prediction:** the model returned `UNKNOWN`; rate `N_missed / N`.
- **Invalid output:** the response could not be evaluated structurally; rate `N_invalid / N`.
- **Token-limit output:** answer generation reached its token limit; rate `N_token_limit / N`. This is an independent diagnostic, so the same example also appears in one of the four outcomes above.

The matcher file's SHA-256 hash contributes to a separate evaluation ID, not the generation run ID. Evaluation is saved to `evaluations/<evaluation_id>.jsonl` and summarized in `run.json`. Reusing the same matcher bytes reuses that evaluation; changing them creates another evaluation from the saved generations without loading the model. Insufficient correct or incorrect counts produce exit code `1` after saving the results.

## Custom task metrics

- **`custom_metrics_path` (optional; default: omitted)** — Absolute path to a
  trusted Python file defining `compute_metrics(records)`. Generic Model
  Behavior metrics are always reported first; custom metrics are additional
  summaries and do not change correctness labels, probe readiness, or probes.

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

Every metric must contain exactly `numerator` and `denominator`, both
nonnegative integers with `numerator <= denominator`. The toolkit calculates
the rate; a zero denominator produces `null`. Return the same metric names for
the overall dataset and every split. Errors name the scope that failed.

The file is loaded during preparation and its bytes are hashed. Results are
stored under `custom_metrics` in `run.json`, identified by the dataset,
generation, evaluation, implementation, and protocol hashes. Changing only
the custom code recalculates these summaries without rerunning the model or
changing saved evaluations and probes. See
the function above as a minimal template.

## Probe training and validation selection

- **`probe_seed` (optional; default: `42`)** — Seed for reproducible linear-probe fitting.
- **`probe_positions` (optional)** — Capture positions to use for probe training,
  such as `[prompt_end]`. When omitted, train on every captured default and
  semantic position.
- **`probe_layers` (optional)** — Model-state numbers to use, such as `[35]`.
  Layer `0` is the embedding output; positive numbers identify returned
  transformer hidden states. When omitted, use every state.
- **`probe_excluded_answers` (optional)** — Normalized model answers to exclude
  from activation capture, probe training, validation selection, and test gate
  evaluation. Model Behavior reporting remains unchanged. For example,
  `[NONE]` restricts an NER gate to actual correction proposals.
- **`probe_regularization_c` (optional; default: `1.0`)** — Positive logistic-
  regression `C` value.
- **`probe_class_weight` (optional; default: `balanced`)** — `balanced` or
  `none`.
- **`target_tpr` (optional; default: `0.90`)** — Minimum fraction of correct validation predictions the selected gate must accept. Must be greater than `0` and at most `1`.

After activation capture, the toolkit trains one probe for every captured position and saved model state, including the embedding output. Multi-token answers and semantic spans are mean-pooled at each state; single-token positions are unchanged.

Each probe is an L2 logistic regression with `C=1`, balanced correct/incorrect class weights, and training-only feature standardization. Its score is `P(correct)`, so larger values indicate greater estimated reliability. Training saves train and validation scores without reading test activations.

For each candidate, validation selection chooses the highest observed `P(correct)` threshold whose acceptance rule, `score >= threshold`, retains at least `target_tpr` of correct validation predictions. Tied scores are accepted together, so achieved TPR may be higher than requested. The candidate with the lowest validation FPR is selected; exact FPR ties use position name and then saved state order. AUROC is saved and reported but is not a selection criterion.

All candidates and selections are stored in `<run_dir>/probes.h5`. The probe ID depends on the activation artifact, evaluation artifact, fixed training protocol, and `probe_seed`. The selection ID depends on that completed probe group and `target_tpr`. Changing the target therefore reuses trained probes and creates another small selection group without rerunning the model.

## Frozen test evaluation

After validation selection, the toolkit applies the selected probe and threshold unchanged to correct and incorrect test predictions. It reports the frozen threshold, TPR, FPR, balanced accuracy, and AUROC.

The output-probability baseline uses the same examples and receives its own threshold selected on validation at the same `target_tpr`. Its answer-level score is the geometric mean of generated answer-token probabilities, which avoids penalizing longer answers merely for containing more tokens. One-token answers keep their original token probability.

Per-example test labels, scores, and accept/reject decisions are stored in the existing `<run_dir>/probes.h5`; `run.json` stores the summary. Neither method uses test data to select a representation or threshold.

## Reporting

Reporting runs automatically after frozen test evaluation and uses only saved scores. It reports TPR, FPR, balanced accuracy, and AUROC.

- `validation_layers_<position>.png` shows validation balanced accuracy across Layer 0 (the token embedding) and every transformer layer.
- `validation_tpr_fpr.png` shows one validation-selected representative per captured position plus output probability. Each representative is the position's layer with the lowest FPR while meeting `target_tpr`.
- `test_tpr_fpr.png` compares only the frozen overall winner and output-probability baseline. Test data does not choose either method or threshold.
- `metrics.json` and CSV files store the exact values behind the figures.

The TPR-FPR plots reverse conventional ROC axes: TPR is on the horizontal axis and FPR is on the vertical axis. Lower-right is better. Reports are stored under `<run_dir>/reports/<report_id>` with a hash-validated manifest and are reused when their frozen inputs match.

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
# answer_matcher_path: "C:/tasks/my-task/answer_matcher.py"
# custom_metrics_path: "C:/tasks/my-task/custom_metrics.py"
generation_seed: 42
direct_batch_size: 8
direct_output_format: final_prefix
decoding_strategy: model_default
probe_seed: 42
probe_regularization_c: 1.0
probe_class_weight: balanced
target_tpr: 0.90
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
- Unknown or duplicate YAML fields are errors. Token limits and `direct_batch_size` must be positive integers; generation, probe, and split seeds must be nonnegative integers; `target_tpr` must be greater than 0 and at most 1; `allow_abstention` must be a Boolean. Direct output format and decoding strategy must use the choices documented above.
- `model_revision`, when supplied, must be a nonempty string and may be used only with a Hub model ID. Device and dtype values must use the supported choices above.
- `answer_matcher_path` and `custom_metrics_path`, when supplied, must be
  absolute paths to existing `.py` files. The runner loads and validates their
  required functions during preparation.
- Probe position, layer, and excluded-answer lists must be nonempty and contain
  no duplicates when supplied. Probe layers are checked against the loaded
  model states before training.
- Split ratios must contain exactly `train`, `validation`, and `test`, each strictly between 0 and 1, summing to 1 within floating-point tolerance.
- Dataset paths must point to existing `.jsonl` files; local checkpoint paths must point to existing directories. Hugging Face IDs are checked syntactically, without accessing the Hub.
- The path locating the YAML may be relative or absolute. Filesystem values inside it must be absolute. The default output path is computed beside the YAML, without creating it.

The configuration loader itself does not read dataset records, inspect model weights, or run generation. The runner coordinates those stages. Use `--prepare-only` to stop before model loading, or `--behavior-only` to stop after generation and answer evaluation without capturing activations or training probes.

## Run identity and reuse

After validation, the runner creates a small `run.json` record. The run ID is derived from the effective configuration and the SHA-256 hash of the exact dataset bytes. YAML and dataset paths do not affect the ID; `output_dir` only controls where the run is stored.

Model revision, device, dtype, generation seed, active generation settings, and pipeline protocol versions affect run identity. Inactive settings are excluded. For example, `reasoning_max_new_tokens` does not affect a direct-mode run, and split settings do not affect a dataset with supplied splits. Folder existence alone is not treated as completed work.

When `model_revision` is omitted, the model loader resolves one exact Hub commit and uses it for the tokenizer and weights. That commit is recorded at the first model load and reused by interrupted and forced runs rather than silently switching weights. Local checkpoint paths are treated as immutable for now; stronger local-checkpoint identity remains **TODO**.

Generations are cached per example under `<output_dir>/.cache/generations`. A cache entry is reusable only when the example ID, exact input, model context, and generation settings match. Each run stores an ordered `generation-manifest.jsonl` that references those shared records. Extending a dataset therefore generates only new or modified inputs; target-answer changes require reevaluation but not model generation.

After evaluation meets the probe-readiness requirements, each run stores all activations in one `<run_dir>/activations.h5` file. Capture replays the saved token sequence, verifies the saved answer-token probabilities, and stores float16 Hugging Face hidden states for `prompt_end`, every `answer_tokens` position, any configured semantic spans, and `final_prompt_end` when a final marker exists. Semantic spans require a fast tokenizer and are mapped before the activation file is modified. Interrupted capture resumes within the same file. When a dataset is extended, compatible unchanged examples are copied from an earlier run without running the model again.

Probe training uses a separate probe ID, so `probe_seed` and matcher changes do not affect generation or activation identity. Validation selection and frozen test evaluation have their own IDs, so downstream setting changes reuse completed upstream work. All groups share `<run_dir>/probes.h5`; completed groups are validated and reused.

The custom metric path does not affect run, generation, evaluation, activation,
or probe identity. Its file hash identifies a small summary stored in
`run.json`. Changing dataset metadata creates a new run because the exact
dataset bytes changed, while generations for unchanged IDs and inputs remain
reusable from the shared generation cache.

Sampled direct generation runs one example at a time so its ID-derived seed is independent of neighboring examples. Greedy direct generation may use `direct_batch_size`. Use `--force-recompute` to bypass cached generations for the current run while retaining the pinned model revision; downstream evaluations, activations, and probes for that run are cleared.

Two YAML files in different folders use different default output roots. Set the same absolute `output_dir` when they should share stored runs.

**TODO:** Complete the configuration for later pipeline stages. The example above is not yet a complete configuration for the full pipeline.
