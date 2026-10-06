# Internal Confidence Gate

Task-adaptable toolkit for building confidence gates from internal model activations.

The toolkit is under active development.

## Quick start

Use Python 3.10 or newer. The first backend supports standard Hugging Face
Transformers text-only, decoder-only chat models through
`AutoModelForCausalLM`; CUDA is optional.

1. Install the dependencies:

   ```sh
   python -m pip install -r requirements.txt
   ```

2. Prepare your data using the required [JSONL format](docs/dataset-format.md).
3. Copy and fill in [configs/task.yaml](configs/task.yaml). Put shared task
   instructions in an optional `system_prompt_path`; each dataset `input`
   contains the complete example-specific user message.
4. Run the complete pipeline:

   ```sh
   python -m src.run configs/task.yaml
   ```

The command validates the inputs, generates and evaluates answers, captures
activations, trains and selects a probe, evaluates it on test data, and creates
a report. Results and an `execution.log` are saved under
`<output_dir>/<run_id>/`; rerunning the same task reuses validated artifacts.

To validate the configuration and dataset without loading a model, run:

```sh
python -m src.run configs/task.yaml --prepare-only
```

To run generation and Model Behavior evaluation without capturing activations
or training probes, run:

```sh
python -m src.run configs/task.yaml --behavior-only
```

This mode saves and reports behavior results even when the correct/incorrect
class counts are too small for probe training.

## Research goal

Language models can produce incorrect answers even when their output probabilities are high. Applications that need to decide which predictions to accept therefore need reliable ways to assess those predictions.

This repository investigates whether signals encoded in a model’s internal activations can distinguish correct from incorrect predictions better than confidence derived from output probabilities alone.

The toolkit provides a shared pipeline for generating model responses, collecting activations, and training and evaluating a lightweight reliability probe. An acceptance threshold turns the probe’s score into an accept-or-reject decision.

The gate does not modify the underlying model or generate replacement answers. Circuit finding is outside the initial scope.

## Task scope

The toolkit targets tasks where:

1. The model produces one final answer or decision that can be evaluated as a whole, although reasoning may precede it.
2. The answer can be evaluated objectively as correct or incorrect using a supplied target answer.
3. The response follows the toolkit’s fixed final-answer contract, allowing the answer to be located without knowing its correctness.

Answers may contain multiple tokens and come from a fixed set or an open vocabulary. Free-form responses containing multiple independently evaluated claims are outside the current scope.

Examples of suitable tasks include:

- **Classification:** “Is this review positive or negative?” → `FINAL: Positive`
- **Multiple choice:** “Which option is correct?” → `FINAL: B`
- **Factual question answering:** “What is the capital of England?” → `FINAL: London`
- **Structured reasoning:** reasoning followed by one final decision → `FINAL: 12`

## How it works

### Activations and probes

As a model processes a prompt and generates a response, it computes internal numerical representations called **activations**. These occur at different layers and token positions. A token is a unit of text processed by the model; one word may contain several tokens.

The toolkit first generates answers for examples with known target answers, then replays eligible saved predictions to collect activations. Comparing each model answer with its target answer determines whether the prediction is correct.

A small classifier, called a **probe**, learns to estimate prediction reliability from those activations.

**Target answers supply supervision for probe training and support evaluation. They are never provided to the probe as input features or inserted into the model’s prompt by the toolkit. Applying a trained probe does not require knowing the target answer.**

### Why select semantic positions?

A model produces activations across many layers and token positions. Collecting and comparing all of them can be expensive, so the toolkit focuses on locations with a meaningful relationship to the task or its final decision.

The default captures use the model hidden states returned across the embedding and layer stack at:

- **Prompt end (`prompt_end`):** the final token of the rendered chat prompt before generation begins.
- **Final-prompt end (`final_prompt_end`):** the final token of the injected `FINAL:` marker.
- **Answer tokens (`answer_tokens`):** every generated answer token.

With reasoning, `prompt_end` precedes the reasoning, while `final_prompt_end` follows the reasoning and final-answer instruction. In direct mode, the rendered prompt ends before the toolkit appends `FINAL:`, so these remain distinct positions. One-token and multi-token answers use the same capture structure.

The first capture backend uses the hidden states returned by Hugging Face Transformers. It saves the embedding output and every returned layer state as float16 tensors. TransformerLens and finer component-level signals, such as individual attention heads, remain a later optional extension.

These locations are candidates for informative signals, not guaranteed indicators of correctness. Training and validation determine which probes are useful; test data evaluates the frozen selection.

### Additional task-specific positions

Users are encouraged to annotate additional meaningful locations in their inputs.

For example, in a name-correction task, the name being checked may provide useful activations. It could appear near the beginning of one input and near the end of another.

A semantic position has a fixed **role**, not a fixed token index. Its text, location, and length may differ across examples.

Users mark these locations with character spans in the original input. The toolkit maps each span to every overlapping token in the exact rendered prompt and captures those tokens across all hidden-state layers. This requires a fast Hugging Face tokenizer. If additional positions are supplied, the same span keys and semantic roles must be present across all examples.

See the [dataset specification](docs/dataset-format.md#fields) for annotation fields and examples.

### Response format and abstention

By default, reasoning and direct-answer generation use:

```text
FINAL: <answer>
```

The toolkit supplies both the output instruction and the `FINAL:` marker. The model generates the answer after the marker. Users do not add either to their dataset prompts.

In direct-answer mode, the toolkit supplies the marker before generation. In reasoning mode, it first lets the model reason, closes reasoning if needed, then inserts the answer instruction and marker before resuming generation. Reasoning and answer generation have separate token limits.

Decoding can preserve model defaults or use greedy generation; see
[Generation](docs/configuration.md#generation).

See [Generation](docs/configuration.md#generation) for the exact injected instructions and token flow. The evaluation stage checks each saved answer against its dataset target.

The model may decline to answer using `FINAL: UNKNOWN`. This behavior is called
**abstention** and is enabled by default:

```yaml
allow_abstention: true
```

This is a shared run setting, not a dataset field. Users can disable it. The
toolkit’s appended instruction reflects the setting.

When abstention is enabled, UNKNOWN predictions are reported separately and excluded from probe training and gate TPR/FPR. `UNKNOWN` cannot then be a target answer. When abstention is disabled, it may be used as a normal target answer.

## Pipeline overview

**Configuration and dataset → model generation → answer evaluation → activation capture → probe training → validation selection → frozen test evaluation → reports.**

1. **Preparation:** validate the configuration and dataset, then preserve supplied splits or create reproducible train, validation, and test splits.
2. **Model Behavior:** generate answers, validate their structure, and compare them with target answers.
3. **Activation Capture:** replay eligible saved predictions and capture hidden states at the default and task-specific semantic positions.
4. **Confidence Probe:** train candidates on the training split, select a probe and threshold on validation, then evaluate that frozen choice on test against output-probability confidence.
5. **Reporting:** save the metrics, layer comparisons, and TPR-FPR curves described below.

By default, answer matching ignores case, trims surrounding whitespace, collapses repeated internal whitespace, and then requires complete-answer equality. Tasks that need different equivalence rules can provide a small `answer_match` function through `answer_matcher_path`; see [Answer evaluation](docs/configuration.md#answer-evaluation).

Before probe training, the runner reports model behavior overall and by split: correct predictions, wrong predictions, missed predictions (`UNKNOWN`), invalid outputs, and token-limit outputs. Every rate uses all examples in its scope as the denominator.

Tasks can also attach free-form `metadata` to dataset examples and configure
`custom_metrics_path` to add task-specific count-based metrics, such as
performance on a metadata-defined subset. These summaries do not alter
correctness labels or probes; see [Custom task metrics](docs/configuration.md#custom-task-metrics).

For eligible predictions:

- **Gate TPR:** accepted correct predictions divided by all correct predictions.
- **Gate FPR:** accepted incorrect predictions divided by all incorrect predictions.

Probe training uses mean-pooled hidden states, training-only standardization, and balanced L2 logistic regression. It saves a reliability score for train and validation while leaving test activations untouched. Because class weighting can alter the fitted class prior, this score is used for ranking and thresholding and is not presented as a calibrated probability.

Validation then selects one probe and acceptance threshold. For each candidate, the toolkit chooses the strictest threshold that retains at least the configured fraction of correct validation predictions, then selects the candidate with the lowest validation FPR. AUROC is reported for context but does not determine the winner.

On test data, the selected probe and threshold are frozen. The output-probability baseline receives its own validation threshold at the same target TPR. For multi-token answers, its confidence is the geometric mean of the generated tokens' probabilities; for a one-token answer, this is simply that token's probability. These probabilities come from the model's raw next-token logits, before sampling filters such as temperature, top-k, or top-p are applied.

The final report uses four gate metrics:

- **TPR:** fraction of correct predictions accepted.
- **FPR:** fraction of incorrect predictions accepted.
- **Balanced accuracy:** `(TPR + (1 - FPR)) / 2`.
- **AUROC:** threshold-independent ranking quality.

Machine-readable test results also include **coverage** (the fraction of predictions accepted) and **accepted-error rate** (incorrect accepted predictions divided by all accepted predictions).

For every captured position, a validation graph shows balanced accuracy from Layer 0—the initial token embedding—through every transformer layer. A validation TPR-FPR graph contains one representative per position: the layer with the lowest FPR while meeting `target_tpr`. The overall winner and output-probability baseline are highlighted. A separate test graph compares only the frozen overall winner with the probability baseline.

The TPR-FPR graph contains the same threshold sweep as a conventional ROC curve with its axes reversed: TPR is horizontal and FPR is vertical, so better gates move toward the lower-right. Head-level comparisons remain a later extension.

## Requirements and limitations

- White-box access to the model’s internal activations is required.
- Provide at least 700 valid examples. Before generation, the toolkit also requires at least 200 train, 100 validation, and 100 test examples.
- After excluding `UNKNOWN` and invalid responses, each split must contain at least:

  - **Training:** 100 correct and 100 incorrect predictions.
  - **Validation:** 50 correct and 50 incorrect predictions.
  - **Test:** 50 correct and 50 incorrect predictions.

  The toolkit checks these counts after generation and stops before probe training if a minimum is unmet.

  These are provisional minimums, not guarantees of reliable probe performance.

- Changes to the model, prompt, generation procedure, or data distribution require revalidation.
- Probe findings are correlational and do not establish causal mechanisms.
- A reliability score is an estimate, not a guarantee of correctness.

Deployment is the user’s responsibility and is outside the toolkit’s training-and-testing pipeline. Applying the trained gate requires the same activation features and position-selection rules used during training, but no target answer.

## Outputs and reuse

The runner creates or reuses:

```text
<output_dir>/.cache/generations/<context_id>/...
<output_dir>/<run_id>/run.json
<output_dir>/<run_id>/execution.log
<output_dir>/<run_id>/generation-manifest.jsonl
<output_dir>/<run_id>/evaluations/<evaluation_id>.jsonl
<output_dir>/<run_id>/activations.h5
<output_dir>/<run_id>/probes.h5
<output_dir>/<run_id>/reports/<report_id>/...
```

`run.json` is the provenance and stage registry; `execution.log` records every
invocation. HDF5 files contain activations and trained probes, while each report
contains `metrics.json`, plot-data CSVs, and PNG figures. See
[Configuration and run reuse](docs/configuration.md#run-identity-and-reuse) for
the complete artifact contract.

Interrupted expensive stages resume from saved work. Extending a dataset reuses
unchanged generations and compatible activations. To recompute a run's
generation and downstream artifacts while retaining its pinned model revision:

```sh
python -m src.run configs/task.yaml --force-recompute
```

## Worked example

The [simple classification example](examples/simple-classification/README.md)
generates a valid dataset and task configuration, then walks through preparation
and Model Behavior execution. It requires no model download for its
`--prepare-only` check.

## Development

Run the complete offline test suite without loading a model:

```sh
python -m unittest discover -s tests -v
```
