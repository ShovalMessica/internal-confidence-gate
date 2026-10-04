# Internal Confidence Gate

Task-adaptable toolkit for building confidence gates from internal model activations.

The toolkit is under development. Items marked **TODO** identify specifications or implementation details that are not yet finalized.

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

The toolkit collects activations while the model answers examples with known target answers. Comparing each model answer with its target answer determines whether the prediction is correct.

A small classifier, called a **probe**, learns to estimate prediction reliability from those activations.

**Target answers supply supervision for probe training and support evaluation. They are never provided to the probe as input features or inserted into the model’s prompt by the toolkit. Applying a trained probe does not require knowing the target answer.**

### Why select semantic positions?

A model produces activations across many layers and token positions. Collecting and comparing all of them can be expensive, so the toolkit focuses on locations with a meaningful relationship to the task or its final decision.

The default captures use residual-stream activations—the representations carried between model layers—across all layers at:

- **Last prompt token:** the last token supplied before the first generation stage.
- **Final-answer marker:** the token containing the colon in the final `FINAL:` marker.
- **Answer:** the answer token following that marker.

With reasoning, the last prompt token precedes the reasoning, while the final-answer marker occurs after it. In direct-answer mode, the supplied prompt ends with the injected `FINAL:` marker, so these two capture locations coincide. Answer-token activations reflect computation after answer generation has begun.

These locations are candidates for informative signals, not guaranteed indicators of correctness. Training and validation determine which probes are useful; test data evaluates the frozen selection.

**TODO:** Define selection for multi-token answers, whitespace and token-boundary handling, and exact layer conventions.

### Additional task-specific positions

Users are encouraged to annotate additional meaningful locations in their inputs.

For example, in a name-correction task, the name being checked may provide useful activations. It could appear near the beginning of one input and near the end of another.

A semantic position has a fixed **role**, not a fixed token index. Its text, location, and length may differ across examples.

Users mark these locations with character spans in the original input. The toolkit maps them to tokens after prompt formatting and tokenization. If additional positions are supplied, the same span keys and semantic roles must be present across all examples.

See the [dataset specification](docs/dataset-format.md#fields) for annotation fields, examples, and unresolved mapping details.

### Response format and abstention

Both reasoning and direct-answer generation use:

```text
FINAL: <answer>
```

The toolkit supplies both the output instruction and the `FINAL:` marker. The model generates the answer after the marker. Users do not add either to their dataset prompts.

In direct-answer mode, the toolkit supplies the marker before generation. In reasoning mode, it first lets the model reason, closes reasoning if needed, then inserts the answer instruction and marker before resuming generation. Reasoning and answer generation have separate token limits.

See [Generation](docs/configuration.md#generation) for the exact injected instructions and token flow. The evaluation stage checks each saved answer against its dataset target.

The model may decline to answer using `FINAL: UNKNOWN`. This behavior is called **abstention** and is enabled by default:

```yaml
allow_abstention: true
```

This is a shared run setting, not a dataset field. Users can disable it. The toolkit’s appended instruction reflects the setting.

When abstention is enabled, UNKNOWN predictions are reported separately and excluded from probe training and gate TPR/FPR. `UNKNOWN` cannot then be a target answer. When abstention is disabled, it may be used as a normal target answer.

## Pipeline overview

1. **Prepare data:** provide complete task prompts and target answers in the required dataset format. Optionally assign splits and annotate additional semantic positions.
2. **Generate:** run the model and save its responses, exact token sequence, and answer-token probabilities.
3. **Evaluate predictions:** validate response formatting and compare answers with target answers after normalization.
4. **Capture activations:** replay the saved token sequence at the selected positions.
5. **Train:** fit probes using training activations and prediction correctness.
6. **Validate:** select probe settings and an acceptance threshold using validation data.
7. **Test:** evaluate the frozen probe and threshold on test data and compare performance with output-probability confidence.

By default, answer matching ignores case, trims surrounding whitespace, collapses repeated internal whitespace, and then requires complete-answer equality. Tasks that need different equivalence rules can provide a small `answer_match` function through `answer_matcher_path`; see [Answer evaluation](docs/configuration.md#answer-evaluation).

Before probe training, the runner reports model behavior overall and by split: correct predictions, wrong predictions, missed predictions (`UNKNOWN`), invalid outputs, and token-limit outputs. Every rate uses all examples in its scope as the denominator.

For eligible predictions:

- **Gate TPR:** accepted correct predictions divided by all correct predictions.
- **Gate FPR:** accepted incorrect predictions divided by all incorrect predictions.

**TODO:** Finalize output-probability scoring, probe configuration, selection procedures, and the final probe report.

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

## Getting started

### Prerequisites and installation

Preparation, model loading, generation, and answer evaluation are implemented. Use Python 3.10 or newer and, from the repository root, install the dependencies:

```sh
python -m pip install -r requirements.txt
```

Model loading currently supports standard Hugging Face Transformers text-only, decoder-only chat models through `AutoModelForCausalLM`. Models requiring custom remote code are outside V1 support. CUDA is optional; model size determines the required CPU/GPU memory.

### Prepare your dataset

Follow the [dataset format](docs/dataset-format.md).

Each example contains:

- `id`: a unique identifier.
- `input`: the complete task prompt.
- `target_answer`: the expected final answer.
- Optional `split`: a training, validation, or test assignment.
- Optional `semantic_spans`: additional character spans for activation capture.

The toolkit adds the output instruction and generates responses itself. Users do not need to supply existing model predictions.

### Configure and run

Fill in [configs/task.yaml](configs/task.yaml), then run preparation, generation, and answer evaluation:

```sh
python -m src.run configs/task.yaml
```

To validate and register the run without loading a model:

```sh
python -m src.run configs/task.yaml --prepare-only
```

The runner reports valid and excluded examples and the final split sizes. It creates or reuses:

```text
<output_dir>/.cache/generations/<context_id>/...
<output_dir>/<run_id>/run.json
<output_dir>/<run_id>/generation-manifest.jsonl
<output_dir>/<run_id>/evaluations/<evaluation_id>.jsonl
```

The run ID represents the generation settings and exact dataset contents. `run.json` stores provenance, matcher-specific evaluation summaries, and completed stages. The manifest references shared per-example generations containing generated tokens, token log probabilities, and recoverable failures. Each evaluation file stores correctness outcomes in dataset order.

Interrupted generation resumes from saved work. If a dataset is extended, unchanged examples with the same ID and input reuse their cached generations; only new or modified inputs run through the model. Changing only a target answer reruns evaluation, while changing only the answer matcher creates another evaluation from the saved generations. To discard a run's generation and downstream evaluations and regenerate with the same pinned model revision:

```sh
python -m src.run configs/task.yaml --force-recompute
```

If any split lacks the required correct or incorrect predictions, evaluation is saved and the runner exits with a clear shortage report. Activation capture is not implemented yet.

Developers can run the tests without a model:

```sh
python -m unittest discover -s tests -v
```

### Worked example

**TODO:** Add a small, complete example demonstrating data preparation, activation capture, probe training, and test evaluation.
