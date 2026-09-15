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

The planned protocol has the toolkit supply both the output instruction and the `FINAL:` marker. The model generates the answer after the marker. Users do not need to add either to their dataset prompts.

In direct-answer mode, the toolkit supplies the marker before generation. In reasoning mode, it first lets the model reason, closes reasoning if needed, then inserts the answer instruction and marker before resuming generation. Reasoning and answer generation have separate token limits.

See [Generation](docs/configuration.md#generation) for exact injected instructions, answer extraction, validation, and remaining implementation TODOs. The injected boundary is known; answer-format failures are reported rather than repaired or silently dropped.

The model may decline to answer using `FINAL: UNKNOWN`. This behavior is called **abstention** and is enabled by default:

```yaml
allow_abstention: true
```

This is a shared run setting, not a dataset field. Users can disable it. The toolkit’s appended instruction reflects the setting.

When abstention is enabled, UNKNOWN predictions are reported separately and excluded from probe training and gate TPR/FPR.

**TODO:** Define where shared run settings are supplied and how UNKNOWN target answers are handled.

## Pipeline overview

1. **Prepare data:** provide complete task prompts and target answers in the required dataset format. Optionally assign splits and annotate additional semantic positions.
2. **Run forward passes and capture:** generate responses while collecting selected activations and relevant output-token probabilities.
3. **Evaluate predictions:** validate response formatting, extract answers, and compare them with target answers after normalization.
4. **Train:** fit probes using training activations and prediction correctness.
5. **Validate:** select probe settings and an acceptance threshold using validation data.
6. **Test:** evaluate the frozen probe and threshold on test data and compare performance with output-probability confidence.

Answer normalization includes case and surrounding whitespace normalization.

For eligible predictions:

- **Gate TPR:** accepted correct predictions divided by all correct predictions.
- **Gate FPR:** accepted incorrect predictions divided by all incorrect predictions.

**TODO:** Finalize normalization, output-probability scoring, probe configuration, selection procedures, and the evaluation report.

## Requirements and limitations

- White-box access to the model’s internal activations is required.
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

**TODO:** Define supported models, runtimes, hardware requirements, and installation steps.

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

**TODO:** Define shared model, generation, capture, and training settings, along with pipeline commands and saved outputs.

### Worked example

**TODO:** Add a small, complete example demonstrating data preparation, activation capture, probe training, and test evaluation.
