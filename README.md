# Internal Confidence Gate

Task-adaptable toolkit for building confidence gates from internal model activations.

## Research goal

This repository investigates whether signals encoded in a model’s internal activations can distinguish correct from incorrect predictions better than confidence derived from output probabilities alone.

The toolkit provides a shared pipeline that can be applied separately to different tasks. It generates model responses, collects internal activations, and trains a lightweight probe to estimate prediction reliability. The resulting task-specific gate supports an accept-or-reject decision without modifying the underlying model or generating a replacement answer.

The initial scope focuses on activation-based probes. Circuit finding is outside this scope.

## Task scope

The toolkit targets tasks where:

1. The model produces one final answer or decision that can be evaluated as a whole, although reasoning may precede it.
2. The answer can be labeled objectively as **correct or incorrect** by comparison with supplied ground truth.
3. The response follows the toolkit’s fixed final-answer contract, allowing the answer span to be located without using ground truth or correctness labels.

The answer may contain multiple tokens and may come from a fixed set or an open vocabulary. Free-form outputs containing multiple independently evaluated claims are outside the current scope.

### Examples of suitable tasks

- **Classification:** “Is this review positive or negative?” → `FINAL: Positive`
- **Multiple choice:** “Which option is correct?” → `FINAL: B`
- **Factual question answering:** “What is the capital of England?” → `FINAL: London`
- **Structured reasoning:** The model may reason before providing one final decision → `FINAL: 12`

### Response format and abstention

Both reasoning and direct-answer generation use:

```text
FINAL: <answer>
```

Users include the toolkit’s required response instruction in their task prompt. Generated responses are checked against the contract. Formatting failures are reported; answer positions are not guessed.

**TODO:** Define the exact prompt instruction, output grammar, answer boundaries, and handling of noncompliant responses.

Abstention is optional and disabled by default:

```yaml
allow_abstention: false
```

When enabled, the model may output `FINAL: UNKNOWN`. These predictions are reported separately and excluded from probe training and gate TPR/FPR. The prompt instruction must reflect this setting.

## How it works

### Requirements

1. You have white-box access to the model’s internal activations—typically through its weights and inference runtime.
2. You provide task examples with ground-truth answers for training, validation, and testing, prepared in the toolkit’s required dataset format.
3. Your data yields sufficient correct and incorrect model predictions to train and evaluate a probe.
4. The model version, prompt, generation procedure, and activation-extraction method can be kept consistent when applying the trained gate.

**Ground truth supplies correctness labels for supervised probe training and evaluation. It is never an input feature to the probe. Applying a trained probe to new predictions does not require ground truth.**

### Pipeline overview

#### Forward pass and activation capture

1. Run the model on the prepared task inputs.
2. Generate responses and collect selected internal activations and relevant output-token probabilities.
3. Validate the response format and locate the final answer.
4. Compare the extracted answer with ground truth after applying the same normalization to both.

Normalization includes case and surrounding whitespace normalization.

**TODO:** Define the complete normalization rules and output-probability scoring method.

#### Activation positions

The last prompt token is a built-in default.

Users may also annotate positions or spans of interest in the prompt before generation. Generated-answer positions are located through the fixed response contract.

Position selection must not depend on ground truth or correctness labels.

**TODO:** Define the annotation format, supported activation representations, exact token-position conventions, and span-pooling methods.

#### Probe training, validation, and testing

1. Train the probe using captured activations and correctness labels.
2. Use validation data to select probe settings and an acceptance threshold.
3. Evaluate the frozen probe and threshold on test data.
4. Compare the gate with confidence derived from output probabilities.

For eligible predictions:

- **Gate TPR:** accepted correct predictions divided by all correct predictions.
- **Gate FPR:** accepted incorrect predictions divided by all incorrect predictions.

**TODO:** Define split handling, probe configuration, selection procedures, and the complete evaluation report.

#### Using the trained gate

Deployment is the user’s responsibility and is outside the toolkit’s training-and-testing pipeline.

For a new completed prediction, the trained probe uses the required activations to estimate reliability. The frozen threshold determines acceptance or rejection without knowing whether the prediction is correct.

Custom prompt annotations must also be supplied for new inputs when required by the trained probe.

The gate observes the model’s existing computation; it does not modify the model or generate a replacement answer.

## Limitations

- A trained gate is not assumed to generalize across models, tasks, prompts, generation procedures, or data distributions; changes require revalidation.
- Probe performance is correlational and does not establish that the detected representations causally control correctness.
- The gate estimates reliability; it does not verify answers or correct mistakes.

## Getting started

### Prerequisites

**TODO:** Define supported models, runtimes, and hardware requirements.

### Installation

**TODO:** Add installation instructions.

### Prepare a task

#### Model

**TODO:** Define model and generation configuration.

#### Dataset

Users preprocess their raw data into the toolkit’s required format. Dataset records contain example IDs, model inputs, ground-truth answers, and optional prompt-position annotations.

**TODO:** Define the exact schema, input representation, split format, and annotation fields in a dedicated dataset specification.

#### Prompt and response

**TODO:** Provide the exact required instruction and a dedicated specification covering reasoning, direct answers, abstention, parsing, and normalization.

#### Activation capture

**TODO:** Document capture configuration and how prompt annotations and generated-answer spans map to activation features.

### Run the pipeline

**TODO:** Define commands, configuration, and saved outputs for capture, training, validation, and testing.

### Worked example

**TODO:** Add a small, complete example demonstrating the workflow before users adapt their own data.
