# Internal Confidence Gate

Task-adaptable toolkit for building confidence gates from internal model signals.

## Research goal

This repository investigates whether signals encoded in a model’s internal activations can distinguish correct from incorrect predictions better than confidence derived from output probabilities alone.

The toolkit provides a shared pipeline that can be applied separately to different tasks. Given model predictions with known correctness labels, it trains a lightweight probe to estimate prediction reliability from internal activations. Once trained, the task-specific gate evaluates completed predictions during inference, supporting an accept-or-reject decision without modifying the underlying model.

## Task scope

The toolkit currently targets tasks where:

1. The model produces one answer or decision that can be evaluated as a whole, although reasoning may precede it.
2. The answer can be labeled objectively as **correct or incorrect**.
3. The token span containing the final decision can be located using a task-defined rule that does not depend on the ground truth or correctness label.

The answer may contain multiple tokens and may come from a fixed set or an open vocabulary. Free-form outputs containing multiple independently evaluated claims are outside the current scope.

### Examples of suitable tasks

- **Classification:** “Is this review\( \text { review positive or negative?} \)” → `Positive`
- **Multiple choice:** “Which option is correct?” → `B`
- **Factual question answering:** “What is the capital of England?” → `London`
- **Structured reasoning:** The model may reason freely, then provide one final decision → `FINAL: 12`

## How it works

### Requirements

1. You have white-box access to the model’s internal activations—typically through its weights and inference runtime.
2. You have enough task examples with objective ground truth for the target model to produce sufficient correct and incorrect predictions.
3. The model version, response format, inference procedure, and activation-extraction method can be kept consistent when deploying the gate.

### Training

1. Run the model on task examples with known ground-truth answers.
2. Record each generated response, the relevant output-token probabilities, and selected internal activations.
3. Identify the final-answer span and label the prediction as correct or incorrect.
4. Train and validate a task-specific probe using the recorded activations.
5. Select an acceptance threshold using development data and compare the gate with output probability alone.

Beyond the final decision, representations may also be studied at consistently identifiable semantic positions or spans, such as the final prompt token, evidence spans, intermediate reasoning anchors, the last reasoning token, the token immediately preceding the final answer, and the answer tokens themselves.

### Inference

1. The model generates a new prediction while the required activations are captured.
2. The trained gate converts those activations into a reliability score.
3. The frozen threshold determines whether the completed prediction is accepted or rejected.

The gate observes the model’s existing computation; it does not modify the model or generate a replacement answer.

## Limitations

- A trained gate is not assumed to generalize across models, tasks, prompts, or data distributions; such changes require revalidation.
- Probe performance is correlational and does not establish that the detected representations causally control correctness.
