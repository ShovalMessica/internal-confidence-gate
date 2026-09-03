# Internal Confidence Gate

Task-adaptable toolkit for building confidence gates from internal model signals.

## Research goal

This repository investigates whether signals encoded in a model’s internal activations can distinguish correct from incorrect predictions better than confidence derived from output probabilities alone.

The toolkit provides a shared pipeline that can be applied separately to different tasks. Given model predictions with known correctness labels, it trains a lightweight probe to estimate prediction reliability from internal activations. Once trained, the task-specific gate evaluates completed predictions during inference, supporting an accept-or-reject decision without modifying the underlying model.

## Task scope and requirements

The toolkit is suitable when:

1. The model produces one answer or decision that can be evaluated as a whole (reasoning may precede the answer).
2. That answer can be labeled objectively as **correct or incorrect**.
3. The token span containing the final decision can be identified without using the ground truth.
   - Representations may also be studied at consistently identifiable semantic positions or spans, such as the final prompt token, evidence spans, intermediate reasoning anchors, the last reasoning token, the token immediately preceding the final answer, and the answer tokens themselves. 
5. You have white-box access to the model’s internal activations during generation—typically through the model weights and inference runtime.
6. You have enough labeled predictions from the **same model and task**, including both correct and incorrect cases, to train and evaluate the probe.
7. The same model, response format, and activation-extraction procedure can be used during deployment.

The answer may contain multiple tokens and may come from a fixed set or an open vocabulary. Reasoning may precede the answer, but free-form outputs containing multiple independently evaluated claims are outside the current scope.

### Examples of suitable tasks

- **Classification:** “Is this review positive or negative?” → `Positive`
- **Multiple choice:** “Which option is correct?” → `B`
- **Factual question answering:** “What is the capital of England?” → `London`
- **Structured reasoning:** The model may reason freely, then provide one final decision → `FINAL: 12`

