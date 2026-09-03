# internal-confidence-gate
Task-agnostic toolkit for building confidence gates from internal model signals.

## Research goal

This repository investigates whether internal model signals can distinguish reliable from unreliable decisions better than the model’s output probability alone. It provides a general pipeline for building activation-based confidence gates: using model predictions with known outcomes, the toolkit trains a lightweight linear classifier to distinguish correct from incorrect predictions based on the model’s internal activations. The trained gate can then assess new predictions at inference time without modifying the underlying model.

