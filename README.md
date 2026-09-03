# Internal Confidence Gate

Task-adaptable toolkit for building confidence gates from internal model signals.

## Research goal

This repository investigates whether internal model signals can distinguish reliable from unreliable decisions better than model output probability alone.

The toolkit provides a shared pipeline that can be applied separately to different tasks. For each task, it uses model predictions with known outcomes to train a lightweight probe that distinguishes correct from incorrect predictions using internal activations. The resulting gate is task-specific and evaluates new predictions without modifying the underlying model.
