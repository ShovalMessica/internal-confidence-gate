# Simple classification example

This example creates 2,000 arithmetic truth-classification records and an
absolute-path task configuration. Most examples use large multiplication while
some use easy addition. Labels are balanced within each explicit split, making
the task suitable for demonstrating both correct and incorrect predictions with
a small model.

From the repository root:

```sh
python examples/simple-classification/prepare.py
python -m src.run examples/simple-classification/generated/task.yaml --prepare-only
```

The second command validates the complete input contract without downloading or
loading a model. To generate predictions and inspect Model Behavior:

```sh
python -m src.run examples/simple-classification/generated/task.yaml --behavior-only
```

The default model is `Qwen/Qwen3-0.6B`; select another supported chat model with
`--model`. Model Behavior downloads and runs the selected model. Review the
reported correct and incorrect counts, then run the full pipeline with the same
configuration:

```sh
python -m src.run examples/simple-classification/generated/task.yaml
```

The full command reuses the saved generations, captures activations, trains and
selects probes, evaluates the frozen gate on test data, and writes the report.
Actual behavior remains model-dependent; the runner stops clearly if a selected
model does not produce enough correct and incorrect predictions.

Generated data, configuration, and results stay under `generated/`, which Git
ignores. The toolkit adds the final-answer instruction and `FINAL:` marker; the
system prompt does not include them.

For a guided explanation of every file, command, and result, follow
[Add your own task](../../docs/add-your-own-task.md).
