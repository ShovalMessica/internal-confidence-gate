# Simple classification example

This example creates 700 even/odd classification records and an absolute-path
task configuration. From the repository root:

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
`--model`. Model Behavior downloads and runs the selected model. A full probe run
also requires enough correct and incorrect predictions in every split, so this
simple task may be too easy for that stage.

Generated data, configuration, and results stay under `generated/`, which Git
ignores. The toolkit adds the final-answer instruction and `FINAL:` marker; the
system prompt does not include them.
