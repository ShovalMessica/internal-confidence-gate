# Name-correction example

This executable example asks a model to identify misspelled or nickname forms
of participant names in synthetic meeting utterances. All people and utterances
are fictional.

The generator uses fixed random seeds to sample names, roster order, speakers,
and utterance context. Clean/corrupted pairs stay in the same split, while task
content is checked for overlap across development, train, validation, and test.

From the repository root:

```sh
python examples/ner/prepare.py
python -m src.run examples/ner/generated/dev-task.yaml --prepare-only
```

The committed `dev-sample.jsonl` is for prompt and output-format checks only.
To run its model behavior check:

```sh
python -m src.run examples/ner/generated/dev-task.yaml --behavior-only
```

After the prompt is stable, use the separately generated full dataset:

```sh
python -m src.run examples/ner/generated/task.yaml --behavior-only
python -m src.run examples/ner/generated/task.yaml
```

The final command reuses saved generations, captures activations, trains and
selects probes, evaluates the frozen gate on test data, and writes the report.
Generated data and outputs stay under `examples/ner/generated/`, which Git
ignores.

Use `--model` to replace the default `Qwen/Qwen3-4B-Instruct-2507`, and `--seed`
to create another deterministic full dataset.

See [Add your own task](../../docs/add-your-own-task.md) for the adaptation
walkthrough and [Dataset format](../../docs/dataset-format.md) for the exact
record contract.
