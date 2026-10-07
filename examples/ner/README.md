# Name-correction example

This executable example asks a model to identify misspelled or nickname forms
of participant names in synthetic meeting utterances. All people and utterances
are fictional.
It demonstrates the toolkit workflow; it is not a benchmark or research result.

The generator uses fixed random seeds to sample names, roster order, speakers,
and utterance context. Clean/corrupted pairs stay in the same split, while task
content is checked for overlap across development, train, validation, and test.
The Markdown prompt source displays as literal text on GitHub; preparation
extracts it without the surrounding code fence before model use.

From the repository root:

```sh
python examples/ner/prepare.py --model Qwen/Qwen3-4B-Instruct-2507 --seed 91337 --examples 1000
python -m src.run examples/ner/generated/dev-task.yaml --prepare-only
```

The committed `dev-sample.jsonl` is for prompt and output-format checks only.
To run its model behavior check:

```sh
python -m src.run examples/ner/generated/dev-task.yaml --behavior-only --show-examples 20
```

After editing `examples/ner/system-prompt.md`, rerun the same `prepare.py`
command before the next behavior check so
`examples/ner/generated/system-prompt.txt` is updated.

After the prompt is stable, use the separately generated full dataset:

```sh
python -m src.run examples/ner/generated/task.yaml --behavior-only
python -m src.run examples/ner/generated/task.yaml
```

The final command reuses saved generations, captures activations, trains and
selects probes, evaluates the gate on test data, and writes the report.
Generated data and outputs stay under `examples/ner/generated/`, which Git
ignores.

In the `python examples/ner/prepare.py` command above, change `--model` to
choose another model and `--seed` to generate a different dataset. To change
runner settings, edit `examples/ner/generated/dev-task.yaml` or
`examples/ner/generated/task.yaml`. Rerunning `prepare.py` rewrites these files.

See [Add your own task](../../docs/add-your-own-task.md) for the adaptation
walkthrough and [Dataset format](../../docs/dataset-format.md) for the exact
record contract.
