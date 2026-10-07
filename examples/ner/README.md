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
To run its Model Behavior check:

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

## Verified run

The complete 1,000-example pipeline passed on 2026-10-07 with
`Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, using the unchanged example prompt
and seed `91337`. In the generated task YAML, this check set `dtype: "float16"`
and `device: "cuda:0"` for a Quadro RTX 8000. Other settings kept the generated
defaults: direct mode, greedy decoding, four answer tokens, and enabled abstention.
The runtime used PyTorch `2.5.1+cu121` and Transformers `4.51.3`.

Model Behavior produced 627 correct, 324 incorrect, and 49 abstained predictions,
with no invalid outputs. All splits met the training minimums. Activation capture
included `span_1`; all 148 probes trained and the reports were created.
Validation selected `answer_tokens / hidden_state_18` at `target_tpr: 0.90`.

Gate evaluation used 191 concrete test predictions: 123 correct and 68 incorrect.
Both thresholds were selected on validation and applied unchanged to test.

| Method | Test TPR | Test FPR | Balanced accuracy | AUROC |
| --- | ---: | ---: | ---: | ---: |
| Selected probe | 87.8% | 1.5% | 93.2% | 0.9935 |
| Output probability | 86.2% | 83.8% | 51.2% | 0.6031 |

Repeating the command reused every stage without loading the model. These results
verify the synthetic workflow; they do not establish performance on real NER data.
