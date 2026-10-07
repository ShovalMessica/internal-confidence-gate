# Add your own task

Use this guide to run the toolkit on your own dataset. Complete the
[installation](../README.md#quick-start) first, then run commands from the
repository root.

Want a ready-made demo instead? See the optional
[synthetic NER example](../examples/ner/README.md). Its generator creates demo
data only; it is not part of preparing your own task.

## 1. Prepare your prompt

Write instructions that tell the model what to do and what kind of answer to
return. Include the context, decision rules, and examples it needs to understand
your task.

Separate the shared instructions from the content that changes between examples:

- **Shared instructions:** save these in a UTF-8 text file, such as
  `system-prompt.txt`. The toolkit sends this same file as the system message
  for every example. Set its path through `system_prompt_path` in step 3.
- **Example content:** put this in each dataset record's `input` field. The
  toolkit sends it as the user message alongside the shared instructions.

For a sentiment task, `system-prompt.txt` could contain:

```text
Classify the review as POSITIVE or NEGATIVE.
```

One example's `input` would be `The battery lasts all day.` The instructions
stay the same; the review changes for each example.

The prompt file is read as written. Copy only the instructions, without the
surrounding code fences shown above. The toolkit does not fill placeholders.
If you prefer a single message, include the instructions in every `input` and
omit `system_prompt_path`.

**Answer formatting is handled by the toolkit.** It adds an instruction to
return only the final answer, then starts the model's answer with `FINAL:`.
The model generates the answer after that marker, for example `POSITIVE`.
This lets the toolkit locate the answer and its tokens. Do not add the marker
yourself. In reasoning mode, this answer step follows a separate reasoning phase.
See [Generation](configuration.md#generation) for the exact instructions.

See [Prompt examples](prompt-examples.md) for fuller task prompts. Try your
instructions on a separate development sample before the full run, as described
in step 4.

## 2. Prepare your dataset

Create a UTF-8 `.jsonl` file with one object per example. Each record needs a
unique integer `id`, the complete user message in `input`, and one plain
`target_answer`:

```json
{"id":1,"input":"The battery lasts all day.","target_answer":"POSITIVE","split":"train"}
```

Assign every record to `train`, `validation`, or `test`, or omit `split` from
all records for automatic splitting. Keep related examples in the same split.
See [Dataset format](dataset-format.md) for the exact fields and minimum sizes.

With abstention enabled, `UNKNOWN` is reserved for model responses and cannot
be a target. Other task answers, such as `NONE`, participate normally.
Targets are never sent to the model or used as probe features. The probe learns
whether a prediction is correct; balanced task answers do not guarantee enough
correct and incorrect predictions.

## 3. Fill in the configuration

Edit [`configs/task.yaml`](../configs/task.yaml). Replace these required fields:

- `model_name_or_path`: a supported Hugging Face model ID or local checkpoint.
- `dataset_path`: the absolute path to your JSONL file.
- `reasoning_mode`: `direct` for an immediate answer, or `reasoning` for reasoning
  followed by an answer.

If you created a shared prompt file, uncomment `system_prompt_path` and set its
absolute path. Set `output_dir` if you want results in a specific directory;
otherwise they go under `outputs` beside the YAML file.

Review the remaining defaults, especially token limits and abstention. See
[Configuration](configuration.md) for supported models and all settings.
Local checkpoints also require `model_revision`.

## 4. Check the prompt on development data

Use a small, separate development dataset while refining your prompt. Copy
your configuration to `configs/dev-task.yaml` and change its `dataset_path`
to that sample. Keep model and generation settings consistent with the planned
full run.

Validate the files without loading a model:

```sh
python -m src.run configs/dev-task.yaml --prepare-only
```

Run the model and inspect the first 10 inputs, predictions, targets, and outcomes:

```sh
python -m src.run configs/dev-task.yaml --behavior-only --show-examples 10
```

This command downloads Hub weights if needed and stops after answer evaluation.
Both checks accept small samples. Revise your prompt file or dataset inputs and
rerun the behavior command as needed. The toolkit reads your files directly.
Freeze the prompt before using the full dataset, including its test split.

## 5. Run Model Behavior on the full dataset

Use `configs/task.yaml`, pointing to your full dataset:

```sh
python -m src.run configs/task.yaml --prepare-only
python -m src.run configs/task.yaml --behavior-only --show-examples 10
```

The terminal and `run.json` report correct, wrong, abstained, invalid, and
token-limit outputs overall and by split. Check the
[probe-training minimums](dataset-format.md#dataset-size) before proceeding.
Behavior-only runs save results even when those counts are insufficient.

## 6. Train and evaluate the gate

Run the same configuration without a stopping flag:

```sh
python -m src.run configs/task.yaml
```

The toolkit reuses saved generations, captures activations, trains probes on
train, selects a probe and threshold on validation, and compares the frozen
gate with output probability on test. If usable counts are insufficient, it
reports the shortage and stops before probe training.

The printed run directory contains:

```text
run.json
execution.log
generation-manifest.jsonl
evaluations/<evaluation_id>.jsonl
activations.h5
probes.h5
reports/<report_id>/
```

The generation manifest references saved model outputs in the shared
`<output_dir>/.cache/generations` directory. Read
`reports/<report_id>/metrics.json` for exact values and the PNG files for layer
comparisons and TPR-FPR curves. Metric definitions are in
[Configuration](configuration.md#frozen-test-evaluation).

## Optional task-specific behavior

- **Custom matching:** use `answer_matcher_path` when normalized complete-answer
  equality does not fit your task.
- **Semantic spans:** mark additional input text for activation capture. Every
  example must supply the same span keys and semantic roles.
- **Custom metrics:** use `metadata` and `custom_metrics_path` for task-specific
  summaries.

For example, this complete record marks `battery` at characters `[4, 11)`:

```json
{"id":1,"input":"The battery lasts all day.","target_answer":"POSITIVE","split":"train","semantic_spans":{"span_1":{"start_char":4,"end_char":11}}}
```

See [Dataset format](dataset-format.md) and [Configuration](configuration.md)
for the exact interfaces.
