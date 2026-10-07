# Add your own task

This guide walks you through preparing and running your task. Follow it in
order, using the linked references for details:

- **[Dataset format](dataset-format.md):** required fields and validation rules.
- **[Configuration](configuration.md):** settings, defaults, and generation behavior.
- **[Prompt examples](prompt-examples.md):** examples to help design your instructions.

Complete the [installation](../README.md#quick-start) first, then run commands
from the repository root.

Want a ready-made demo instead? See the optional
[synthetic NER example](../examples/ner/README.md). Its generator creates demo
data only; it is not part of preparing your own task.

## 1. Prepare your prompt

Describe the task, expected answer, and any rules or examples the model needs.
Choose one of these two options:

- **Shared instructions in a system prompt**

  Use this when the same instructions apply to every example. Save them in a
  UTF-8 text file, such as `system-prompt.txt`. Both `.txt` and `.md` files are
  accepted; the toolkit reads their contents literally. You can keep it anywhere.
  In [`configs/task.yaml`](../configs/task.yaml), uncomment `system_prompt_path`
  and replace the example path with your file's absolute path.
  [Step 4](#4-fill-in-the-configuration) covers the configuration.

  Contents of `system-prompt.txt`:

  ```text
  Classify the review as POSITIVE or NEGATIVE.
  ```

  Each dataset record's `input` contains only the changing content:

  ```text
  The battery lasts all day.
  ```

  The toolkit sends the file's contents as a **system message** and each `input`
  as a **user message**. [Step 3: Prepare your dataset](#3-prepare-your-dataset)
  explains how to supply that user message in each record.

- **Complete prompt in each input**

  Use this when you already have complete prompts or need different instructions
  per example. Leave `system_prompt_path` commented out in `configs/task.yaml`
  and put everything in each JSONL record's `input`:

  ```text
  Classify the review as POSITIVE or NEGATIVE.

  Review: The battery lasts all day.
  ```

  The toolkit sends this as **one user message**, without a system message.

These examples show two ways to supply the same task.

Both options use the model tokenizer's **chat template** to format the messages
and mark where the assistant's response begins. A compatible chat model is
required, even without a system prompt. You do not add role markers yourself.

Supply finished text without unresolved placeholders or code fences around the
prompt. **Test your instructions on separate development examples, then keep the
instruction template and message structure consistent across training,
validation, and test.**

See [Prompt examples](prompt-examples.md) for more detailed task prompts.

## 2. Reasoning mode

Both modes use this final-answer instruction, supplied by the toolkit:

```text
Answer immediately after FINAL: with only the final answer. Do not repeat the prefix or add reasoning or explanation.
```

In [`configs/task.yaml`](../configs/task.yaml), replace `reasoning_mode: ""`
with `reasoning_mode: "direct"` or `reasoning_mode: "reasoning"`:

- **`direct`:** the toolkit appends the answer instruction to your user message,
  formats the messages, starts the assistant's response with `FINAL:`, then lets
  the model generate the answer.
- **`reasoning`:** the model first generates reasoning. When reasoning ends or
  reaches its token limit, the toolkit appends the final-answer instruction and
  `FINAL:` after it. The model then continues with the answer. This happens
  within the same assistant response, without a new user message.

The toolkit supplies `FINAL:` so it can locate the answer and identify its
tokens for activation capture. **You do not add this marker yourself.**

For example, the toolkit supplies `FINAL:` and the model generates `POSITIVE`,
producing `FINAL: POSITIVE`.

In the same YAML file, `allow_abstention` controls uncertainty in both modes.
Keep `allow_abstention: true` or change it to `allow_abstention: false`:

- **`true` (default):** adds "If you cannot determine the answer, return UNKNOWN."
  Exact `UNKNOWN` responses are reported as abstentions and excluded from probe
  training and gate evaluation. Dataset targets cannot be `UNKNOWN`.
- **`false`:** adds "Provide your best answer. Do not return UNKNOWN."
  If the model still returns `UNKNOWN`, it is compared with the target as an
  ordinary answer.

This setting changes instructions and evaluation; it does not block the model
from generating `UNKNOWN`. Keep your task instructions consistent with it.

See [Generation settings](configuration.md#generation) for the reasoning
instruction and token limits.

## 3. Prepare your dataset

Prepare your examples using the fields and rules in
[Dataset format](dataset-format.md). Save the `.jsonl` file anywhere on your
machine, then set `dataset_path` in [`configs/task.yaml`](../configs/task.yaml)
to its absolute path.

Create a UTF-8 `.jsonl` file with one object per example. Each record needs a
unique integer `id`, an `input`, and one expected `target_answer`.

- **With a system prompt:** `input` contains the example-specific content.
- **Without a system prompt:** `input` contains the full task instructions and
  example-specific content.

For example, with the shared sentiment instructions from step 1:

```json
{"id":1,"input":"The battery lasts all day.","target_answer":"POSITIVE","split":"train"}
```

In your JSONL file, set each record's `split` to `train`, `validation`, or
`test`, or omit `split` from all records for automatic splitting. Keep related
examples in the same split.
See [dataset fields](dataset-format.md#fields),
[validation rules](dataset-format.md#validation), and
[minimum sizes](dataset-format.md#dataset-size) when preparing your records.

With abstention enabled, `UNKNOWN` is reserved for model responses and cannot
be a target.
Targets are never sent to the model or used as probe features. The probe learns
whether a prediction is correct; balanced task answers do not guarantee enough
correct and incorrect predictions.

## 4. Fill in the configuration

Edit [`configs/task.yaml`](../configs/task.yaml). Replace these required fields:

- `model_name_or_path`: a supported Hugging Face model ID or local checkpoint.
- `dataset_path`: the absolute path to your JSONL file.
- `reasoning_mode`: `direct` for an immediate answer, or `reasoning` for reasoning
  followed by an answer.

In that YAML file, uncomment `system_prompt_path` if you use shared instructions
and enter the prompt file's absolute path. To choose a results directory,
uncomment `output_dir` and enter its absolute path. Otherwise, results go under
`outputs/` at the repository root, regardless of where the YAML is saved.

Review the remaining defaults, especially token limits and abstention. See
[Model and paths](configuration.md#model-and-paths) for supported models and
[Generation](configuration.md#generation) for answer settings.
For a local checkpoint, also uncomment `model_revision` in the YAML and enter
a version string that you change whenever the checkpoint files change.

## 5. Check the prompt on development data

Refine your prompt on a small development subset, kept separate from training,
validation, and test. Adjust the instructions, examples, and decision rules
until the model consistently follows the task and answer format. Then keep the
prompt design fixed for the full run.

Copy `configs/task.yaml` to `configs/dev-task.yaml`. In that copy, set
`dataset_path` to the absolute path of your development JSONL file. Keep model
and generation settings consistent with the planned full run.

Validate the files without loading a model or writing files. Results appear only
in the terminal:

```sh
python -m src.run configs/dev-task.yaml --prepare-only
```

Run the model and inspect the first 10 inputs, predictions, targets, and outcomes:

```sh
python -m src.run configs/dev-task.yaml --behavior-only --show-examples 10
```

This command downloads Hub weights if needed and stops after answer evaluation.
After the examples, it prints a final summary and the results folder path.
Both checks accept small samples. Revise your prompt file or dataset inputs and
rerun the behavior command as needed. The toolkit reads your files directly.

## 6. Run Model Behavior on the full dataset

In `configs/task.yaml`, make sure `dataset_path` points to your full JSONL file,
then run:

```sh
python -m src.run configs/task.yaml --prepare-only
python -m src.run configs/task.yaml --behavior-only --show-examples 10
```

The terminal and `run.json` report correct, wrong, abstained, invalid, and
token-limit outputs overall and by split. Check the
[probe-training minimums](dataset-format.md#dataset-size) before proceeding.
Behavior-only runs save results even when those counts are insufficient. See
[Answer evaluation](configuration.md#answer-evaluation) for matching rules and
metric definitions.

## 7. Train and evaluate the gate

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

- **[Custom matching](configuration.md#answer-evaluation):** save your
  `answer_match` function in a Python file. In `configs/task.yaml`, uncomment
  `answer_matcher_path` and set it to that file's absolute path.
- **[Semantic spans](dataset-format.md#capture-locations):** add `semantic_spans`
  to each JSONL record to mark input text for activation capture. Use the same
  span keys and semantic roles across examples.
- **[Custom metrics](configuration.md#custom-task-metrics):** add any needed
  `metadata` to your JSONL records and save `compute_metrics` in a Python file.
  In `configs/task.yaml`, uncomment `custom_metrics_path` and set it to that
  file's absolute path.

For example, this complete record marks `battery` at characters `[4, 11)`:

```json
{"id":1,"input":"The battery lasts all day.","target_answer":"POSITIVE","split":"train","semantic_spans":{"span_1":{"start_char":4,"end_char":11}}}
```
