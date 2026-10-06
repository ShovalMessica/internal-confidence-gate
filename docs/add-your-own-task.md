# Add your own task

This walkthrough uses the executable synthetic
[name-correction example](../examples/ner/README.md). It first checks a prompt
on a small development sample, then runs a separate full dataset through probe
training and reporting.
The synthetic task demonstrates the workflow rather than supporting a research
claim about gate quality.

Install the repository dependencies first, and run every command below from
the repository root. `--prepare-only` validates files without loading a model.
`--behavior-only` loads and runs the model, downloading Hub weights when they
are not already available, then stops after answer evaluation.

Use [Dataset format](dataset-format.md) and
[Configuration](configuration.md) as the exact references.

## 1. Define the task

Each example needs one final answer that can be evaluated as correct or
incorrect. In the example, dataset targets are a participant label (`A`-`J`) or
`NONE`. With abstention enabled, `UNKNOWN` is reserved for a model that cannot
decide and cannot be a dataset target.

Fixed rules and demonstrations are in
[`system-prompt.md`](../examples/ner/system-prompt.md). It uses one text code
block so GitHub displays the exact prompt literally; `prepare.py` extracts that
block without its fences for the model. Each dataset `input`
contains the participant list and utterance for one example. Do not include
`FINAL:` in either place; the toolkit supplies its answer instruction and marker.

## 2. Check the prompt on development data

The repository includes a small [`dev-sample.jsonl`](../examples/ner/dev-sample.jsonl)
for prompt development. Generate configurations and the separate full dataset:

```sh
python examples/ner/prepare.py --model Qwen/Qwen3-4B-Instruct-2507 --seed 91337 --examples 1000
```

Validate the development configuration without loading a model:

```sh
python -m src.run examples/ner/generated/dev-task.yaml --prepare-only
```

Then inspect the first 20 inputs, raw model answers, targets, and outcomes:

```sh
python -m src.run examples/ner/generated/dev-task.yaml --behavior-only --show-examples 20
```

Revise the task instructions, demonstrations, or decision rules if the model
misunderstands the task or output contract. After editing `system-prompt.md`,
rerun the same `prepare.py` command—with the same model, seed, and example
count—before the next behavior check. This regenerates the text file referenced
by the YAML. The development sample is separate from the final test split. Once
behavior is satisfactory, freeze the prompt before the full run.

## 3. Understand one record

A record contains a unique integer ID, the complete example-specific user
message, its target answer, its split, and optional annotations:

```json
{"id":1000000,"input":"<PARTICIPANTS>\nA William Gonzalez\nB Taylor Allen\nC Thomas Moore\nD Christopher Lopez\nE Isabella Jones\nF Olivia Harris\nG Knox Smith\nH Kenneth Scott\nI Edward Thompson\nJ Teresa Thomas\n</PARTICIPANTS>\n\n<MEETING_TRANSCRIPT>\n<846479><Speaker 14>We still need feedback from Will about hiring.\n</MEETING_TRANSCRIPT>","target_answer":"A","split":"train","semantic_spans":{"span_1":{"start_char":266,"end_char":270}},"metadata":{"case_family":"dev_00000","example_type":"corrupted"}}
```

Here `span_1` selects characters `[266, 270)`, the complete substring `Will`.
The span marks an additional task-specific activation location; it does not
reveal whether the model's answer is correct.

The target answer is never sent to the model or used as a probe feature. It is
used only to decide whether the saved model prediction is correct.

Do not confuse **task answers** with **probe labels**. `A` and `NONE` are task
answers. The probe label is whether the model's answer matched the target:
`correct` or `incorrect`. Balancing task answers does not guarantee enough
correct and incorrect model predictions for probe training.

## 4. Review the configuration

`prepare.py` writes `generated/task.yaml` with absolute paths and these central
settings:

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
dataset_path: <absolute path>/dataset.jsonl
system_prompt_path: <absolute path>/generated/system-prompt.txt
reasoning_mode: direct
decoding_strategy: greedy
answer_max_new_tokens: 4
allow_abstention: true
output_dir: <absolute path>/outputs
```

The generated dataset has 1,000 examples with fixed 600/200/200 train,
validation, and test splits. Seeded sampling varies names, participant order,
speaker metadata, and utterance context. Each clean/corrupted pair remains in
one split, and generation stops if task content overlaps development or another
split.

## 5. Run Model Behavior

Run generation and correctness evaluation before spending time on activation
capture:

```sh
python -m src.run examples/ner/generated/task.yaml --behavior-only
```

The terminal and `run.json` report correct, wrong, abstained, invalid, and
token-limit outputs overall and by split. The full pipeline requires enough
correct and incorrect concrete predictions in every split. The runner reports
the exact shortage when this requirement is not met.

## 6. Train and evaluate the gate

Run the same configuration without a stopping flag:

```sh
python -m src.run examples/ner/generated/task.yaml
```

The toolkit reuses saved generations, then captures activations, trains probe
candidates on train, chooses a probe and threshold on validation, and compares
the frozen probe with output probability on test.

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

Read `reports/<report_id>/metrics.json` for exact values and the PNG files for
layer comparisons and TPR-FPR curves. The main gate metrics are:

- **TPR:** fraction of correct predictions accepted.
- **FPR:** fraction of incorrect predictions accepted.
- **Balanced accuracy:** `(TPR + 1 - FPR) / 2`.
- **AUROC:** ranking quality across all thresholds.
- **Coverage:** fraction of eligible concrete predictions accepted.
- **Accepted-error rate:** fraction of accepted predictions that are wrong.

Invalid outputs and enabled `UNKNOWN` abstentions are outside the gate's
eligible population.

## 7. Adapt the files to another task

Replace these parts:

1. **Shared instructions:** rewrite the text block in `system-prompt.md` with the task, valid
   answers, decision rules, and useful demonstrations.
2. **Example inputs:** put each complete example-specific user message in
   `input`.
3. **Target answers:** supply the expected final answer for every record.
4. **Splits:** assign train, validation, and test, or omit every assignment and
   let the toolkit split reproducibly.
5. **Configuration:** choose the model, direct or reasoning mode, token limits,
   abstention behavior, and output directory.

For example, a sentiment task could replace the prompt with “Classify the review
as POSITIVE or NEGATIVE” and use records such as:

```json
{"id":1,"input":"Review: The battery lasts all day.","target_answer":"POSITIVE","split":"train"}
```

The pipeline and probe labels remain unchanged.

## Optional task-specific behavior

Use these only when the default contract is insufficient:

- **Custom answer matching:** configure `answer_matcher_path` when normalized
  complete-answer equality is not the right correctness rule.
- **Semantic spans:** add `semantic_spans` to every example to capture an
  additional role-specific input location. The NER generator marks the name
  mention as `span_1`.
- **Custom metrics:** attach free-form `metadata` and configure
  `custom_metrics_path` for task-specific count-based summaries.

Their exact interfaces, validation rules, and reuse behavior are documented in
[Configuration](configuration.md) and [Dataset format](dataset-format.md).

## Reproducibility

Keep the model revision, prompt, dataset, generation settings, and split
assignments fixed for the final run. The toolkit records hashes and provenance,
resumes interrupted stages, and reuses validated artifacts. A changed prompt or
dataset creates a distinct run rather than silently overwriting the earlier one.
