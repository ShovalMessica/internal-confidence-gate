# Add your own task

This walkthrough uses the executable arithmetic-classification example to take
one task from prompt design through a trained confidence gate and final report.
The exact dataset and configuration contracts remain in
[Dataset format](dataset-format.md) and [Configuration](configuration.md).

## 1. Define one objectively scored answer

Each example must have one final answer that can be marked correct or incorrect.
The example asks whether an arithmetic equality is true and uses `TRUE` or
`FALSE` as its complete answer.

Shared task instructions live in
[`system-prompt.md`](../examples/simple-classification/system-prompt.md):

```text
Decide whether the supplied arithmetic equality is true.

Return TRUE if it is correct or FALSE if it is incorrect. The toolkit supplies the required final-answer format.
```

Do not include `FINAL:` in this prompt. The toolkit adds its answer instruction,
opens the assistant response, and supplies the marker automatically.

Before building a large dataset, test prompt wording and any demonstrations on a
small development sample. Keep the final prompt fixed once data collection and
gate evaluation begin.

## 2. Create the JSONL dataset

Each line supplies a unique integer ID, the complete example-specific user
message, its target answer, and optionally its split:

```json
{"id":0,"input":"Is this arithmetic equality true or false?\n100003 * 100019 = 10002200057","target_answer":"TRUE","split":"train"}
```

The target answer is used only to evaluate predictions and train the reliability
probe. It is never sent to the model or used as a probe feature.

Generate the complete example from the repository root:

```sh
python examples/simple-classification/prepare.py
```

This writes `dataset.jsonl` and `task.yaml` under the Git-ignored `generated/`
directory. The default 2,000 examples include balanced labels and explicit
train, validation, and test splits. Most equalities are deliberately difficult,
so a small model is likely to produce both correct and incorrect predictions.

## 3. Review the generated configuration

The generated YAML contains the model, absolute data and output paths, direct
generation mode, and a short answer limit:

```yaml
model_name_or_path: Qwen/Qwen3-0.6B
dataset_path: <absolute path>/dataset.jsonl
system_prompt_path: <absolute path>/system-prompt.md
reasoning_mode: direct
decoding_strategy: greedy
answer_max_new_tokens: 4
allow_abstention: false
output_dir: <absolute path>/outputs
```

Choose another supported chat model with:

```sh
python examples/simple-classification/prepare.py --model <hugging-face-model-id>
```

## 4. Validate before loading a model

Run preparation first:

```sh
python -m src.run examples/simple-classification/generated/task.yaml --prepare-only
```

Check the valid and excluded record counts, split source, and split sizes. This
command creates the run record but does not download or load a model. Small
datasets are accepted in this mode while a prompt is still being developed.

## 5. Inspect Model Behavior

Generate and evaluate answers without collecting activations:

```sh
python -m src.run examples/simple-classification/generated/task.yaml --behavior-only
```

The Model Behavior table reports correct, wrong, abstained, invalid, and
token-limit outputs overall and by split. Inspect `execution.log` and `run.json`
under the printed run directory. If formatting failures are common, revise the
prompt and repeat the smoke check before training probes.

The full pipeline requires enough correct and incorrect concrete predictions in
every split. The runner reports any shortage precisely. `UNKNOWN` abstentions and
invalid outputs do not count toward these classes.

## 6. Train and evaluate the gate

Once behavior is suitable, run the same configuration without a stopping flag:

```sh
python -m src.run examples/simple-classification/generated/task.yaml
```

Saved generations are reused. The runner then:

1. Captures hidden states at `prompt_end`, `final_prompt_end`, and
   `answer_tokens`.
2. Trains a linear probe for every configured position and model state.
3. Selects a probe and threshold on validation data at `target_tpr`.
4. Applies that frozen choice to test data and compares it with raw output
   probability.
5. Creates machine-readable metrics, CSV plot data, and PNG figures.

## 7. Interpret the report

Open `<run_dir>/reports/<report_id>/metrics.json` and the generated figures.

- **TPR** is the fraction of correct predictions accepted.
- **FPR** is the fraction of incorrect predictions accepted.
- **Balanced accuracy** averages TPR and `1 - FPR`.
- **AUROC** measures score ranking across all thresholds.
- **Coverage** is the accepted fraction of eligible concrete predictions.
- **Accepted-error rate** is the incorrect fraction among accepted predictions.

Validation chooses the representation and thresholds. Test data only evaluates
the frozen choices. Invalid outputs and enabled `UNKNOWN` abstentions are
excluded from gate metrics. Any other task label, including `NONE`, is a normal
concrete answer and participates in probing.

## Optional task-specific behavior

Start with the default complete-answer matcher and default capture positions.
Add customization only when the task needs it:

- Use [`answer_matcher_path`](configuration.md#answer-evaluation) for aliases,
  punctuation rules, or other task-specific answer equivalence.
- Add [`semantic_spans`](dataset-format.md#capture-locations) when a meaningful
  input region should be compared with the default positions.
- Use [`custom_metrics_path`](configuration.md#custom-task-metrics) for
  task-specific behavior summaries. These metrics do not change correctness or
  probe labels.

Changing a prompt or input creates new generation work. Changing only matching,
metrics, or downstream probe settings reuses compatible upstream artifacts as
described in [Run identity and reuse](configuration.md#run-identity-and-reuse).
