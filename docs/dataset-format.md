# Dataset format

Provide a UTF-8 `.jsonl` file with one JSON object per line.

```text
Example
├── id
├── input
├── target_answer
├── split                         [optional]
├── metadata                      [optional]
└── semantic_spans                [optional]
    ├── span_1
    │   ├── start_char
    │   └── end_char
    └── span_2, ...
```

## Fields

- **`id`** — Unique integer identifier. Boolean values are not accepted as integers.

- **`input`** — The complete nonempty user message for this example. When an
  optional fixed [`system_prompt_path`](configuration.md#model-and-paths) is
  configured, put shared task instructions in that file and only the
  example-specific user content here. Otherwise, include all task instructions
  directly in every `input`. Do not supply unresolved placeholders. During
  generation, the toolkit adds its output instruction and `FINAL:` prefix. See
  [Generation](configuration.md#generation).

- **`target_answer`** — One expected answer as a nonempty string, without `FINAL:`. Targets containing `FINAL:` (case-insensitive) are rejected. When abstention is enabled, `UNKNOWN` is reserved for model abstention and cannot be a target answer. When abstention is disabled, it is allowed as a normal target.

  Correctness uses complete-answer matching after ignoring case, trimming surrounding whitespace, and collapsing repeated whitespace. Extra words remain significant. Tasks requiring different equivalence rules can configure a custom [`answer_match`](configuration.md#answer-evaluation) function.

  Target answers support supervised probe training and evaluation. They never enter the probe as features, and applying a trained probe does not require them.

- **`split` (optional)** — `"train"`, `"validation"`, or `"test"`. Supply it for every example or none.

  If omitted, the toolkit randomly assigns 70%/15%/15% using seed 42. Both proportions and seed are configurable. Fractions are rounded down, then remaining examples go to the splits with the largest fractional remainders. Ties follow train, validation, test order.

  Automatic splitting treats records independently. If several records come from the same source, conversation, document, or template instance, assign splits yourself so related examples cannot cross split boundaries.

- **`metadata` (optional)** — Free-form JSON object passed to a configured
  [custom metric function](configuration.md#custom-task-metrics). You choose
  its inner field names and values; they may differ between examples. Metadata
  is never sent to the model or used as a probe feature. Omit it or use `{}`
  when no metadata is needed; `null` and non-object values are invalid.

- **`semantic_spans` (optional)** — Additional input spans for activation capture, using keys `span_1`, `span_2`, etc.

  Each span contains integer `start_char` and `end_char` offsets into the original input: zero-based, start inclusive, end exclusive.

  Require `0 <= start_char < end_char <= len(input)`. Boolean offsets are invalid. Keys use positive numbers without leading zeros; numbers need not be consecutive. Omit `semantic_spans` or use `{}` for no additional spans; `null` is invalid.

  Semantic spans are optional. However, **every span you choose to include must be supplied for every example**, with the same key and semantic role. Its text and character offsets may differ between examples. Locations must be identifiable without the target answer.

  For example, `{"start_char":11,"end_char":15}` selects `Jhon` in `Please ask Jhon.`. Count Unicode code points using Python string indexing into the decoded input, not bytes or JSON escape characters. Combining marks count separately. The loader preserves input text without trimming or normalization.

## Capture locations

As explained in the [README](../README.md), probes use internal activations to estimate prediction reliability.

Default capture uses the Hugging Face hidden states at three locations:

- `prompt_end`: the final token of the rendered chat prompt before generation.
- `final_prompt_end`: the final token of the toolkit's injected `FINAL:` marker.
- `answer_tokens`: every generated answer token.

The toolkit saves the embedding output and every returned layer state. When a
final marker is used, the two prompt positions remain distinct. One-token and
multi-token answers use the same tensor structure.

Additional semantic positions are encouraged when useful—for example, the name being checked in a name-correction task. Their role stays consistent even when their location changes.

For each correct or incorrect prediction, the toolkit renders the exact initial chat prompt and maps each character span to every overlapping prompt token. One-token spans retain a token dimension of one; multi-token spans preserve every token. Toolkit instructions and special tokens cannot belong to these spans because their offsets are outside the original `input`.

Semantic spans require a fast Hugging Face tokenizer with character-offset support. Before writing activations, the toolkit verifies that rendering reproduces the saved prompt tokens and that every span maps to at least one token. A mapping failure stops activation capture and reports the example ID and span name.

## Validation

- Ignore extra fields, including extra properties inside individual spans;
  retain only recognized fields. Preserve `metadata` exactly.
- Skip malformed JSON, blank lines, non-object records, duplicate JSON keys, missing required fields, invalid values or spans, and normalized `UNKNOWN` targets when abstention is enabled. Whitespace-only input or target text is invalid.
- The first occurrence of an integer ID reserves it, even if that record is invalid. Skip later duplicates.
- After excluding invalid records, reject datasets with partially assigned splits or differing semantic-span key sets. The loader checks keys and offsets; users are responsible for semantic meaning and target independence.
- Stop if the file cannot be read as UTF-8 or no valid examples remain.
- Report exclusions with a one-based line number, integer ID when available (`null` otherwise), and reasons.

## Loading and splitting

From Python, with the repository root as the working directory:

```python
from src.dataset import assign_splits, load_dataset

result = load_dataset(config.dataset_path, allow_abstention=config.allow_abstention)
result = assign_splits(result, config.split_ratios, config.split_seed)
```

`DatasetResult.examples` contains valid records in file order. `DatasetResult.excluded` contains dictionaries with `line`, `id`, and `reasons`. File-level, consistency, or split-readiness failures raise `DatasetError`, whose `errors` and `excluded` attributes preserve the failure details and exclusions collected so far.

Full probe runs require at least 700 valid examples. Each resulting split must contain at least 200 train, 100 validation, and 100 test examples. User-supplied splits are preserved but must meet the same requirements. `--prepare-only` and `--behavior-only` defer these minimums so small prompt-development samples can be checked.

After generation, the toolkit requires at least 100 correct and 100 incorrect usable predictions in train, and 50 correct and 50 incorrect in both validation and test. Invalid and `UNKNOWN` predictions do not count. Evaluation results are saved before a shortage stops the runner.

These loading functions do not run a model, print messages, or save files. The runner coordinates generation and evaluation separately.

## Example

```json
{"id":1,"input":"Classify this review as Positive or Negative:\nI loved this product.","target_answer":"Positive","metadata":{"source":"customer_reviews"}}
```

## Adding examples later

Append records with new unique IDs while preserving the same field and semantic-span rules, then rerun the same configuration. With the same `output_dir`, model, and generation settings, unchanged ID-and-input pairs reuse saved generations and copy compatible activations into the new run's single activation file. Changing only semantic-span offsets reuses generation but recaptures the affected activations. New IDs or changed inputs run through the model; target-answer changes only rerun evaluation.

Automatic splitting is recalculated for the new dataset. Supply explicit splits on every record when existing split assignments must remain fixed.
