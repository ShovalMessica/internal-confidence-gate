# Dataset format

Provide a UTF-8 `.jsonl` file with one JSON object per line.
The fields below belong in each JSON object. Save the file anywhere and set
`dataset_path` in [`configs/task.yaml`](../configs/task.yaml) to its absolute path.

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

- **`id`** - Unique integer identifier. Boolean values are not accepted as integers.

- **`input`** - The complete nonempty user message for this example. When an
  [`system_prompt_path`](configuration.md#model-and-paths) is set in
  `configs/task.yaml`, put only example-specific content here. Otherwise, include the
  task instructions in every `input`. Do not leave unresolved placeholders or
  add `FINAL:`; the toolkit adds its own output instruction and marker.

- **`target_answer`** - One expected answer as a nonempty string, without `FINAL:`. Targets containing `FINAL:` (case-insensitive) are rejected. When abstention is enabled, `UNKNOWN` is reserved for model abstention and cannot be a target answer. When abstention is disabled, it is allowed as a normal target.

  Correctness ignores case, surrounding whitespace, and repeated internal
  whitespace, but keeps punctuation and extra words. To
  [customize answer matching](configuration.md#customize-answer-matching), save
  `answer_match` in a Python file
  and set `answer_matcher_path` in `configs/task.yaml` to its absolute path.
  Targets create probe labels; they are never probe features.

- **`split` (optional)** - `"train"`, `"validation"`, or `"test"`. Supply it for every example or none.

  If omitted, the toolkit assigns reproducible 70%/15%/15% splits. Change
  `split_ratios` and `split_seed` in `configs/task.yaml` to use other settings.

  Automatic splitting treats records independently. If several records come from the same source, conversation, document, or template instance, assign splits yourself so related examples cannot cross split boundaries.

- **`metadata` (optional)** - Free-form JSON object passed to your metric function
  when you [customize task metrics](configuration.md#customize-task-metrics). You choose
  its inner field names and values; they may differ between examples. Metadata
  is never sent to the model or used as a probe feature. Omit it or use `{}`
  when no metadata is needed; `null` and non-object values are invalid.

- **`semantic_spans` (optional)** - Additional input spans for activation capture, using keys `span_1`, `span_2`, etc.

  Each span contains integer `start_char` and `end_char` offsets into the original input: zero-based, start inclusive, end exclusive.

  Require `0 <= start_char < end_char <= len(input)`. Boolean offsets are invalid. Keys use positive numbers without leading zeros; numbers need not be consecutive. Omit `semantic_spans` or use `{}` for no additional spans; `null` is invalid.

  Semantic spans are optional. However, **every span you choose to include must be supplied for every example**, with the same key and semantic role. Its text and character offsets may differ between examples. Locations must be identifiable without the target answer.

  For example, `{"start_char":11,"end_char":15}` selects `Jhon` in `Please ask Jhon.`. Count Unicode code points using Python string indexing into the decoded input, not bytes or JSON escape characters. Combining marks count separately. The loader preserves input text without trimming or normalization.

## Semantic spans

As explained in the [README](../README.md), probes use internal activations to estimate prediction reliability.

Customize semantic spans by adding `semantic_spans` to each JSONL record. They
mark task-specific text in `input`, such as the name being checked in a
correction task. The role stays fixed even when the text and location change.

The toolkit maps each character span to every overlapping prompt token.
One-token and multi-token spans keep the same tensor structure. Spans cannot
include toolkit instructions or special tokens because they refer only to
`input`.

Semantic spans require a fast Hugging Face tokenizer with character-offset support. Before writing activations, the toolkit verifies that rendering reproduces the saved prompt tokens and that every span maps to at least one token. A mapping failure stops activation capture and reports the example ID and span name.

## Default capture positions

Without annotations, the toolkit captures Hugging Face hidden states at:

- `prompt_end`: the final token of the rendered chat prompt before generation.
- `final_prompt_end`: the final token of the toolkit's injected `FINAL:` marker.
- `answer_tokens`: every generated answer token.

The toolkit saves the embedding output and every returned layer state for
default capture positions and semantic spans. The two prompt positions remain
distinct. One-token and multi-token answers use the same tensor structure.

## Validation

- Ignore extra fields, including extra properties inside individual spans;
  retain only recognized fields. Preserve `metadata` exactly.
- Skip malformed JSON, blank lines, non-object records, duplicate JSON keys, missing required fields, invalid values or spans, and normalized `UNKNOWN` targets when abstention is enabled. Whitespace-only input or target text is invalid.
- The first occurrence of an integer ID reserves it, even if that record is invalid. Skip later duplicates.
- After excluding invalid records, reject datasets with partially assigned splits or differing semantic-span key sets. The loader checks keys and offsets; users are responsible for semantic meaning and target independence.
- Stop if the file cannot be read as UTF-8 or no valid examples remain.
- Report exclusions with a one-based line number, integer ID when available (`null` otherwise), and reasons.

## Dataset size

Full probe runs require at least 700 valid examples. Each resulting split must contain at least 200 train, 100 validation, and 100 test examples. User-supplied splits are preserved but must meet the same requirements. `--prepare-only` and `--behavior-only` defer these minimums so small prompt-development samples can be checked.

After generation, the toolkit requires at least 100 correct and 100 incorrect usable predictions in train, and 50 correct and 50 incorrect in both validation and test. Invalid outputs do not count. `UNKNOWN` predictions are excluded only when `allow_abstention: true`; otherwise, they count as ordinary correct or incorrect answers. Evaluation results are saved before a shortage stops the runner.

## Example

```json
{"id":1,"input":"Classify this review as Positive or Negative:\nI loved this product.","target_answer":"Positive","metadata":{"source":"customer_reviews"}}
```

## Adding examples later

Append records with new IDs and rerun the same configuration. Unchanged IDs and
inputs reuse saved generations. New or changed inputs run through the model;
target changes rerun evaluation.

Automatic splitting is recalculated for the new dataset. Set `split` in every
JSONL record when existing split assignments must remain fixed.
