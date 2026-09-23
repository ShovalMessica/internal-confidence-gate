# Dataset format

Provide a UTF-8 `.jsonl` file with one JSON object per line. Loading and validation are implemented; later pipeline steps remain under development. **TODO** marks unresolved details.

```text
Example
├── id
├── input
├── target_answer
├── split                         [optional]
└── semantic_spans            [optional]
    ├── span_1
    │   ├── start_char
    │   └── end_char
    └── span_2, ...
```

## Fields

- **`id`** — Unique integer identifier. Boolean values are not accepted as integers.

- **`input`** — The full prompt as a nonempty string, including task instructions and example-specific content already inserted. Do not supply a template with unresolved placeholders. The planned generation stage adds its output instruction and `FINAL:` prefix; you do not add them. See [Generation](configuration.md#generation).

- **`target_answer`** — One expected answer as a nonempty string, without `FINAL:`. Targets containing `FINAL:` (case-insensitive) are rejected. When abstention is enabled, `UNKNOWN` is reserved for model abstention and cannot be a target answer. When abstention is disabled, it is allowed as a normal target.

  Correctness uses complete-answer matching after ignoring case, trimming surrounding whitespace, and collapsing repeated whitespace. Extra words remain significant.

  Target answers support supervised probe training and evaluation. They never enter the probe as features, and applying a trained probe does not require them.

- **`split` (optional)** — `"train"`, `"validation"`, or `"test"`. Supply it for every example or none.

  If omitted, the toolkit randomly assigns 70%/15%/15% using seed 42. Both proportions and seed are configurable. Fractions are rounded down, then remaining examples go to the splits with the largest fractional remainders. Ties follow train, validation, test order.

- **`semantic_spans` (optional)** — Additional input spans for activation capture, using keys `span_1`, `span_2`, etc.

  Each span contains integer `start_char` and `end_char` offsets into the original input: zero-based, start inclusive, end exclusive.

  Require `0 <= start_char < end_char <= len(input)`. Boolean offsets are invalid. Keys use positive numbers without leading zeros; numbers need not be consecutive. Omit `semantic_spans` or use `{}` for no additional spans; `null` is invalid.

  Semantic spans are optional. However, **every span you choose to include must be supplied for every example**, with the same key and semantic role. Its text and character offsets may differ between examples. Locations must be identifiable without the target answer.

  For example, `{"start_char":11,"end_char":15}` selects `Jhon` in `Please ask Jhon.`. Count Unicode code points using Python string indexing into the decoded input, not bytes or JSON escape characters. Combining marks count separately. The loader preserves input text without trimming or normalization.

## Capture locations

As explained in the [README](../README.md), probes use internal activations to estimate prediction reliability.

Default captures use residual-stream activations across all layers at the last prompt token, the supplied `FINAL:` colon, and the answer position. In direct mode, the first two locations coincide.

Additional semantic positions are encouraged when useful—for example, the name being checked in a name-correction task. Their role stays consistent even when their location changes.

## Validation

- Ignore extra fields, including extra properties inside individual spans; retain only recognized fields.
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

Splitting requires at least 700 valid examples. Each resulting split must contain at least 200 train, 100 validation, and 100 test examples. User-supplied splits are preserved but must meet the same requirements.

These are pre-generation checks. After generation, the toolkit will require at least 100 correct and 100 incorrect usable predictions in train, and 50 correct and 50 incorrect in both validation and test. Invalid and `UNKNOWN` predictions do not count. This later check is not implemented yet.

These functions do not run a model, print messages, or save files. The runner displays their results; saving reports is planned for a later stage.

## Example

```json
{"id":1,"input":"Classify this review as Positive or Negative:\nI loved this product.","target_answer":"Positive"}
```

**TODO:** Token alignment, multi-token captures, exact layer conventions, post-generation minimum enforcement, and task-specific answer matching.
