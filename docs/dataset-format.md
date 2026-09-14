# Dataset format

Provide a `.jsonl` file with one example per line. This is a design draft; **TODO** marks unresolved details.

```text
Example
├── id
├── input
├── target_answer
├── split                         [optional]
└── semantic_positions            [optional]
    ├── position_1
    │   ├── start_char
    │   └── end_char
    └── position_2, ...
```

## Fields

- **`id`** — Unique integer identifier.

- **`input`** — The full prompt sent to the model as a string, including task instructions and example-specific content already inserted. Do not supply a template with unresolved placeholders. The toolkit adds its output instruction and `FINAL:` prefix; you do not add them. See [Generation](configuration.md#generation).

- **`target_answer`** — One expected answer as a string, without `FINAL:`. `UNKNOWN` is reserved for abstention and cannot be a target answer.

  Correctness uses complete-answer matching after ignoring case, trimming surrounding whitespace, and collapsing repeated whitespace. Extra words remain significant.

  Target answers support supervised probe training and evaluation. They never enter the probe as features, and applying a trained probe does not require them.

- **`split` (optional)** — `"train"`, `"validation"`, or `"test"`. Supply it for every example or none.

  If omitted, the toolkit randomly splits 70%/15%/15% using seed 42. Both proportions and seed are configurable.

- **`semantic_positions` (optional)** — Additional input spans for activation capture, using keys `position_1`, `position_2`, etc.

  Each span contains integer `start_char` and `end_char` offsets into the original input: zero-based, start inclusive, end exclusive.

  Every example must supply the same keys and semantic roles, although text and offsets may differ. Locations must be identifiable without the target answer.

  For example, `{"start_char":11,"end_char":15}` selects `Jhon` in `Please ask Jhon.`. Count characters in the decoded input, not its JSON encoding.

## Capture locations

As explained in the [README](../README.md), probes use internal activations to estimate prediction reliability.

Default captures use residual-stream activations across all layers at the last prompt token, the supplied `FINAL:` colon, and the answer position. In direct mode, the first two locations coincide.

Additional semantic positions are encouraged when useful—for example, the name being checked in a name-correction task. Their role stays consistent even when their location changes.

## Validation

- Skip and report malformed records, missing required fields, invalid values, invalid annotations, and `UNKNOWN` targets.
- Keep the first occurrence of an ID; skip later duplicates.
- Reject datasets with partially assigned splits.
- Report excluded records by ID or line number, with a reason.

## Example

```json
{"id":1,"input":"Classify this review as Positive or Negative:\nI loved this product.","target_answer":"Positive"}
```

**TODO:** Unicode counting, token alignment, multi-token captures, exact layer conventions, expected annotation-key detection, validation order, split rounding, minimum usable data, and task-specific answer matching.
