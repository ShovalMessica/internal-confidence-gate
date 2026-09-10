# Dataset format

Provide a `.jsonl` file with one JSON object per line. Each object represents one example.

All fields are required unless marked optional. **TODO** indicates an unresolved specification.

## Fields

- **`id`** — A unique example identifier.  
  **TODO:** Accepted ID type.

- **`input`** — The full task prompt as text. The toolkit appends an instruction requiring `FINAL: <answer>` so it can extract the answer and locate its tokens. You do not add this instruction yourself. It also permits `UNKNOWN` when `allow_abstention` is enabled, which is the default.  
  **TODO:** Exact appended instruction, chat-template handling, and response-validation rules.

- **`target_answer`** — The expected answer as text, without `FINAL:`. Compared with the model’s answer after case and surrounding whitespace normalization. Used for supervised probe training and evaluation; never supplied as a probe input. Applying the trained probe does not require it.  
  **TODO:** Further normalization rules, multiple accepted answers, and UNKNOWN target handling.

- **`split` — optional** — `"train"`, `"validation"`, or `"test"`. When omitted across the dataset, splits are assigned randomly.  
  **TODO:** Default proportions, seed configuration, and partially assigned datasets.

- **Additional semantic annotations — optional; field name TODO** — Mark words or passages in `input` for additional activation capture, such as the name being checked in a name-correction task. Encouraged when useful; the same annotation names must appear across all examples. Locations must be identifiable without the target answer. If omitted, only default captures are used.  
  **TODO:** Field name, annotation structure, coordinates, token alignment, and multi-token handling.

## Default captures

The last prompt token and captures from the `FINAL: <answer>` segment require no annotations.

**TODO:** Exact token selection and activation representations.

## Example

```json
{"id":"example_001","input":"Classify this review as Positive or Negative:\nI loved this product.","target_answer":"Positive","split":"train"}
```

**TODO:** Add an annotated example and finalize dataset-validation rules.
