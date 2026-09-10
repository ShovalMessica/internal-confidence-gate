# Dataset format

Provide a `.jsonl` file with one JSON object per line. Each object represents one example.

```text
dataset.jsonl
└── Example (one per line)
    ├── id
    ├── input
    ├── target_answer
    ├── split                          [optional]
    └── semantic_positions             [optional]
        ├── position_1
        │   ├── start_char
        │   └── end_char
        ├── position_2
        │   ├── start_char
        │   └── end_char
        └── ...
```

All fields are required unless marked optional. **TODO** indicates an unresolved specification.

## Fields

- **`id`** — A unique example identifier. IDs must be unique across the dataset.  
  **TODO:** Accepted ID type.

- **`input`** — The complete task prompt as text, including instructions and example-specific content. The toolkit appends an instruction requiring `FINAL: <answer>` so it can extract the answer and locate its tokens. You do not add this instruction yourself.  
  **TODO:** Exact appended instruction, chat-template handling, and response-validation rules.

- **`target_answer`** — The expected answer as text, without `FINAL:`. The toolkit compares it with the model’s answer after normalizing case and surrounding whitespace. This determines correctness for supervised probe training and evaluation. The target answer is never supplied as a probe input or inserted into the prompt by the toolkit. Applying the trained probe does not require it.  
  **TODO:** Further normalization rules and support for multiple accepted answers.

- **`split` (optional)** — `"train"`, `"validation"`, or `"test"`. When omitted across the dataset, splits are assigned randomly.  
  **TODO:** Default proportions, reproducibility, and handling of partially assigned datasets.

- **`semantic_positions` (optional)** — Additional task-specific text spans where the toolkit should collect activations. If omitted across the dataset, only the default positions are used. The annotation format is explained below.

## Semantic positions

As explained in the [main README](../README.md), the probe estimates prediction reliability from internal activations rather than output probabilities alone.

The model produces activations across many layers and token positions. We focus on selected locations that may contain informative signals and compare probes trained on their activations. Validation data determines which probes work best; test data evaluates the frozen selection.

**Default positions**

The toolkit captures residual-stream activations across all layers at:

- The last prompt token, before response generation.
- The token containing the colon in the final `FINAL:` marker.
- The answer token following that marker.

These positions are located automatically and require no annotations.

**TODO:** Define selection for multi-token answers, whitespace and token-boundary handling, and exact layer conventions.

**Additional positions**

Additional annotations are encouraged when the task has meaningful locations to study.

Use `semantic_positions` with keys following the fixed pattern `position_<number>`, starting at 1. Each entry identifies a span using character offsets in the original `input` text:

```json
"semantic_positions": {
  "position_1": {
    "start_char": 11,
    "end_char": 15
  }
}
```

- **`start_char`** — Index of the first character included in the span. Counting starts at 0.
- **`end_char`** — Index immediately after the last included character.

For the input `Please ask Jhon.`, the span above selects `Jhon`.

Count characters in the actual input text, not in its JSON representation. For example, an escaped newline (`\n`) represents one character.

Users do not need to supply token indices. The toolkit maps character spans to tokens after applying prompt formatting and tokenization. This keeps dataset annotations independent of the model’s tokenizer.

If annotations are supplied:

- Every example must contain the same position keys.
- Each key must represent the same semantic role across all examples.
- Character offsets may differ between examples.
- Each span must contain at least one character.
- Locations must be identifiable without using the target answer or the model’s correctness.

Only annotate additional locations of interest. The entire input does not need to be divided into spans.

**Example: name correction**

Suppose the task asks the model to correct name typos in an ASR transcript. You could use `position_1` for the name being checked.

In “Please ask Jhon,” its character span identifies `Jhon`. In “Micheal will present,” it identifies `Micheal`.

The text and offsets differ, but `position_1` consistently means “the name being checked.” The toolkit can compare activations from this location with those from its default positions.

**Validation and token mapping**

The toolkit must check that:

- Character offsets are integers.
- Each span satisfies `0 <= start_char < end_char <= length of input`.
- Position keys are consistent across examples.
- Each span maps to tokens within the supplied input after prompt formatting.

These checks verify the annotation structure. Users remain responsible for selecting the intended text.

**TODO:** Specify Unicode character-counting conventions for annotations prepared in different programming languages.

**TODO:** Define token selection when a character boundary falls inside a token and how activations from multi-token spans become probe features.

**TODO:** Define handling of invalid annotations and missing positions. Examples must not be silently omitted.

## Dataset example

This record uses only the default capture positions:

```json
{"id":"example_001","input":"Classify this review as Positive or Negative:\nI loved this product.","target_answer":"Positive","split":"train"}
```

The toolkit adds the final-answer instruction before running the model.

**TODO:** Add a complete dataset example with additional character-span annotations.

**TODO:** Finalize validation rules for malformed records and missing fields.
