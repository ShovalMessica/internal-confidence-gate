# Dataset format

This guide explains how to prepare your task data in the format expected by the toolkit.

Provide a `.jsonl` file with one JSON object per line. Each object represents one example.

All fields are required unless marked optional. **TODO** indicates an unresolved specification.

## Fields

- **`id`** — A unique example identifier. IDs must be unique across the dataset.  
  **TODO:** Accepted ID type.

- **`input`** — The full task prompt as text, including instructions and example-specific content. The toolkit appends an instruction requiring `FINAL: <answer>` so it can extract the final answer and locate its tokens. You do not add this instruction yourself.  
  **TODO:** Exact appended instruction, chat-template handling, and response-validation rules.

- **`target_answer`** — The expected final answer as text, without `FINAL:`. The toolkit compares it with the model’s answer after normalizing case and surrounding whitespace. This determines correctness for supervised probe training and evaluation. The target answer is never supplied as a probe input or inserted into the prompt by the toolkit. Applying the trained probe does not require it.  
  **TODO:** Further normalization rules and support for multiple accepted answers.

- **`split` (optional)** — `"train"`, `"validation"`, or `"test"`. When omitted across the dataset, splits are assigned randomly.  
  **TODO:** Default proportions, reproducibility, and handling of partially assigned datasets.

- **`semantic_positions` (optional)** — Additional task-specific token locations where the toolkit should capture activations. The format and requirements are explained below. If omitted across the dataset, only default positions are used.

## Semantic positions

As explained in the [main README](../README.md), the probe estimates prediction reliability from internal activations, rather than output probabilities alone.

A model produces activations across many layers and token positions. To focus on potentially informative signals, the toolkit compares probes trained on activations from selected semantic locations—positions with a meaningful relationship to the task or its final answer.

Probe selection uses validation data. Test data evaluates the frozen selection.

### Default positions

The toolkit captures residual-stream activations across all layers at:

- **Last prompt token:** the final token of the complete prompt passed to the model, before response generation.
- **Final-answer marker:** the token containing the colon in the final `FINAL:` marker.
- **Answer:** the answer token following that marker.

These positions are located automatically and require no annotations.

**TODO:** Define answer-token selection for multi-token answers, whitespace and token-boundary handling, and exact layer-index conventions.

### Additional positions

Adding task-specific locations is encouraged when meaningful locations exist.

Use `semantic_positions` to associate each location’s semantic name with its token indices:

```json
"semantic_positions": {
  "entity_name": [42],
  "supporting_context": [55, 56, 57]
}
```

The indices above are illustrative. Actual indices depend on the example, model tokenizer, and complete prompt formatting.

You choose the semantic names. The toolkit reads the entries in `semantic_positions`; numbered fields such as `semantic_positions_1` are unnecessary.

If this field is supplied:

- Every example across training, validation, and testing must contain the same semantic names.
- The token indices may differ between examples.
- Each named location must contain at least one token.
- Locations must be identifiable from the input without using the target answer or the correctness of the model’s response.

You only annotate additional locations of interest. You do not need to divide the entire prompt into spans.

### Example: name correction

Suppose your task asks the model to correct name typos in an ASR transcript. The name being checked is a potentially useful location for activation capture.

For an input containing “Please ask Jhon,” you identify the tokens representing `Jhon`. For another containing “Micheal will present,” you identify the tokens representing `Micheal`.

Both locations are named `entity_name`. Their text and token indices differ, but they represent the same semantic role.

The toolkit can then compare probes using entity-name activations with probes using its default positions.

### Coordinates and validation

The proposed coordinate convention is zero-based token indices into the exact prompt processed by the model, including chat-template tokens and the toolkit’s appended instruction, but excluding batch padding.

**TODO:** Finalize this convention and define how users obtain the exact tokenized prompt before annotating it.

The toolkit must validate that:

- Indices are integers and fall within the prompt.
- Each named location is nonempty.
- Semantic names are consistent across examples.

These checks establish structural validity. Users remain responsible for identifying the intended semantic locations.

**TODO:** Decide how invalid annotations are handled. Examples must not be silently omitted.

**TODO:** Define how activations from locations containing multiple tokens become probe features.

## Dataset example

This record contains a complete task prompt, its target answer, and a split assignment. It uses only the default capture positions.

```json
{"id":"example_001","input":"Classify this review as Positive or Negative:\nI loved this product.","target_answer":"Positive","split":"train"}
```

The toolkit adds the final-answer instruction before running the model.

**TODO:** Add a complete example with verified semantic token annotations once the coordinate workflow is finalized.

**TODO:** Finalize dataset-validation rules for malformed records and missing fields.
