# Dataset format

This guide explains how to prepare your task data in the format expected by the toolkit.

For the motivation behind confidence probes and the explanation of activation capture locations, first read [How the approach works](../README.md#how-the-approach-works).

Provide the dataset as a `.jsonl` file, with one JSON object per line representing one example.

## Example fields

Each example supplies the complete prompt and its target answer. Model responses, correctness assessments, probabilities, and activation records are produced by the toolkit.

All fields are required unless marked optional.

**`id`**

A unique identifier for the example. IDs must be unique across the dataset.

**`input`**

The complete prompt sent to the model, including task instructions and example-specific content. Supply it as text, without placeholders or a separate prompt template.

The prompt must instruct the model to produce its final answer in the toolkit’s `FINAL: <answer>` format, whether or not reasoning precedes it.

**TODO:** Provide the exact required response instruction and define how the input is passed through the model’s chat template.

**`target_answer`**

The expected final answer as text, without the `FINAL:` prefix.

The toolkit compares the model’s extracted answer with the target answer after normalizing both to determine whether the prediction is correct. This information is used for supervised probe training and evaluation.

The target answer is never inserted into the model’s prompt or provided to the probe as an input. Once trained, the probe estimates prediction reliability without knowing the target answer.

Normalization includes case and surrounding whitespace normalization.

**TODO:** Define the complete normalization rules and whether multiple accepted target answers are supported.

**`split` — optional**

Assigns the example to `"train"`, `"validation"`, or `"test"`.

If assignments are omitted across the dataset, the toolkit creates random splits.

**TODO:** Define default proportions, reproducibility settings, and handling of partially assigned datasets.

**Additional semantic annotations — optional; field name TODO**

These annotations identify particular words or passages in `input` where you want the toolkit to collect activations, alongside its default captures.

For example, a name-correction dataset could identify the name being checked. You do not need to divide the entire input into spans—only annotate the additional locations you want to study.

Adding meaningful task-specific locations is encouraged. If annotations are included, the same semantic names must be supplied across all examples. Their actual locations may differ.

Locations must be identifiable from the input without using the target answer or the correctness of the model’s response.

If annotations are omitted, the toolkit uses only its default capture locations.

**TODO:** Define the field name, annotation structure, coordinate system, mapping to tokens, and exact default capture locations.

## Example

The following record contains a complete prompt, its target answer, and a split assignment. No additional semantic annotations are supplied.

```json
{"id":"example_001","input":"Classify this review as Positive or Negative.\nReview: I loved this product.\nEnd your response with FINAL: <answer>.","target_answer":"Positive","split":"train"}
```

The output instruction above is illustrative until the exact response contract is finalized.

**TODO:** Add an example with additional semantic annotations once their format is settled.
