# Internal Confidence Gate

A toolkit for building confidence gates from internal model activations for
user-defined tasks.

The toolkit trains lightweight probes to estimate whether a model's prediction
is correct, then compares their performance with confidence derived from output
probabilities. It supports task-specific prompts, correctness rules, and token
positions for collecting activations without modifying the underlying model.

## Quick start

Use Python 3.10 or newer. The current backend supports standard Hugging Face
Transformers text-only, decoder-only chat models through
`AutoModelForCausalLM`; CUDA is optional.

```sh
git clone https://github.com/ShovalMessica/internal-confidence-gate.git
cd internal-confidence-gate
python -m pip install -r requirements.txt
```

**New task? Start with [Add your own task](docs/add-your-own-task.md)** to create
the prompt, [JSONL dataset](docs/dataset-format.md), and configuration.
Run the commands below from the repository root after filling in your configuration.

Validate those inputs without loading the model:

```sh
python -m src.run configs/task.yaml --prepare-only
```

Run the model and inspect its answers before training probes:

```sh
python -m src.run configs/task.yaml --behavior-only --show-examples 10
```

Then run the full pipeline:

```sh
python -m src.run configs/task.yaml
```

The full command captures activations, trains and selects a probe, evaluates
the frozen gate on test data, and creates a report under
`<output_dir>/<run_id>/`. Preparation and behavior checks accept small
development samples; the full run enforces the data minimums listed below.

## Research goal

Output probability can be high for an incorrect answer. This repository studies
whether signals in a model's internal activations provide a better reliability
score for deciding which completed predictions to accept.

The pipeline is:

**Configuration and dataset → generation → answer evaluation → activation
capture → probe training → validation selection → frozen test evaluation →
report.**

Target answers are used to determine whether saved model predictions are
correct. They supervise probe training and evaluation, but are never inserted
into the model prompt or used as probe features. Applying a trained gate does
not require a target answer.

## Supported tasks

The toolkit targets tasks where:

1. The model produces one final answer that can be evaluated as a whole.
2. A supplied target answer or task-specific matcher can determine correctness.
3. The response can follow the toolkit's fixed final-answer contract.

Suitable tasks include classification, multiple choice, factual question
answering, and reasoning followed by one final decision. Free-form responses
with multiple independently evaluated claims are outside the current scope.

The toolkit supplies the final-answer instruction and the `FINAL:` marker.
Direct mode generates the answer immediately; reasoning mode first allows a
reasoning phase and then requests the final answer. Exact behavior and settings
are documented under [Generation](docs/configuration.md#generation).

## Confidence gate

The current backend captures Hugging Face hidden states at three default token
locations: the end of the initial prompt, the supplied `FINAL:` marker, and the
generated answer tokens. Optional character spans can identify additional
task-specific locations in the input.

Linear probes are trained on the training split across configured positions and
layers. Validation data chooses the probe and the strictest threshold that
retains at least `target_tpr` of correct predictions while minimizing FPR. Test
data is used once to evaluate that frozen choice against an independently
thresholded output-probability baseline.

The final gate report includes TPR, FPR, balanced accuracy, AUROC, coverage, and
accepted-error rate. See [Configuration](docs/configuration.md) for the precise
training, selection, baseline, reporting, and reuse contracts.

## Data requirements

A full probe run requires at least 700 valid examples and at least:

| Split | Total | Correct predictions | Incorrect predictions |
| --- | ---: | ---: | ---: |
| Train | 200 | 100 | 100 |
| Validation | 100 | 50 | 50 |
| Test | 100 | 50 | 50 |

Correct and incorrect counts are known only after generation. Enabled
`UNKNOWN` abstentions and invalid outputs do not contribute to these counts.
These are provisional minimums, not guarantees of reliable probe performance.

Develop and check the prompt on a separate small sample, then freeze the prompt,
model, generation settings, and split assignments before the final run. Related
examples must not cross splits.

## Outputs and reuse

Each run directory contains its provenance record and log, generation and
evaluation manifests, one activation file, one probe file, and report files.
Validated artifacts are reused across repeated commands; extending a dataset
reuses compatible per-example generations and activations. See
[Run identity and reuse](docs/configuration.md#run-identity-and-reuse) for the
complete layout and rules.

## Documentation

| File | Purpose |
| --- | --- |
| [Add your own task](docs/add-your-own-task.md) | Prepare your own data and run the pipeline |
| [Dataset format](docs/dataset-format.md) | Exact JSONL record contract |
| [Configuration](docs/configuration.md) | Complete settings and artifact reference |
| [Prompt examples](docs/prompt-examples.md) | Direct and reasoning prompt designs |
| [NER example](examples/ner/README.md) | Executable synthetic starter task |

## Limitations

- White-box access to model hidden states is required.
- A trained gate is tied to its model, prompt, generation procedure, activation
  features, and data distribution; changes require revalidation.
- Probe results are correlational and do not establish that a representation
  causally controls correctness.
- A reliability score estimates risk; it does not guarantee correctness.
- Deployment of the trained gate is outside the current pipeline.

Circuit finding, TransformerLens/PEAP signals, head-level analysis, and
uncertainty intervals remain future extensions.

## Development

Run the offline suite without loading a model:

```sh
python -m unittest discover -s tests -v
```
