# Design checklist

Working list for completing the toolkit's design before implementation. Agreed behavior belongs in the [configuration](configuration.md) and [dataset format](dataset-format.md) specifications. Unchecked items remain **TODO**, not finalized requirements.

## Agreed direction

- [x] Use Hugging Face Transformers with `AutoModelForCausalLM`; users supply a model ID or local pretrained checkpoint.
- [x] Support direct answers and reasoning followed by a final answer, subject to model capabilities.
- [x] Use the model's generation defaults, allow user overrides, and record effective settings.
- [x] Have the toolkit supply the final-answer instruction and prefill `FINAL:` in both modes; allow `UNKNOWN` by default. See [Generation](configuration.md#generation).
- [x] Keep complete prompts and target answers in JSONL, with optional user-assigned splits and additional semantic character spans.
- [x] Use target answers for supervised probe training and evaluation, never as probe features.
- [x] Focus on activation probes and train/validation/test evaluation. Deployment remains the user's responsibility; circuit finding is outside scope.

## Reasoning and direct answers

- [ ] Define the user-facing mode setting and its default; verify unsupported combinations are rejected.
- [x] Specify injected instructions and the stage order for each mode. Model-specific chat-template and boundary handling remain TODO in the configuration specification.
- [x] Use separate reasoning and answer limits, following the original bounded-reasoning implementation. Numerical defaults and field names remain TODO.
- [x] On reasoning-budget exhaustion, close reasoning and proceed to the answer stage. Report exhausted, incomplete answer generation as invalid.
- [x] Extract from the recorded injected-marker boundary rather than searching reasoning for `FINAL:`; define structural validation in the configuration specification.
- [x] Record that last-prompt and marker captures coincide in direct mode and differ in reasoning mode. Exact token alignment and multi-token features remain TODO.
- [ ] Define which output-token probabilities form the baseline when reasoning precedes the answer.
- [ ] Record the effective mode and formatting alongside generation settings so saved results can be interpreted and reproduced.

## Remaining specification work

- [ ] **Configuration structure:** choose the file format, exact setting names, required fields, defaults, and a minimal complete example. See [Configuration](configuration.md).
- [ ] **Model and execution:** define compatibility checks, tested models, versions, tokenizer/revision handling, device, precision, and batch size.
- [ ] **Generation:** define overrides, missing model-default behavior, response validation, and remaining UNKNOWN rules.
- [ ] **Dataset:** settle ID type, normalization, multiple accepted answers, split defaults, and malformed-record handling. See [Dataset format](dataset-format.md).
- [ ] **Additional positions:** finalize Unicode coordinates, token alignment, multi-token features, and invalid/missing annotation handling. Preserve the same semantic roles across examples without requiring fixed offsets.
- [ ] **Capture:** define exact residual-stream layer locations, answer-token selection, and whether probes are compared separately by layer and semantic position.
- [ ] **Training and evaluation:** define probe type, preprocessing, selection procedure, retention target, probability baseline, and report contents. Keep test data out of selection.
- [ ] **Saved outputs:** define paths, artifact contents, reuse/resume rules, and reproducibility records.
- [ ] **Documentation:** reconcile the main README with finalized specifications, then add run instructions and a worked task example.
