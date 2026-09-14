# Design checklist

Working list for completing the toolkit's design before implementation. Agreed behavior belongs in the [configuration](configuration.md) and [dataset format](dataset-format.md) specifications. Unchecked items remain **TODO**, not finalized requirements.

## Agreed direction

- [x] Use Hugging Face Transformers with `AutoModelForCausalLM`; users supply a model ID or local pretrained checkpoint.
- [x] Support direct answers and reasoning followed by a final answer, subject to model capabilities.
- [x] Use the model's generation defaults, allow user overrides, and record effective settings.
- [x] Have the toolkit append the `FINAL: <answer>` instruction; allow `UNKNOWN` by default.
- [x] Keep complete prompts and target answers in JSONL, with optional user-assigned splits and additional semantic character spans.
- [x] Use target answers for supervised probe training and evaluation, never as probe features.
- [x] Focus on activation probes and train/validation/test evaluation. Deployment remains the user's responsibility; circuit finding is outside scope.

## Reasoning and direct answers

- [ ] Define the user-facing mode setting, its default, and unsupported-mode behavior.
- [ ] Define how the appended output instruction and the model's chat template work together in each mode.
- [ ] Finalize output-token limits. One total limit is proposed; its value and whether any separate limits are needed remain open.
- [ ] Define handling when reasoning consumes the budget and no complete final answer is produced.
- [ ] Define final-answer parsing when reasoning includes marker-like text, repeated `FINAL:` markers, or model-specific reasoning delimiters. Report formatting failures rather than guessing.
- [ ] Keep capture meanings explicit: the last prompt token precedes reasoning, while the final-answer marker follows it. Finalize answer-token boundaries and multi-token handling.
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
