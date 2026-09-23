# Prompt design examples

Design and refine the task prompt you supply so the model reliably answers in the form your task expects—for example, a participant ID rather than a name or explanation. Your instructions, context, examples, and decision rules all influence this behavior. Test different versions on a small development sample before settling on a prompt design.

**You supply the task prompt. The toolkit will add its answer-format instructions and `FINAL:` prefix automatically; do not add these yourself.** See [Generation](configuration.md#generation). Generation is not implemented yet.

Below are prompt examples for two specific tasks: correcting misspelled names without reasoning, and identifying a speaker with reasoning. The task instructions are adapted from our research prompts; the formatting steps show how the toolkit is planned to handle each mode.

Placeholders show where example-specific content was inserted. Supply the complete, filled-in prompt in each dataset record’s `input`; the toolkit does not fill placeholders.

## Named-entity correction — no reasoning

The task is to identify which participant a misspelled name refers to. The answer is a participant label (`A`–`J`), or `NONE` when no correction is needed. Our original experiment used Qwen3-4B-Instruct-2507 with thinking disabled.

The prompt separates utterance text from metadata, defines which spelling changes count, and includes two demonstrations: a correctly spelled name that needs no correction, and a nickname mapped to a participant.

**Task instructions supplied by the user** — adapted from condition `7p7-41-spkout`:

```text
# Task
Given a participant list and a transcript chunk, output exactly one of:
- The participant's label (`A`-`J`) ONLY if a name is MISSPELLED (does not match any participant's spelling) but sounds like one participant.
- Your job is to catch a SPELLING error, not to identify who is speaking or who is mentioned. A correctly-written participant name is not an error.
- `NONE` if no correction should be made.
- `UNKNOWN` if you cannot determine whether or how to correct the name.

Each participant in the list is identified by a single letter label (A through J); answer with that label.

# Critical Rule
A participant being mentioned is not a reason to output a label.
If the utterance already spells a participant's name correctly, output `NONE`.
Only output a participant label (`A`-`J`) when a name in the utterance text is clearly a garbled or misheard spelling of exactly one participant.

# Tags And Metadata
- Transcript lines look like `<utterance_id><speaker_name_or_id>utterance_text`.
- The first two angle-bracket fields are metadata, not content to correct.
- The speaker name is metadata; never answer from it, even when the speaker is also a participant.
- Judge only the utterance text that follows the second `>`.

# How To Decide
Read the utterance text and compare any person mention to the participant list.
- If the mention is an exact spelling of a participant's name, output `NONE`.
- If the mention is an ordinary word or phrase not used as a person's name, output `NONE`.
- If the mention is not a valid spelling of any participant but clearly sounds like exactly one participant's name, output that participant's label (`A`-`J`).
- People are sometimes referred to by a nickname or shortened form (for example, Kate for Katherine, Beth for Elizabeth, Nikki for Nicole, Terry for Teresa). If the utterance uses such a short form or nickname that is not itself any participant's listed name but clearly stands for exactly one participant, treat it as a corrupted mention and output that participant.
- A name can be mis-heard by speech recognition and written as a different word or name that SOUNDS similar when spoken aloud (similar syllable count and key consonant/vowel sounds), even if spelled differently. For example: march for Mark, nickel for Nicole, debit for David, tailor for Taylor, knocks for Knox. If a word in the utterance is not itself a participant's name but sounds like exactly one participant's name when pronounced, treat it as a corrupted mention and output that participant.
- To judge whether a word is a mis-heard name, compare how the two sound out loud: a shared beginning consonant sound, similar vowels, and similar overall shape. If a word shares most of a participant's sounds (for example debit and David, cattie and Kathy, braien and Brian), treat it as that participant's mis-heard name and output that participant.
- When a name IS clearly mis-spelled or mis-heard, DO output the participant it sounds like; do not be over-cautious about corrupted names.
- Remember: nicknames (Kate for Katherine) and sound-alike garbles (march for Mark, nickel for Nicole) ARE mis-spellings -- correct those to the participant.
- If unsure, output `UNKNOWN`.

# Example 1 Input
<PARTICIPANTS>
A Nancy Harvey
B Nancy Tran
C Michele Tucker
D Jocelyn Jones
E Chad Hamilton
F Jaclyn Neal
G Vincent Howell
H Brittney Wright
I Kelly Martinez
J Abigail Medina
</PARTICIPANTS>

<MEETING_TRANSCRIPT>
<59><Nancy Harvey>Can we get input from Vincent on what objections admins are raising in calls lately?
</MEETING_TRANSCRIPT>

# Example 1 Output
NONE

# Example 2 Input
<PARTICIPANTS>
A Jeffrey Rocha
B Corey Dalton
C Ashley Erickson
D Makayla Sawyer
E Patrick Oneal
F Mathew Lozano
G Barbara Taylor
H Brian Collins
I Katherine Smith
J David Burns
</PARTICIPANTS>

<MEETING_TRANSCRIPT>
<50><David Burns>Since legal is involved, should we pull Kate into the review cycle early?
</MEETING_TRANSCRIPT>

# Example 2 Output
I
```

**Example-specific content** — include this after the task instructions in the same `input`, replacing the placeholders:

```text
<PARTICIPANTS>
{participant_list}
</PARTICIPANTS>

<MEETING_TRANSCRIPT>
{transcript_chunk}
</MEETING_TRANSCRIPT>
```

**Added by the toolkit, not the user** — with `reasoning_mode: direct` and the default `allow_abstention: true`, append this instruction to the task prompt:

```text
Return only the final answer on one line, without reasoning or explanation. The prefix FINAL: is already supplied; do not repeat it.
If you cannot determine the answer, return UNKNOWN.
```

The toolkit then starts the model's reply with:

```text
FINAL:
```

The model generates only what follows that prefix, such as `I`, giving `FINAL: I`. No reasoning stage runs.

This adaptation separates `NONE` (no correction needed) from `UNKNOWN` (uncertain). The original prompt used `NONE` for both and requested a bare answer without `FINAL:`; those output instructions have been replaced here by the toolkit's planned formatting.

Source in the original research repository: `research/ner/experiments/7p7-41-spkout/prompt.md` and its accompanying `condition.json`.

## Speaker attribution — with reasoning

The task is to identify the participant behind one anonymous speaker label in a transcript. The answer is a candidate ID or `UNKNOWN`. Our original Setup 20 experiment used Qwen3-8B with a separate reasoning stage.

This prompt uses evidence rules instead of demonstrations: self-identification, immediate responses to a named addressee, misleading name mentions, and conflicting evidence. It tells the model when to stop reasoning and when to abstain.

**Task prompt supplied by the user** — adapted from Setup 20; replace the placeholders before saving it as `input`:

```text
Identify the named participant hidden behind the anonymous label
{target_speaker}.
Answer with the candidate ID or UNKNOWN, never the participant name.

IMPORTANT: `Speaker 1`, `Speaker 2`, and similar labels are anonymous aliases
for people in the participant list. They are not additional people.

PARTICIPANTS:
{candidate_list}

Use the ordinary transcript to infer the identity:

- An explicit target self-identification (`I am NAME` or `my name is NAME`) is
  sufficient evidence.
- If another speaker directly calls, questions, requests, greets, or invites
  exactly one named participant and {target_speaker} gives the immediate
  response, infer that {target_speaker} is the named participant.
- A name spoken by {target_speaker}, such as `You go, NAME` or `Hey NAME`,
  normally names another person rather than the target.
- If a speaker refers to a named person as `he`, `she`, or `they` in the same
  statement, that person is being discussed rather than directly addressed.
- Choose UNKNOWN when there is no sufficient identity evidence, several
  people are addressed, another speaker intervenes, or the evidence conflicts.

Consecutive rows from one anonymous speaker are one turn.

Do not guess from meeting topics or from the participant-list order. Once one
identity is unambiguously established, state only
`{target_speaker} = NAME = ID` in your reasoning and stop.

TRANSCRIPT:
{transcript}
```

**Added by the toolkit before reasoning** — with `reasoning_mode: reasoning`, append this to the task prompt:

```text
Reason about the task first. A separate final-answer instruction will follow.
```

The model generates reasoning until it ends or reaches `reasoning_max_new_tokens` (default: 1,024).

**Added by the toolkit after reasoning** — close the reasoning block if needed, then append the following to the model's reply, with `allow_abstention: true`:

```text
Return only the final answer on one line, without reasoning or explanation. The prefix FINAL: is already supplied; do not repeat it.
If you cannot determine the answer, return UNKNOWN.
FINAL:
```

The original task prompt and reasoning stay in context. The model continues after `FINAL:` with a candidate ID or `UNKNOWN`. **The toolkit supplies this entire continuation instruction and prefix; users do not include them in `input`.**

This example uses the toolkit's planned final-answer instruction in place of the original experiment's task-specific continuation. The final-answer instruction is the same in both modes; its placement differs. Model-specific chat formatting and reasoning boundaries remain under design.

Source in the original research repository: the `prompt` and `inference` fields in `research/speaker_attribution/Behavioral Anlysis/Behavior - Per setup/setup_20/setup.json`; continuation logic in `src/confidence_gate/speaker_attribution/run.py`.
