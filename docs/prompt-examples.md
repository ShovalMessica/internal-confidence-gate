# Prompt design examples

These examples show how task instructions, rules, and demonstrations can guide
the expected answer. Test prompts on development data before freezing them.
Text marked **toolkit-added** is automatic and should not be copied into your
prompt. See [Generation](configuration.md#generation).

## Named-entity correction - no reasoning

In [`configs/task.yaml`](../configs/task.yaml), set `reasoning_mode: direct`
and keep `allow_abstention: true`. Save the system prompt below in a text file
without its code fences, then set `system_prompt_path` in the YAML to its
absolute path. Put the example-specific content in each JSONL record's `input`.

**Example system prompt (fixed across examples):**

*This fixed message contains the task rules and two demonstrations.*

```text
# Task
Given a participant list and a transcript chunk, output exactly one of:
- The participant's label (`A`-`J`) ONLY if a name is MISSPELLED (does not match any participant's spelling) but sounds like one participant.
- Your job is to catch a SPELLING error, not to identify who is speaking or who is mentioned. A correctly-written participant name is not an error.
- `NONE` if no correction should be made.
- `UNKNOWN` if the evidence is ambiguous.

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

# Output Rule
Output exactly `NONE`, `UNKNOWN`, or one participant label like `E`.
Output only that single token.
No explanation. No reasoning. No quoted phrase. No markdown. No transcript rewrite.
```

**Example dataset `input` (changes between examples):**

```text
<PARTICIPANTS>
A John Wilson
B Christopher Clark
C Mary Brown
D Melissa Stone
E Kathryn Wilson
F Jeffrey Weiss
G Joshua Waters
H Jason Cochran
I Angela Francis
J Juan West
</PARTICIPANTS>

<MEETING_TRANSCRIPT>
<119><Russell Peterson>I heard from Wan that ops mostly wants accountability, not self-serve, so support-only might be fine initially.
</MEETING_TRANSCRIPT>
```

**Toolkit-added instruction:**

```text
Answer immediately after FINAL: with only the final answer. Do not repeat the prefix or add reasoning or explanation.
If you cannot determine the answer, return UNKNOWN.
```

The toolkit starts the reply with `FINAL:`; the model supplies the answer. **Illustrative reply:** `FINAL: J`.

## Speaker attribution - with reasoning

In `configs/task.yaml`, set `reasoning_mode: reasoning`, keep
`allow_abstention: true`, and leave `system_prompt_path` commented out for this
complete-prompt example. Fill the placeholders yourself and put the resulting
text in each JSONL record's `input`.

**Example prompt:**

*`{target_speaker}`, `{candidate_list}`, and `{transcript}` change between examples. The toolkit uses the fixed abstention label `UNKNOWN` when abstention is enabled.*

```text
Identify the named participant hidden behind the anonymous label
{target_speaker}.

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
- Choose `UNKNOWN` when there is no sufficient identity evidence, several
  people are addressed, another speaker intervenes, or the evidence conflicts.

Consecutive rows from one anonymous speaker are one turn.

Do not guess from meeting topics or from the participant-list order. Once one
identity is unambiguously established, state only
`{target_speaker} = NAME = ID` in your reasoning and stop.

TRANSCRIPT:
{transcript}

The final answer must be one candidate ID or `UNKNOWN`, never the participant
name.
```

**Toolkit-added instruction before reasoning:**

```text
Reason about the task first. A separate final-answer instruction will follow.
```

**Toolkit-added instruction after reasoning:**

```text
Answer immediately after FINAL: with only the final answer. Do not repeat the prefix or add reasoning or explanation.
If you cannot determine the answer, return UNKNOWN.
FINAL:
```

The model continues after the supplied `FINAL:` with `B`. **Illustrative final answer:** `FINAL: B`.
