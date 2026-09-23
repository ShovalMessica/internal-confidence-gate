# Prompt design examples

Design and refine the task prompt you supply so the model reliably answers in the form your task expects—for example, a participant ID rather than a name or explanation. Your instructions, context, examples, and decision rules all influence this behavior. Test different versions on a small development sample before settling on a prompt design.

Below are two complete prompt examples with illustrative data. They show how instructions, decision rules, and demonstrations can be combined in a prompt. Each **Example prompt** represents the text supplied in one dataset record's `input`.

The separately marked **toolkit-added** text is automatic; users do not write it. It illustrates the [planned generation behavior](configuration.md#generation), which is not implemented yet.

## Named-entity correction — no reasoning

**Example prompt** — includes two fixed demonstrations, followed by the input to solve:

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

# Now solve this input
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
<73><Speaker 2>Please send the updated draft to Kate before tomorrow's review.
</MEETING_TRANSCRIPT>
```

*The participant list and transcript under “Now solve this input” change between dataset records. The instructions and the two demonstration examples stay the same.*

**Toolkit-added instruction:**

```text
Return only the final answer on one line, without reasoning or explanation. The prefix FINAL: is already supplied; do not repeat it.
If you cannot determine the answer, return UNKNOWN.
```

The toolkit starts the reply with `FINAL:`; the model supplies the answer. **Illustrative reply:** `FINAL: I`.

## Speaker attribution — with reasoning

**Example prompt:**

```text
Identify the named participant hidden behind the anonymous label
Speaker 2.
Answer with the candidate ID or UNKNOWN, never the participant name.

IMPORTANT: `Speaker 1`, `Speaker 2`, and similar labels are anonymous aliases
for people in the participant list. They are not additional people.

PARTICIPANTS:
A: Daniel Brooks
B: Maya Chen
C: Sara Patel

Use the ordinary transcript to infer the identity:

- An explicit target self-identification (`I am NAME` or `my name is NAME`) is
  sufficient evidence.
- If another speaker directly calls, questions, requests, greets, or invites
  exactly one named participant and Speaker 2 gives the immediate
  response, infer that Speaker 2 is the named participant.
- A name spoken by Speaker 2, such as `You go, NAME` or `Hey NAME`,
  normally names another person rather than the target.
- If a speaker refers to a named person as `he`, `she`, or `they` in the same
  statement, that person is being discussed rather than directly addressed.
- Choose UNKNOWN when there is no sufficient identity evidence, several
  people are addressed, another speaker intervenes, or the evidence conflicts.

Consecutive rows from one anonymous speaker are one turn.

Do not guess from meeting topics or from the participant-list order. Once one
identity is unambiguously established, state only
`Speaker 2 = NAME = ID` in your reasoning and stop.

TRANSCRIPT:
Speaker 1: Maya, could you walk us through the revised schedule?
Speaker 2: Sure. The first review is on Tuesday, and the final handoff is on Friday.
Speaker 1: Thanks. Sara will review the draft afterward.
```

*The target speaker (`Speaker 2` wherever it appears in the instructions), participant list, and transcript change between dataset records. The decision rules stay the same.*

**Toolkit-added instruction before reasoning:**

```text
Reason about the task first. A separate final-answer instruction will follow.
```

**Illustrative model reasoning:**

```text
Speaker 2 = Maya Chen = B
```

**Toolkit-added instruction after reasoning:**

```text
Return only the final answer on one line, without reasoning or explanation. The prefix FINAL: is already supplied; do not repeat it.
If you cannot determine the answer, return UNKNOWN.
FINAL:
```

The model continues after the supplied `FINAL:` with `B`. **Illustrative final answer:** `FINAL: B`.
