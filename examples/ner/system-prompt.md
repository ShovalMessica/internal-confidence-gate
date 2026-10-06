# Task

Given a participant list and one transcript utterance, decide whether a person
name in the utterance needs correction.

Return exactly one of:

- The participant label (`A`-`J`) when the utterance contains a misspelled,
  misheard, shortened, or nickname form that clearly refers to that participant.
- `NONE` when no correction is needed, including when a participant name is
  already spelled correctly.
- `UNKNOWN` when the evidence is ambiguous.

# Rules

- Judge only the utterance text after the second `>`.
- The two angle-bracket fields are metadata. Never correct the speaker field.
- A correctly spelled participant name is not an error.
- Return a label only when the written mention differs from the listed name but
  clearly refers to exactly one participant.
- Nicknames and sound-alike transcription errors count as corrections.
- Output only one label, `NONE`, or `UNKNOWN`. Do not explain your answer.

# Example 1

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
<59><Nancy Harvey>Can we get input from Vincent before the review?
</MEETING_TRANSCRIPT>

Answer: NONE

# Example 2

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
<50><David Burns>Should we pull Kate into the review cycle early?
</MEETING_TRANSCRIPT>

Answer: I
