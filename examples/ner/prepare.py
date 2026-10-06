"""Create the full synthetic name-correction example and its configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


NAME_PAIRS = (
    ("Katherine", "Kate"),
    ("Nicole", "Nickel"),
    ("Mark", "March"),
    ("David", "Debit"),
    ("Taylor", "Tailor"),
    ("Teresa", "Terry"),
    ("Elizabeth", "Beth"),
    ("Brian", "Braien"),
    ("Kathy", "Cattie"),
    ("Knox", "Nocks"),
    ("Christopher", "Chris"),
    ("Rebecca", "Becky"),
    ("William", "Will"),
    ("Margaret", "Maggie"),
    ("Robert", "Rob"),
    ("Jennifer", "Jenny"),
    ("Daniel", "Dan"),
    ("Patricia", "Patty"),
    ("Michael", "Mike"),
    ("Anthony", "Tony"),
)

LAST_NAMES = (
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Wilson",
    "Anderson",
)

UTTERANCES = (
    "Please send the revised agenda to {mention} before tomorrow.",
    "Could {mention} review the final draft this afternoon?",
    "I will ask {mention} to confirm the launch date.",
    "We still need feedback from {mention} on the proposal.",
    "Let us invite {mention} to the next planning session.",
)


def build_record(example_id: int, split: str, split_index: int) -> dict:
    target_slot = (split_index // 2) % 10
    pair_offset = (example_id * 7) % len(NAME_PAIRS)
    pairs = [NAME_PAIRS[(pair_offset + index) % len(NAME_PAIRS)] for index in range(10)]
    participants = [
        f"{chr(65 + index)} {first} {LAST_NAMES[(example_id + index) % 10]}"
        for index, (first, _) in enumerate(pairs)
    ]

    corrupted = split_index % 2 == 0
    first_name, variant = pairs[target_slot]
    mention = variant if corrupted else first_name
    target_answer = chr(65 + target_slot) if corrupted else "NONE"
    utterance = UTTERANCES[example_id % len(UTTERANCES)].format(mention=mention)
    speaker = participants[(target_slot + 3) % 10].split(" ", 1)[1]
    input_text = (
        "<PARTICIPANTS>\n"
        + "\n".join(participants)
        + "\n</PARTICIPANTS>\n\n<MEETING_TRANSCRIPT>\n"
        + f"<{1000 + example_id}><{speaker}>{utterance}\n"
        + "</MEETING_TRANSCRIPT>"
    )
    start = input_text.rindex(mention)
    return {
        "id": example_id,
        "input": input_text,
        "target_answer": target_answer,
        "split": split,
        "semantic_spans": {
            "span_1": {"start_char": start, "end_char": start + len(mention)}
        },
    }


def build_records(count: int, *, id_offset: int = 0) -> list[dict]:
    train_count = int(count * 0.70)
    validation_count = int(count * 0.15)
    boundaries = (
        ("train", 0, train_count),
        ("validation", train_count, train_count + validation_count),
        ("test", train_count + validation_count, count),
    )
    records: list[dict] = []
    for split, start, end in boundaries:
        records.extend(
            build_record(id_offset + index, split, index - start)
            for index in range(start, end)
        )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_config(
    path: Path,
    *,
    model: str,
    dataset: Path,
    system_prompt: Path,
    output_dir: Path,
) -> None:
    config = {
        "model_name_or_path": model,
        "dataset_path": str(dataset.resolve()),
        "system_prompt_path": str(system_prompt.resolve()),
        "reasoning_mode": "direct",
        "decoding_strategy": "greedy",
        "answer_max_new_tokens": 4,
        "allow_abstention": True,
        "output_dir": str(output_dir.resolve()),
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--examples", type=int, default=2_000)
    args = parser.parse_args()
    if args.examples < 700:
        parser.error("--examples must be at least 700")

    root = Path(__file__).resolve().parent
    generated = root / "generated"
    generated.mkdir(exist_ok=True)

    dataset = generated / "dataset.jsonl"
    write_jsonl(dataset, build_records(args.examples))
    write_config(
        generated / "task.yaml",
        model=args.model,
        dataset=dataset,
        system_prompt=root / "system-prompt.md",
        output_dir=generated / "outputs",
    )
    write_config(
        generated / "dev-task.yaml",
        model=args.model,
        dataset=root / "dev-sample.jsonl",
        system_prompt=root / "system-prompt.md",
        output_dir=generated / "dev-outputs",
    )
    print(f"Wrote {dataset}")
    print(f"Wrote {generated / 'dev-task.yaml'}")
    print(f"Wrote {generated / 'task.yaml'}")


if __name__ == "__main__":
    main()
