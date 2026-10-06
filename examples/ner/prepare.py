"""Create a deterministic synthetic name-correction example."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml


DEV_SEED = 7_041
FULL_SEED = 91_337
SPLIT_RATIOS = {"train": 0.60, "validation": 0.20, "test": 0.20}

NAME_PAIRS = (
    ("Katherine", "Kate"),
    ("Nicole", "Nickel"),
    ("Mark", "March"),
    ("David", "Debit"),
    ("Taylor", "Tailor"),
    ("Teresa", "Terry"),
    ("Elizabeth", "Beth"),
    ("Brian", "Braien"),
    ("Kathleen", "Cattie"),
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
    ("Alexander", "Alex"),
    ("Benjamin", "Ben"),
    ("Cynthia", "Cindy"),
    ("Deborah", "Debbie"),
    ("Edward", "Eddie"),
    ("Frederick", "Freddy"),
    ("Gabrielle", "Gabby"),
    ("Harold", "Harry"),
    ("Isabella", "Izzy"),
    ("Jacqueline", "Jackie"),
    ("Kenneth", "Kenny"),
    ("Lawrence", "Larry"),
    ("Matthew", "Matty"),
    ("Nathaniel", "Nate"),
    ("Olivia", "Oliviah"),
    ("Penelope", "Penny"),
    ("Richard", "Rich"),
    ("Samantha", "Sammy"),
    ("Thomas", "Tommy"),
    ("Victoria", "Vicky"),
)

LAST_NAMES = (
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Wilson", "Anderson", "Thomas", "Moore", "Martin", "Jackson",
    "Thompson", "White", "Lopez", "Lee", "Gonzalez", "Harris", "Clark",
    "Lewis", "Walker", "Hall", "Allen", "Young", "King", "Wright",
    "Scott", "Green",
)

UTTERANCE_TEMPLATES = (
    "Please send the {document} to {mention} before {deadline}.",
    "Could {mention} review the {document} {deadline}?",
    "I will ask {mention} to confirm the {topic} decision.",
    "We still need feedback from {mention} about {topic}.",
    "Let us invite {mention} to the {topic} session.",
    "Has {mention} approved the latest {document}?",
    "Please tell {mention} that the {topic} meeting moved to {deadline}.",
    "I left a note for {mention} in the {document}.",
    "Can {mention} present the {topic} update {deadline}?",
    "The next action belongs to {mention}, according to the {document}.",
)

DOCUMENTS = (
    "agenda", "budget", "contract", "design brief", "launch plan",
    "meeting notes", "proposal", "release checklist", "risk report",
    "status update", "test plan", "timeline",
)

TOPICS = (
    "budget", "customer research", "design", "hiring", "launch",
    "legal review", "migration", "operations", "planning", "security",
    "support", "testing",
)

DEADLINES = (
    "before lunch", "before tomorrow", "by Friday", "by noon", "next week",
    "on Monday", "this afternoon", "this evening", "today", "tomorrow morning",
)

_LEADING_UTTERANCE_ID = re.compile(r"(<MEETING_TRANSCRIPT>\n)<\d+>")


def content_key(input_text: str) -> str:
    """Return task content with the arbitrary utterance ID normalized."""

    return _LEADING_UTTERANCE_ID.sub(r"\1<ID>", input_text, count=1)


def _split_family_counts(family_count: int) -> dict[str, int]:
    raw = {name: family_count * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = family_count - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def _new_family(
    rng: random.Random,
    *,
    family_id: str,
    split: str,
    first_example_id: int,
    target_index: int,
) -> list[dict]:
    pairs = rng.sample(NAME_PAIRS, 10)
    surnames = rng.sample(LAST_NAMES, 10)
    participants = [
        {"first": first, "variant": variant, "last": last}
        for (first, variant), last in zip(pairs, surnames)
    ]
    rng.shuffle(participants)
    target = participants[target_index]
    label = chr(65 + target_index)
    speaker = f"Speaker {rng.randint(1, 20)}"
    template = rng.choice(UTTERANCE_TEMPLATES)
    fields = {
        "document": rng.choice(DOCUMENTS),
        "topic": rng.choice(TOPICS),
        "deadline": rng.choice(DEADLINES),
    }
    roster = "\n".join(
        f"{chr(65 + index)} {person['first']} {person['last']}"
        for index, person in enumerate(participants)
    )

    records = []
    for pair_index, (example_type, mention, target_answer) in enumerate(
        (
            ("corrupted", target["variant"], label),
            ("clean", target["first"], "NONE"),
        )
    ):
        example_id = first_example_id + pair_index
        utterance = template.format(mention=mention, **fields)
        input_text = (
            f"<PARTICIPANTS>\n{roster}\n</PARTICIPANTS>\n\n"
            f"<MEETING_TRANSCRIPT>\n<{example_id}><{speaker}>{utterance}\n"
            "</MEETING_TRANSCRIPT>"
        )
        start = input_text.rindex(mention)
        records.append(
            {
                "id": example_id,
                "input": input_text,
                "target_answer": target_answer,
                "split": split,
                "semantic_spans": {
                    "span_1": {"start_char": start, "end_char": start + len(mention)}
                },
                "metadata": {
                    "case_family": family_id,
                    "example_type": example_type,
                },
            }
        )
    return records


def build_records(
    count: int,
    *,
    seed: int,
    family_prefix: str,
    id_offset: int = 0,
    forbidden_content: Iterable[str] = (),
) -> list[dict]:
    if count <= 0 or count % 2:
        raise ValueError("The example count must be a positive even integer.")

    rng = random.Random(seed)
    seen = set(forbidden_content)
    records: list[dict] = []
    family_number = 0
    next_id = id_offset
    split_counts = _split_family_counts(count // 2)

    for split in ("train", "validation", "test"):
        created = 0
        while created < split_counts[split]:
            family_id = f"{family_prefix}_{family_number:05d}"
            candidate = _new_family(
                rng,
                family_id=family_id,
                split=split,
                first_example_id=next_id,
                target_index=family_number % 10,
            )
            keys = {content_key(record["input"]) for record in candidate}
            if len(keys) != len(candidate) or keys & seen:
                continue
            records.extend(candidate)
            seen.update(keys)
            family_number += 1
            next_id += len(candidate)
            created += 1

    validate_partition_isolation(records)
    return records


def validate_partition_isolation(
    records: Iterable[dict],
    *,
    external_content: Iterable[str] = (),
) -> None:
    seen = set(external_content)
    family_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        key = content_key(record["input"])
        if key in seen:
            raise ValueError(f"Repeated task content detected for example {record['id']}.")
        seen.add(key)
        family_splits[record["metadata"]["case_family"]].add(record["split"])

    crossing = [family for family, splits in family_splits.items() if len(splits) != 1]
    if crossing:
        raise ValueError(f"Related examples cross splits: {', '.join(crossing[:5])}")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_system_prompt(source: Path, destination: Path) -> None:
    """Extract the literal prompt from the Markdown display code block."""

    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3 or lines[0] != "```text" or lines[-1] != "```":
        raise ValueError(f"{source} must contain one outer ```text code block.")
    destination.write_text("\n".join(lines[1:-1]) + "\n", encoding="utf-8")


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
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--examples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=FULL_SEED)
    args = parser.parse_args()
    if args.examples < 700 or args.examples % 2:
        parser.error("--examples must be an even integer of at least 700")

    root = Path(__file__).resolve().parent
    generated = root / "generated"
    generated.mkdir(exist_ok=True)
    system_prompt = generated / "system-prompt.txt"
    write_system_prompt(root / "system-prompt.md", system_prompt)

    development = read_jsonl(root / "dev-sample.jsonl")
    validate_partition_isolation(development)
    development_keys = {content_key(record["input"]) for record in development}
    records = build_records(
        args.examples,
        seed=args.seed,
        family_prefix="full",
        forbidden_content=development_keys,
    )
    validate_partition_isolation(records, external_content=development_keys)

    dataset = generated / "dataset.jsonl"
    write_jsonl(dataset, records)
    write_config(
        generated / "task.yaml",
        model=args.model,
        dataset=dataset,
        system_prompt=system_prompt,
        output_dir=generated / "outputs",
    )
    write_config(
        generated / "dev-task.yaml",
        model=args.model,
        dataset=root / "dev-sample.jsonl",
        system_prompt=system_prompt,
        output_dir=generated / "dev-outputs",
    )
    print(f"Wrote {len(records)} disjoint examples to {dataset}")
    print(f"Wrote {generated / 'dev-task.yaml'}")
    print(f"Wrote {generated / 'task.yaml'}")


if __name__ == "__main__":
    main()
