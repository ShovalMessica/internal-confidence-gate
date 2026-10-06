"""Create a self-contained starter dataset and configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


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
    train_end = int(args.examples * 0.70)
    validation_end = train_end + int(args.examples * 0.15)

    def record(index: int) -> dict:
        if index < train_end:
            split, split_index = "train", index
        elif index < validation_end:
            split, split_index = "validation", index - train_end
        else:
            split, split_index = "test", index - validation_end

        pattern = split_index % 8
        is_true = pattern % 2 == 0
        if pattern < 6:
            left = 100_003 + (index * 7_919) % 800_000
            right = 100_019 + (index * 1_543) % 800_000
            operator = "*"
            result = left * right
        else:
            left = 10 + (index * 17) % 80
            right = 10 + (index * 29) % 80
            operator = "+"
            result = left + right
        shown_result = result if is_true else result + 1 + index % 9
        return {
            "id": index,
            "input": (
                "Is this arithmetic equality true or false?\n"
                f"{left} {operator} {right} = {shown_result}"
            ),
            "target_answer": "TRUE" if is_true else "FALSE",
            "split": split,
        }

    records = (record(index) for index in range(args.examples))
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    config = {
        "model_name_or_path": args.model,
        "dataset_path": str(dataset),
        "system_prompt_path": str(root / "system-prompt.md"),
        "reasoning_mode": "direct",
        "decoding_strategy": "greedy",
        "answer_max_new_tokens": 4,
        "allow_abstention": False,
        "output_dir": str(generated / "outputs"),
    }
    config_path = generated / "task.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(config_path)


if __name__ == "__main__":
    main()
