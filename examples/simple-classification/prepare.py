"""Create a self-contained starter dataset and configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--examples", type=int, default=700)
    args = parser.parse_args()
    if args.examples < 700:
        parser.error("--examples must be at least 700")

    root = Path(__file__).resolve().parent
    generated = root / "generated"
    generated.mkdir(exist_ok=True)
    dataset = generated / "dataset.jsonl"
    records = (
        {
            "id": number,
            "input": f"Integer: {number}",
            "target_answer": "EVEN" if number % 2 == 0 else "ODD",
        }
        for number in range(args.examples)
    )
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
