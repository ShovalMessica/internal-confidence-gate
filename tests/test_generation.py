"""Offline tests for the two generation protocols."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import StoppingCriteriaList

from src.config import TaskConfig
from src.generation import (
    ALLOW_ABSTENTION_INSTRUCTION,
    ANSWER_INSTRUCTION,
    DISALLOW_ABSTENTION_INSTRUCTION,
    FINAL_MARKER,
    REASONING_INSTRUCTION,
    THINKING_BOUNDARY,
    _stable_seed,
    generation_units,
)
from src.model import LoadedModel


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"
    model_max_length = 10_000

    def __init__(self):
        self.rendered_contents = []
        self.rendered_modes = []
        self.rendered_messages = []

    @staticmethod
    def _ids(text):
        return [ord(character) + 10 for character in text]

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def decode(self, ids, **_):
        return "".join(chr(token_id - 10) for token_id in ids if token_id > 10)

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking
    ):
        self.rendered_messages.append(messages)
        self.rendered_contents.append(messages[-1]["content"])
        self.rendered_modes.append(enable_thinking)
        content = "\n".join(message["content"] for message in messages)
        return [8, *self._ids(content), 9]

    def pad(self, encoded, padding, return_tensors):
        rows = encoded["input_ids"]
        width = max(map(len, rows))
        padded = [[self.pad_token_id] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(masks),
        }


class _Model:
    device = torch.device("cpu")

    def __init__(self, outputs, context_limit=10_000):
        self.outputs = iter(outputs)
        self.config = SimpleNamespace(max_position_embeddings=context_limit)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        generated_rows = next(self.outputs)
        if generated_rows and isinstance(generated_rows[0], int):
            generated_rows = [generated_rows]
        input_ids = kwargs["input_ids"]
        rows = [
            torch.cat((input_ids[index], torch.tensor(generated)))
            for index, generated in enumerate(generated_rows)
        ]
        sequences = torch.stack(rows)
        steps = len(generated_rows[0])
        scores = tuple(torch.zeros(len(rows), 1) for _ in range(steps))
        return SimpleNamespace(sequences=sequences, scores=scores, beam_indices=None)

    def compute_transition_scores(self, sequences, scores, beam_indices, normalize_logits):
        return torch.tensor(
            [[-(index + 1) / 10 for index in range(len(scores))]
             for _ in range(sequences.shape[0])]
        )


def _config(**changes):
    values = {
        "model_name_or_path": "organization/model",
        "dataset_path": Path("C:/data.jsonl"),
        "reasoning_mode": "direct",
        "output_dir": Path("C:/outputs"),
        "answer_max_new_tokens": 3,
        "reasoning_max_new_tokens": 4,
    }
    values.update(changes)
    return TaskConfig(**values)


def _loaded(model, tokenizer):
    return LoadedModel(model, tokenizer, "cpu", "float32", "revision")


class GenerationTests(unittest.TestCase):
    def test_direct_instructions_markers_batches_scores_and_seeds(self):
        tokenizer = _Tokenizer()
        answer = tokenizer.encode("OK") + [tokenizer.eos_token_id]
        model = _Model([[answer] * 8, [answer] * 2])
        examples = [
            {"id": index, "input": f"Task {index}", "split": "train"}
            for index in range(10)
        ]
        seed = Mock()

        with patch(
            "src.generation._runtime",
            return_value=(torch, StoppingCriteriaList, seed),
        ):
            units = list(generation_units(_loaded(model, tokenizer), examples, _config()))

        self.assertEqual([len(unit.records) for unit in units], [8, 2])
        self.assertEqual(
            [call.args[0] for call in seed.call_args_list],
            [_stable_seed(42, 0), _stable_seed(42, 8)],
        )
        expected_instruction = (
            f"{ANSWER_INSTRUCTION}\n{ALLOW_ABSTENTION_INSTRUCTION}"
        )
        self.assertEqual(
            tokenizer.rendered_contents[0], f"Task 0\n\n{expected_instruction}"
        )
        self.assertEqual(tokenizer.rendered_modes, [False] * 10)
        record = units[0].records[0]
        marker_ids = tokenizer.encode(FINAL_MARKER)
        self.assertNotEqual(
            record["formatted_prompt_token_ids"][-len(marker_ids):], marker_ids
        )
        self.assertEqual(record["final_control"]["token_ids"], marker_ids)
        self.assertEqual(record["final_control"]["final_marker_token_ids"], marker_ids)
        self.assertEqual(
            record["final_control"]["answer_instruction_token_ids"],
            tokenizer.encode(expected_instruction),
        )
        self.assertEqual(record["answer"]["text"], "OK")
        for actual, expected in zip(
            record["answer"]["token_logprobs"], (-0.1, -0.2)
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(record["answer"]["stop_reason"], "eos")
        for call in model.calls:
            self.assertNotIn("do_sample", call)
            self.assertNotIn("temperature", call)
            self.assertTrue(call["output_scores"])

    def test_direct_abstention_disabled_uses_exact_sentence(self):
        tokenizer = _Tokenizer()
        model = _Model([[[tokenizer.eos_token_id]]])
        examples = [{"id": 1, "input": "Task", "split": "test"}]
        list(
            generation_units(
                _loaded(model, tokenizer),
                examples,
                _config(allow_abstention=False),
            )
        )
        self.assertEqual(
            tokenizer.rendered_contents,
            [f"Task\n\n{ANSWER_INSTRUCTION}\n{DISALLOW_ABSTENTION_INSTRUCTION}"],
        )

    def test_fixed_system_prompt_precedes_example_user_message(self):
        tokenizer = _Tokenizer()
        model = _Model([[[tokenizer.eos_token_id]]])
        examples = [{"id": 1, "input": "Example input", "split": "test"}]

        list(
            generation_units(
                _loaded(model, tokenizer),
                examples,
                _config(system_prompt="Fixed instructions"),
            )
        )

        self.assertEqual(
            tokenizer.rendered_messages,
            [
                [
                    {"role": "system", "content": "Fixed instructions"},
                    {
                        "role": "user",
                        "content": (
                            f"Example input\n\n{ANSWER_INSTRUCTION}\n"
                            f"{ALLOW_ABSTENTION_INSTRUCTION}"
                        ),
                    },
                ]
            ],
        )

    def test_reasoning_boundary_is_replaced_by_canonical_control(self):
        tokenizer = _Tokenizer()
        thought = tokenizer.encode("work")
        boundary = tokenizer.encode(THINKING_BOUNDARY)
        answer = tokenizer.encode("A") + [tokenizer.eos_token_id]
        model = _Model([[*thought, *boundary], answer])
        example = {"id": 4, "input": "Solve", "split": "validation"}

        unit = next(
            generation_units(
                _loaded(model, tokenizer),
                [example],
                _config(reasoning_mode="reasoning"),
            )
        )
        record = unit.records[0]
        self.assertEqual(
            tokenizer.rendered_contents, [f"Solve\n\n{REASONING_INSTRUCTION}"]
        )
        self.assertEqual(tokenizer.rendered_modes, [True])
        self.assertEqual(record["reasoning"]["text"], "work")
        self.assertEqual(record["reasoning"]["stop_reason"], "thinking_boundary")
        self.assertFalse(record["reasoning"]["thinking_boundary_forced"])
        self.assertEqual(
            record["final_control"]["token_ids"][:len(boundary)], boundary
        )
        marker_start, marker_end = record["final_control"]["final_marker_span"]
        self.assertEqual(
            record["final_control"]["token_ids"][marker_start:marker_end],
            tokenizer.encode(FINAL_MARKER),
        )

    def test_reasoning_eos_and_limit_force_closure(self):
        tokenizer = _Tokenizer()
        cases = (
            ([*tokenizer.encode("x"), tokenizer.eos_token_id], "eos"),
            (tokenizer.encode("long"), "token_limit"),
        )
        for reasoning_output, expected_stop in cases:
            with self.subTest(stop=expected_stop):
                model = _Model([reasoning_output, [tokenizer.eos_token_id]])
                record = next(
                    generation_units(
                        _loaded(model, tokenizer),
                        [{"id": 1, "input": "Task", "split": "train"}],
                        _config(reasoning_mode="reasoning"),
                    )
                ).records[0]
                self.assertEqual(record["reasoning"]["stop_reason"], expected_stop)
                self.assertTrue(record["reasoning"]["thinking_boundary_forced"])

    def test_reasoning_abstention_disabled_uses_exact_final_instruction(self):
        tokenizer = _Tokenizer()
        model = _Model([
            [tokenizer.eos_token_id],
            [tokenizer.eos_token_id],
        ])
        record = next(
            generation_units(
                _loaded(model, tokenizer),
                [{"id": 1, "input": "Task", "split": "train"}],
                _config(reasoning_mode="reasoning", allow_abstention=False),
            )
        ).records[0]
        self.assertEqual(
            record["final_control"]["answer_instruction"],
            f"{ANSWER_INSTRUCTION}\n{DISALLOW_ABSTENTION_INSTRUCTION}",
        )

    def test_context_length_is_a_recoverable_record_failure(self):
        tokenizer = _Tokenizer()
        model = _Model([], context_limit=5)
        unit = next(
            generation_units(
                _loaded(model, tokenizer),
                [{"id": 9, "input": "Too long", "split": "test"}],
                _config(),
            )
        )
        self.assertEqual(unit.records[0]["status"], "failed")
        self.assertEqual(unit.records[0]["failure"]["reason"], "context_length")
        self.assertEqual(model.calls, [])

    def test_resume_batches_only_missing_direct_examples(self):
        tokenizer = _Tokenizer()
        answer = tokenizer.encode("A") + [tokenizer.eos_token_id]
        model = _Model([[[*answer]] * 7])
        examples = [
            {"id": index, "input": str(index), "split": "train"} for index in range(8)
        ]
        self.assertEqual(
            list(generation_units(_loaded(model, tokenizer), examples, _config(), range(8))),
            [],
        )
        units = list(generation_units(
            _loaded(model, tokenizer), examples, _config(), [0]
        ))
        self.assertEqual([record["id"] for record in units[0].records], list(range(1, 8)))

    def test_sampled_direct_generation_uses_one_id_seed_per_example(self):
        tokenizer = _Tokenizer()
        answer = tokenizer.encode("A") + [tokenizer.eos_token_id]
        model = _Model([answer, answer])
        model.generation_config = SimpleNamespace(do_sample=True)
        examples = [
            {"id": 10, "input": "first", "split": "train"},
            {"id": 20, "input": "second", "split": "train"},
        ]
        seed = Mock()
        with patch(
            "src.generation._runtime",
            return_value=(torch, StoppingCriteriaList, seed),
        ):
            units = list(generation_units(_loaded(model, tokenizer), examples, _config()))
        self.assertEqual([len(unit.records) for unit in units], [1, 1])
        self.assertEqual(
            [call.args[0] for call in seed.call_args_list],
            [_stable_seed(42, 10), _stable_seed(42, 20)],
        )


if __name__ == "__main__":
    unittest.main()
