"""Offline checks for exact replay and default hidden-state capture."""

from math import exp, log
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from src.activation import (
    ActivationError,
    build_replay_plan,
    capture_hidden_states,
)
from src.activation_store import (
    activation_file_path,
    append_activation_record,
    build_activation_context,
    build_activation_identity,
    finalize_activation_file,
    load_activation_records,
    reuse_activation_records,
)
from src.config import TaskConfig
from src.generation_cache import build_generation_context
from src.model import LoadedModel


VOCAB_SIZE = 16
TOKEN_LOGPROB = 2.0 - log(exp(2.0) + VOCAB_SIZE - 1)


def _generation(*, reasoning=False, answer_ids=(5,)):
    reasoning_record = (
        {
            "text": "reason",
            "token_ids": [7, 8],
            "tokens": 2,
            "stop_reason": "thinking_boundary",
            "thinking_boundary_forced": False,
        }
        if reasoning
        else None
    )
    control_ids = [9, 10, 11] if reasoning else [3, 4]
    marker_span = [2, 3] if reasoning else [0, 2]
    return {
        "id": 1,
        "split": "train",
        "status": "success",
        "formatted_prompt_token_ids": [1, 2] if not reasoning else [1],
        "reasoning": reasoning_record,
        "final_control": {
            "token_ids": control_ids,
            "final_marker_span": marker_span,
        },
        "answer": {
            "text": "answer",
            "token_ids": list(answer_ids),
            "token_logprobs": [TOKEN_LOGPROB] * len(answer_ids),
        },
    }


class _ReplayModel:
    device = torch.device("cpu")

    def __init__(self, answer_positions, answer_ids):
        self.answer_positions = answer_positions
        self.answer_ids = answer_ids

    def __call__(self, *, input_ids, attention_mask, **settings):
        self.last_input_ids = input_ids.tolist()[0]
        self.last_attention_mask = attention_mask.tolist()[0]
        self.last_settings = settings
        length = input_ids.shape[1]
        logits = torch.zeros(1, length, VOCAB_SIZE)
        for position, token_id in zip(self.answer_positions, self.answer_ids):
            logits[0, position - 1, token_id] = 2.0
        base = torch.arange(length * 4, dtype=torch.float32).reshape(1, length, 4)
        hidden_states = (base, base + 100, base + 200)
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


def _loaded(model):
    return LoadedModel(model, object(), "cpu", "float32", "commit")


class ActivationTests(unittest.TestCase):
    def test_direct_replay_positions_and_single_token_shape(self):
        plan = build_replay_plan(_generation())
        self.assertEqual(plan.token_ids, (1, 2, 3, 4, 5))
        self.assertEqual(
            plan.positions,
            {"prompt_end": (1,), "final_prompt_end": (3,), "answer_tokens": (4,)},
        )
        model = _ReplayModel(plan.positions["answer_tokens"], plan.answer_token_ids)

        result = capture_hidden_states(_loaded(model), plan)

        self.assertEqual(model.last_input_ids, list(plan.token_ids))
        self.assertEqual(model.last_attention_mask, [1] * len(plan.token_ids))
        self.assertEqual(result.state_labels, ("embedding", "hidden_state_1", "hidden_state_2"))
        self.assertEqual(result.tensors["prompt_end"].shape, (3, 1, 4))
        self.assertEqual(result.tensors["answer_tokens"].shape, (3, 1, 4))
        self.assertEqual(result.tensors["answer_tokens"].dtype, torch.float16)
        self.assertLess(result.max_logprob_difference, 1e-6)

    def test_reasoning_replay_and_multi_token_answer(self):
        plan = build_replay_plan(_generation(reasoning=True, answer_ids=(5, 6)))
        self.assertEqual(plan.token_ids, (1, 7, 8, 9, 10, 11, 5, 6))
        self.assertEqual(plan.positions["prompt_end"], (0,))
        self.assertEqual(plan.positions["final_prompt_end"], (5,))
        self.assertEqual(plan.positions["answer_tokens"], (6, 7))
        model = _ReplayModel(plan.positions["answer_tokens"], plan.answer_token_ids)

        result = capture_hidden_states(_loaded(model), plan)

        self.assertEqual(result.tensors["answer_tokens"].shape, (3, 2, 4))

    def test_replay_probability_mismatch_is_rejected(self):
        record = _generation()
        record["answer"]["token_logprobs"] = [-20.0]
        plan = build_replay_plan(record)
        model = _ReplayModel(plan.positions["answer_tokens"], plan.answer_token_ids)
        with self.assertRaisesRegex(ActivationError, "probability mismatch"):
            capture_hidden_states(_loaded(model), plan)

    def test_invalid_saved_boundaries_are_rejected(self):
        record = _generation()
        record["final_control"]["final_marker_span"] = [2, 2]
        with self.assertRaisesRegex(ActivationError, "final_marker_span"):
            build_replay_plan(record)

    def test_one_hdf5_file_supports_resume_and_cross_run_reuse(self):
        import h5py

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            output.mkdir()
            config = TaskConfig(
                model_name_or_path="organization/model",
                dataset_path=root / "dataset.jsonl",
                reasoning_mode="direct",
                output_dir=output,
            )
            generation_context = build_generation_context(config)
            metadata = {
                "identifier": "organization/model",
                "requested_revision": None,
                "resolved_revision": "commit",
                "model_class": "Model",
                "tokenizer_class": "Tokenizer",
                "dtype": "float32",
                "device": "cpu",
                "torch_version": "test",
                "transformers_version": "test",
            }
            context = build_activation_context(config, generation_context, metadata)
            plan = build_replay_plan(_generation())
            model = _ReplayModel(plan.positions["answer_tokens"], plan.answer_token_ids)
            result = capture_hidden_states(_loaded(model), plan)
            example = {"id": 1, "input": "task", "split": "train"}
            second_plan = build_replay_plan(
                _generation(reasoning=True, answer_ids=(5, 6))
            )
            second_model = _ReplayModel(
                second_plan.positions["answer_tokens"], second_plan.answer_token_ids
            )
            second_result = capture_hidden_states(_loaded(second_model), second_plan)
            second_example = {"id": 2, "input": "task 2", "split": "test"}
            expected = {
                1: (example, "a" * 64, plan),
                2: (second_example, "b" * 64, second_plan),
            }

            first_run = output / "first"
            first_run.mkdir()
            first_identity = build_activation_identity(context, "g" * 64)
            stored = append_activation_record(
                first_run,
                first_identity,
                context,
                example,
                "a" * 64,
                plan,
                result,
            )
            append_activation_record(
                first_run,
                first_identity,
                context,
                second_example,
                "b" * 64,
                second_plan,
                second_result,
            )
            self.assertEqual(set(first_run.iterdir()), {activation_file_path(first_run)})
            with h5py.File(activation_file_path(first_run), "a") as source:
                source["_pending"].create_group("unfinished")

            records, recovered = load_activation_records(
                first_run, first_identity, context, expected
            )
            self.assertTrue(recovered)
            self.assertEqual(records[1], stored)
            with h5py.File(activation_file_path(first_run), "r") as source:
                self.assertEqual(list(source["_pending"]), [])
                self.assertEqual(
                    source["examples/1/answer_tokens"].shape,
                    (3, 1, 4),
                )
                self.assertEqual(source["examples/1/answer_tokens"].dtype.name, "float16")
                self.assertEqual(
                    source["examples/2/answer_tokens"].shape,
                    (3, 2, 4),
                )

            artifact_hash = finalize_activation_file(
                first_run, first_identity, context, expected, records
            )
            (first_run / "run.json").write_text(
                json.dumps(
                    {
                        "completed_stages": ["activation_capture"],
                        "activation_capture": {
                            "context_fingerprint": context.fingerprint,
                            "artifact": "activations.h5",
                            "artifact_sha256": artifact_hash,
                        },
                    }
                ),
                encoding="utf-8",
            )

            second_run = output / "second"
            second_run.mkdir()
            second_identity = build_activation_identity(context, "h" * 64)
            reused = reuse_activation_records(
                output,
                second_run,
                second_identity,
                context,
                expected,
                set(),
            )
            self.assertEqual(set(reused), {1, 2})
            copied, _ = load_activation_records(
                second_run, second_identity, context, expected
            )
            self.assertEqual(set(copied), {1, 2})
            self.assertEqual(set(second_run.iterdir()), {activation_file_path(second_run)})


if __name__ == "__main__":
    unittest.main()
