"""Offline checks for linear probe training and storage."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.probe import (
    PROBE_FILE,
    ProbeError,
    build_probe_identity,
    train_probes,
    validate_probe_group,
)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.activation_path = self.directory / "activations.h5"
        self.evaluations = {
            1: self.evaluation(1, "train", "correct"),
            2: self.evaluation(2, "train", "correct"),
            3: self.evaluation(3, "train", "incorrect"),
            4: self.evaluation(4, "train", "incorrect"),
            5: self.evaluation(5, "validation", "correct"),
            6: self.evaluation(6, "validation", "incorrect"),
            # Deliberately has no activation group. Probe training must not read test.
            7: self.evaluation(7, "test", "correct"),
        }
        self.write_activations()

    @staticmethod
    def evaluation(example_id, split, outcome):
        return {
            "id": example_id,
            "split": split,
            "outcome": outcome,
            "is_correct": outcome == "correct",
        }

    def write_activations(self, *, nonfinite=False):
        labels = ("embedding", "hidden_state_1")
        with h5py.File(self.activation_path, "w") as output:
            output.attrs["completed"] = True
            output.attrs["state_labels"] = json.dumps(labels)
            output.attrs["state_count"] = len(labels)
            output.attrs["hidden_size"] = 2
            examples = output.create_group("examples")
            for example_id, record in list(self.evaluations.items())[:-1]:
                sign = 1.0 if record["outcome"] == "correct" else -1.0
                group = examples.create_group(str(example_id))
                group.attrs["split"] = record["split"]
                prompt = np.asarray(
                    [
                        [[sign * example_id, sign * 2]],
                        [[sign * 2 * example_id, sign * 3]],
                    ],
                    dtype=np.float16,
                )
                answer = np.asarray(
                    [
                        [[sign, 0], [sign * 3, sign * 2]],
                        [[sign * 2, sign], [sign * 4, sign * 3]],
                    ],
                    dtype=np.float16,
                )
                if nonfinite and example_id == 1:
                    prompt[0, 0, 0] = np.nan
                group.create_dataset("prompt_end", data=prompt)
                group.create_dataset("answer_tokens", data=answer)

    @staticmethod
    def identity(seed=42):
        return build_probe_identity("a" * 64, "e" * 64, seed)

    def test_trains_every_position_and_state_without_reading_test(self):
        progress = []
        identity = self.identity()
        result = train_probes(
            self.directory,
            identity,
            self.evaluations,
            lambda done, total, _started, starting: progress.append(
                (done, total, starting)
            ),
        )

        self.assertEqual(result.trained_candidates, 4)
        self.assertEqual(progress[-1], (4, 4, 0))
        self.assertEqual(result.summary["candidates"], 4)
        self.assertEqual(result.summary["score"], "probability_correct")
        self.assertEqual(result.summary["token_pooling"], "mean")

        with h5py.File(self.directory / PROBE_FILE, "r") as source:
            group = source[f"trainings/{identity.probe_id}"]
            self.assertEqual(group["train_ids"][...].tolist(), [1, 2, 3, 4])
            self.assertEqual(group["validation_ids"][...].tolist(), [5, 6])
            self.assertEqual(group["train_labels"][...].tolist(), [1, 1, 0, 0])
            settings = json.loads(group.attrs["settings"])
            self.assertEqual(settings["class_weight"], "balanced")
            self.assertEqual(settings["regularization_C"], 1.0)

            answer = group["position_models/answer_tokens"]
            # The first state averages [1, 0] and [3, 2] for example 1.
            expected_train = np.asarray(
                [[2, 1], [2, 1], [-2, -1], [-2, -1]], dtype=np.float32
            )
            np.testing.assert_allclose(
                answer["scaler_mean"][0], expected_train.mean(axis=0)
            )
            validation = np.asarray([[2, 1], [-2, -1]], dtype=np.float64)
            scaled = (
                validation - answer["scaler_mean"][0]
            ) / answer["scaler_scale"][0]
            logits = (
                scaled @ answer["coefficients"][0]
                + answer["intercepts"][0]
            )
            expected_scores = 1.0 / (1.0 + np.exp(-logits))
            np.testing.assert_allclose(
                answer["validation_scores"][0], expected_scores, rtol=1e-12
            )
            self.assertGreater(expected_scores[0], expected_scores[1])

        content_hash, summary = validate_probe_group(
            self.directory, identity, result.content_sha256
        )
        self.assertEqual(content_hash, result.content_sha256)
        self.assertEqual(summary, result.summary)

        reused = train_probes(self.directory, identity, self.evaluations)
        self.assertEqual((reused.starting_candidates, reused.trained_candidates), (4, 0))

        repeat_directory = self.directory / "repeat"
        repeat_directory.mkdir()
        shutil.copy2(self.activation_path, repeat_directory / "activations.h5")
        repeated = train_probes(repeat_directory, identity, self.evaluations)
        self.assertEqual(repeated.content_sha256, result.content_sha256)

        second = train_probes(self.directory, self.identity(7), self.evaluations)
        self.assertEqual(second.trained_candidates, 4)
        with h5py.File(self.directory / PROBE_FILE, "r") as source:
            self.assertEqual(len(source["trainings"]), 2)

    def test_interrupted_training_resumes_missing_candidates(self):
        calls = []

        def interrupt(done, *_):
            calls.append(done)
            raise RuntimeError("interrupted")

        with self.assertRaisesRegex(ProbeError, "Cannot train or save probes"):
            train_probes(
                self.directory, self.identity(), self.evaluations, interrupt
            )
        self.assertEqual(calls, [1])

        resumed = train_probes(self.directory, self.identity(), self.evaluations)
        self.assertEqual(
            (resumed.starting_candidates, resumed.trained_candidates), (1, 3)
        )

    def test_nonfinite_convergence_and_corruption_fail_clearly(self):
        self.write_activations(nonfinite=True)
        with self.assertRaisesRegex(ProbeError, "nonfinite.*prompt_end.*embedding"):
            train_probes(self.directory, self.identity(), self.evaluations)

        (self.directory / PROBE_FILE).unlink()
        self.write_activations()
        with patch(
            "sklearn.linear_model.LogisticRegression.fit",
            side_effect=ConvergenceWarning("failed"),
        ):
            with self.assertRaisesRegex(ProbeError, "did not converge"):
                train_probes(self.directory, self.identity(), self.evaluations)

        (self.directory / PROBE_FILE).unlink()
        result = train_probes(self.directory, self.identity(), self.evaluations)
        with h5py.File(self.directory / PROBE_FILE, "a") as source:
            coefficients = source[
                f"trainings/{self.identity().probe_id}/"
                "position_models/prompt_end/coefficients"
            ]
            coefficients[0, 0] += 1
        with self.assertRaisesRegex(ProbeError, "has changed"):
            validate_probe_group(
                self.directory, self.identity(), result.content_sha256
            )


if __name__ == "__main__":
    unittest.main()
