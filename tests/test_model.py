"""Offline tests for model loading and compatibility checks."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.config import TaskConfig
from src.model import ModelLoadError, load_model, load_tokenizer


class _Cuda:
    def __init__(self, available=True, count=2):
        self.available = available
        self.count = count

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count


class _Tokenizer:
    eos_token_id = 2
    eos_token = "</s>"
    pad_token_id = None
    pad_token = None

    def get_chat_template(self):
        return "{{ messages }}"


class _Model:
    dtype = "torch.float16"
    hf_device_map = {"": "cuda:0"}

    def __init__(self, can_generate=True):
        self._can_generate = can_generate
        self.eval_called = False

    def can_generate(self):
        return self._can_generate

    def eval(self):
        self.eval_called = True
        return self


class ModelTests(unittest.TestCase):
    def config(self, **changes):
        values = {
            "model_name_or_path": "organization/model",
            "dataset_path": Path("C:/data/dataset.jsonl"),
            "reasoning_mode": "direct",
            "output_dir": Path("C:/results"),
        }
        values.update(changes)
        return TaskConfig(**values)

    def dependencies(self, *, model_config=None, tokenizer=None, model=None,
                     cuda_available=True, cuda_count=2):
        torch = SimpleNamespace(
            cuda=_Cuda(cuda_available, cuda_count),
            float16="torch.float16",
            bfloat16="torch.bfloat16",
            float32="torch.float32",
        )
        auto_config = Mock()
        auto_config.from_pretrained.return_value = model_config or SimpleNamespace(
            is_encoder_decoder=False,
            vision_config=None,
            _commit_hash="resolved-commit",
        )
        auto_tokenizer = Mock()
        auto_tokenizer.from_pretrained.return_value = tokenizer or _Tokenizer()
        auto_model = Mock()
        auto_model.from_pretrained.return_value = model or _Model()
        return torch, auto_config, auto_model, auto_tokenizer

    def load(self, config=None, dependencies=None):
        dependencies = dependencies or self.dependencies()
        with patch("src.model._dependencies", return_value=dependencies):
            return load_model(config or self.config()), dependencies

    def test_loads_one_resolved_hub_revision_and_returns_metadata(self):
        loaded, (_, auto_config, auto_model, auto_tokenizer) = self.load()

        auto_config.from_pretrained.assert_called_once_with(
            "organization/model", trust_remote_code=False
        )
        auto_tokenizer.from_pretrained.assert_called_once_with(
            "organization/model", trust_remote_code=False,
            revision="resolved-commit",
        )
        auto_model.from_pretrained.assert_called_once_with(
            "organization/model",
            config=auto_config.from_pretrained.return_value,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=False,
            revision="resolved-commit",
        )
        self.assertEqual(loaded.resolved_revision, "resolved-commit")
        self.assertEqual(loaded.resolved_device, {"": "cuda:0"})
        self.assertEqual(loaded.resolved_dtype, "float16")
        self.assertTrue(loaded.model.eval_called)
        self.assertEqual(loaded.tokenizer.pad_token, "</s>")

    def test_tokenizer_can_load_without_weights_and_be_reused(self):
        dependencies = self.dependencies()
        with patch("src.model._dependencies", return_value=dependencies):
            tokenizer = load_tokenizer(
                self.config(), pinned_revision="resolved-commit"
            )
            loaded = load_model(
                self.config(),
                pinned_revision="resolved-commit",
                tokenizer=tokenizer,
            )

        self.assertIs(loaded.tokenizer, tokenizer)
        dependencies[3].from_pretrained.assert_called_once()
        dependencies[2].from_pretrained.assert_called_once()

    def test_forwards_requested_revision_and_explicit_loading_settings(self):
        config = self.config(
            model_revision="requested-tag", device="cuda:1", dtype="bfloat16"
        )
        loaded, (_, auto_config, auto_model, _) = self.load(config)

        auto_config.from_pretrained.assert_called_once_with(
            "organization/model", trust_remote_code=False,
            revision="requested-tag",
        )
        self.assertEqual(
            auto_model.from_pretrained.call_args.kwargs["torch_dtype"],
            "torch.bfloat16",
        )
        self.assertEqual(auto_model.from_pretrained.call_args.kwargs["device_map"], "cuda:1")
        self.assertEqual(loaded.resolved_revision, "resolved-commit")

    def test_local_checkpoint_version_is_recorded_but_not_forwarded(self):
        source = str(Path("C:/models/local-checkpoint"))
        config = self.config(
            model_name_or_path=source,
            model_revision="checkpoint-v3",
        )
        loaded, (_, auto_config, auto_model, auto_tokenizer) = self.load(config)

        auto_config.from_pretrained.assert_called_once_with(
            source, trust_remote_code=False
        )
        auto_tokenizer.from_pretrained.assert_called_once_with(
            source, trust_remote_code=False
        )
        self.assertNotIn("revision", auto_model.from_pretrained.call_args.kwargs)
        self.assertEqual(loaded.resolved_revision, "checkpoint-v3")

    def test_local_checkpoint_requires_version_even_for_direct_api_use(self):
        with self.assertRaisesRegex(ModelLoadError, "requires model_revision"):
            self.load(self.config(model_name_or_path="C:/models/local-checkpoint"))

    def test_pinned_revision_overrides_a_moving_requested_tag(self):
        dependencies = self.dependencies()
        with patch("src.model._dependencies", return_value=dependencies):
            loaded = load_model(
                self.config(model_revision="main"),
                pinned_revision="resolved-commit",
            )
        dependencies[1].from_pretrained.assert_called_once_with(
            "organization/model",
            trust_remote_code=False,
            revision="resolved-commit",
        )
        self.assertEqual(loaded.resolved_revision, "resolved-commit")

    def test_rejects_a_pinned_revision_that_does_not_resolve_exactly(self):
        dependencies = self.dependencies(
            model_config=SimpleNamespace(
                is_encoder_decoder=False,
                vision_config=None,
                _commit_hash="different-commit",
            )
        )
        with patch("src.model._dependencies", return_value=dependencies):
            with self.assertRaisesRegex(ModelLoadError, "does not match the pinned"):
                load_model(self.config(), pinned_revision="resolved-commit")
        dependencies[3].from_pretrained.assert_not_called()

    def test_rejects_unsupported_architectures_before_loading_weights(self):
        cases = (
            (SimpleNamespace(is_encoder_decoder=True, vision_config=None), "decoder-only"),
            (SimpleNamespace(is_encoder_decoder=False, vision_config={}), "text-only"),
        )
        for model_config, message in cases:
            with self.subTest(message=message):
                dependencies = self.dependencies(model_config=model_config)
                with self.assertRaisesRegex(ModelLoadError, message):
                    self.load(dependencies=dependencies)
                dependencies[2].from_pretrained.assert_not_called()

    def test_requires_an_exact_hub_revision(self):
        model_config = SimpleNamespace(
            is_encoder_decoder=False,
            vision_config=None,
            _commit_hash=None,
        )
        dependencies = self.dependencies(model_config=model_config)
        with self.assertRaisesRegex(ModelLoadError, "exact Hub revision"):
            self.load(dependencies=dependencies)
        dependencies[3].from_pretrained.assert_not_called()
        dependencies[2].from_pretrained.assert_not_called()

    def test_rejects_missing_chat_template_before_loading_weights(self):
        tokenizer = Mock()
        tokenizer.get_chat_template.side_effect = ValueError("missing")
        dependencies = self.dependencies(tokenizer=tokenizer)
        with self.assertRaisesRegex(ModelLoadError, "chat template"):
            self.load(dependencies=dependencies)
        dependencies[2].from_pretrained.assert_not_called()

    def test_rejects_missing_eos_token_before_loading_weights(self):
        tokenizer = _Tokenizer()
        tokenizer.eos_token_id = None
        dependencies = self.dependencies(tokenizer=tokenizer)
        with self.assertRaisesRegex(ModelLoadError, "no EOS token"):
            self.load(dependencies=dependencies)
        dependencies[2].from_pretrained.assert_not_called()

    def test_rejects_unavailable_or_out_of_range_cuda(self):
        cases = (
            ("cuda", self.dependencies(cuda_available=False), "unavailable"),
            ("cuda:2", self.dependencies(cuda_count=2), "only 2"),
        )
        for device, dependencies, message in cases:
            with self.subTest(device=device):
                with self.assertRaisesRegex(ModelLoadError, message):
                    self.load(self.config(device=device), dependencies)
                dependencies[1].from_pretrained.assert_not_called()

    def test_rejects_model_without_generation_support(self):
        dependencies = self.dependencies(model=_Model(can_generate=False))
        with self.assertRaisesRegex(ModelLoadError, "does not support text generation"):
            self.load(dependencies=dependencies)


if __name__ == "__main__":
    unittest.main()
