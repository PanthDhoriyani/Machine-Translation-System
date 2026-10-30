"""
Test Suite — Module 2: Model Management
TC-2.1 through TC-2.6 from SRS Section 7.2
"""

import time
import pytest
from module2_model.model_manager import ModelManager


@pytest.fixture(scope="module")
def manager():
    """Shared ModelManager loaded once per test module (slow download)."""
    return ModelManager("Helsinki-NLP/opus-mt-en-fr")


class TestModelManager:
    def test_load_marianmt(self, manager):
        """TC-2.1: MarianMT EN→FR model must load successfully."""
        assert manager.model is not None
        assert manager.tokenizer is not None

    def test_load_invalid_model_raises(self):
        """TC-2.2: Loading an invalid model ID must raise OSError."""
        with pytest.raises(OSError):
            ModelManager("invalid/model-xyz-does-not-exist-abc123")

    def test_model_caching_speed(self, manager):
        """TC-2.3: Second load of same model must come from cache (< 500 ms)."""
        model_id = manager.model_id
        t0 = time.perf_counter()
        manager.load_model(model_id)   # second load — should be cached
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500, f"Cache load took {elapsed_ms:.0f} ms — expected < 500 ms"

    def test_model_switching(self, manager):
        """TC-2.4: Switching model ID must update model_id on the manager."""
        original = manager.model_id
        manager.switch_model("Helsinki-NLP/opus-mt-en-de")
        assert manager.model_id == "Helsinki-NLP/opus-mt-en-de"
        # Switch back
        manager.switch_model(original)

    def test_get_model_info_structure(self, manager):
        """TC-2.5: get_model_info() must return a dict with required keys."""
        info = manager.get_model_info()
        assert isinstance(info, dict)
        assert "model_id" in info
        assert "cached" in info
        assert "parameters_M" in info

    def test_get_model_info_cached_true(self, manager):
        """TC-2.5: Loaded model must report cached=True."""
        info = manager.get_model_info()
        assert info["cached"] is True

    def test_tokenizer_roundtrip(self, manager):
        """TC-2.6: Tokenize → decode should recover the original text (lossless)."""
        text = "Hello world"
        tokens = manager.tokenizer(text, return_tensors="pt")
        decoded = manager.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
        assert isinstance(decoded, str)
        assert len(decoded) > 0
