"""
Test Suite — Module 1: Preprocessing & Dataset Management
TC-1.1 through TC-1.8 from SRS Section 7.1
"""

import pytest
import pandas as pd
import os
import tempfile

from module1_preprocessing.text_input import TextInput
from module1_preprocessing.preprocessor import Preprocessor
from module1_preprocessing.dataset_manager import DatasetManager


# ── TextInput tests ──────────────────────────────────────────────────────

class TestTextInput:
    def test_validate_empty_raises(self):
        """TC-1.4: Empty input must raise ValueError."""
        ti = TextInput("", tgt_lang="fr")
        with pytest.raises(ValueError, match="must not be empty"):
            ti.validate()

    def test_validate_whitespace_raises(self):
        """TC-1.4: Whitespace-only input must raise ValueError."""
        ti = TextInput("   ", tgt_lang="fr")
        with pytest.raises(ValueError):
            ti.validate()

    def test_validate_valid_returns_true(self):
        ti = TextInput("Hello, how are you?", tgt_lang="fr")
        assert ti.validate() is True

    def test_detect_language_english(self):
        """TC-1.1: English text should be detected as 'en'."""
        ti = TextInput("Hello, how are you?", tgt_lang="fr")
        lang = ti.detect_language()
        assert lang == "en"

    def test_detect_language_french(self):
        """TC-1.2: French text should be detected as 'fr'."""
        ti = TextInput("Bonjour, comment allez-vous?", tgt_lang="en")
        lang = ti.detect_language()
        assert lang == "fr"

    def test_explicit_src_lang_not_overridden(self):
        """If src_lang is provided, detect_language should return it as-is."""
        ti = TextInput("Hello", src_lang="de", tgt_lang="fr")
        assert ti.detect_language() == "de"

    def test_sanitize_html(self):
        """TC-1.8: HTML tags must be stripped from input."""
        ti = TextInput("<script>alert()</script>Hello", tgt_lang="fr")
        result = ti.sanitize_html()
        assert "<" not in result
        assert "Hello" in result


# ── Preprocessor tests ───────────────────────────────────────────────────

class TestPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor with a lightweight tokenizer for testing."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        return Preprocessor(tokenizer)

    def test_unicode_normalization(self, preprocessor):
        """TC-1.3: Unicode 'café' must be preserved after normalization."""
        result = preprocessor.normalize("caf\u00e9")
        assert result == "café"

    def test_html_removal(self, preprocessor):
        """TC-1.8: HTML must be removed during normalization."""
        result = preprocessor.normalize("<b>Hello</b> world")
        assert "<b>" not in result
        assert "Hello" in result

    def test_tokenize_output_has_input_ids(self, preprocessor):
        """Tokenizer must return dict with input_ids."""
        out = preprocessor.tokenize("Hello world")
        assert "input_ids" in out

    def test_tokenize_respects_max_length(self, preprocessor):
        """TC-1.5: Token count must not exceed max_length."""
        long_text = " ".join(["word"] * 600)
        out = preprocessor.tokenize(long_text)
        assert out["input_ids"].shape[-1] <= preprocessor.max_length

    def test_detokenize_roundtrip(self, preprocessor):
        """TC-2.6 / general: tokenize → detokenize should recover readable text."""
        text = "Hello world"
        enc = preprocessor.tokenize(text)
        decoded = preprocessor.detokenize(enc["input_ids"][0])
        assert isinstance(decoded, str)
        assert len(decoded) > 0


# ── DatasetManager tests ─────────────────────────────────────────────────

class TestDatasetManager:
    @pytest.fixture
    def manager(self):
        return DatasetManager()

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a temporary bilingual CSV file."""
        df = pd.DataFrame({"src": [f"sentence {i}" for i in range(100)],
                           "tgt": [f"phrase {i}" for i in range(100)]})
        path = tmp_path / "corpus.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_load_csv(self, manager, sample_csv):
        """TC-1.6: CSV file must load as DataFrame with src/tgt columns."""
        df = manager.load_corpus(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert "src" in df.columns
        assert "tgt" in df.columns

    def test_load_nonexistent_raises(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.load_corpus("/nonexistent/path.csv")

    def test_load_unsupported_extension(self, manager, tmp_path):
        path = tmp_path / "corpus.xml"
        path.write_text("<root/>")
        with pytest.raises(ValueError, match="Unsupported"):
            manager.load_corpus(str(path))

    def test_split_800_100_100(self, manager, sample_csv):
        """TC-1.7: 1000 pairs should split 800/100/100 (here 100→80/10/10)."""
        df = manager.load_corpus(sample_csv)
        train, val, test = manager.split_dataset(df)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_split_ratios_must_sum_to_one(self, manager, sample_csv):
        df = manager.load_corpus(sample_csv)
        with pytest.raises(ValueError, match="sum to 1.0"):
            manager.split_dataset(df, ratios=(0.7, 0.2, 0.2))
