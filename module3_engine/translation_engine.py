"""
TranslationEngine — Core inference engine for machine translation.

FR-3.1: Translate a single sentence within 3 seconds (CPU)
FR-3.2: Support batch translation of up to 1000 sentences
FR-3.3: Support beam search (default=4), greedy, top-k decoding
FR-3.5: Return errors for unsupported language pairs
FR-3.6: Detokenize and clean output before returning
"""

import time
import logging
from typing import Literal, Optional

import torch

from module1_preprocessing.preprocessor import Preprocessor
from module2_model.model_manager import ModelManager, SUPPORTED_MODELS

logger = logging.getLogger(__name__)

# Language pair → model mapping for auto-selection
LANG_PAIR_MODEL_MAP: dict[tuple[str, str], str] = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    # mBART / M2M can handle many pairs
    ("en", "zh"): "facebook/m2m100_418M",
    ("en", "ar"): "facebook/m2m100_418M",
    ("en", "ru"): "facebook/m2m100_418M",
}

# Pairs supported by M2M-100 (used as fallback)
M2M_MODEL_ID = "facebook/m2m100_418M"


class TranslationEngine:
    """Core translation engine wrapping a ModelManager and Preprocessor.

    Supports single-sentence, chunked (long input), and batch translation
    with configurable decoding strategies.

    Attributes:
        beam_size (int): Number of beams for beam search (default 4).
        max_new_tokens (int): Maximum tokens in generated output.
        model_manager (ModelManager): Active model manager instance.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        beam_size: int = 4,
        max_new_tokens: int = 512,
    ):
        """Initialise the TranslationEngine.

        Args:
            model_manager (ModelManager): Pre-configured model manager.
            beam_size (int): Beam search width. Set to 1 for greedy decoding.
            max_new_tokens (int): Maximum output token length.
        """
        self.model_manager = model_manager
        self.beam_size = beam_size
        self.max_new_tokens = max_new_tokens
        self._preprocessor: Optional[Preprocessor] = None
        self._refresh_preprocessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        decoding: Literal["beam", "greedy", "top_k"] = "beam",
        beam_size: Optional[int] = None,
    ) -> dict:
        """Translate a single input string. (FR-3.1, TC-3.1, TC-3.5)

        Automatically chunks inputs longer than 512 tokens. (TC-3.6)

        Args:
            text (str): Source text to translate.
            src_lang (str): ISO 639-1 source language code.
            tgt_lang (str): ISO 639-1 target language code.
            decoding (str): Decoding strategy — 'beam', 'greedy', 'top_k'. (FR-3.3)
            beam_size (Optional[int]): Override engine-level beam size.

        Returns:
            dict with keys: translation (str), src_lang, tgt_lang,
            model_used (str), latency_ms (float).

        Raises:
            ValueError: If the language pair is not supported. (FR-3.5)
        """
        # Ensure the right model is loaded for this language pair
        self._ensure_model_for_pair(src_lang, tgt_lang)

        preprocessor = self._preprocessor
        t0 = time.perf_counter()

        # Normalize input
        normalized = preprocessor.normalize(text)

        # Handle long inputs via chunking (TC-3.6)
        chunks = preprocessor.chunk_text(normalized)
        translated_chunks = []

        bs = beam_size if beam_size is not None else self.beam_size

        for chunk in chunks:
            tokens = preprocessor.tokenize(chunk)
            output_ids = self._run_inference(tokens, decoding=decoding, beam_size=bs)
            translated_chunks.append(preprocessor.detokenize(output_ids[0]))

        translation = " ".join(translated_chunks)
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Translated %d chars (%d chunk(s)) in %.0f ms via %s.",
            len(text), len(chunks), latency_ms, self.model_manager.model_id,
        )

        return {
            "translation": translation,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "model_used": self.model_manager.model_id,
            "latency_ms": round(latency_ms, 2),
        }

    def batch_translate(
        self,
        sentences: list[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 32,
    ) -> list[str]:
        """Translate a list of sentences in mini-batches. (FR-3.2, TC-3.4)

        Args:
            sentences (list[str]): Up to 1000 source sentences.
            src_lang (str): Source language code.
            tgt_lang (str): Target language code.
            batch_size (int): Number of sentences per inference call.

        Returns:
            List of translated strings, same order as input.

        Raises:
            ValueError: If sentences list is empty or language pair unsupported.
        """
        if not sentences:
            raise ValueError("sentences list must not be empty.")
        if len(sentences) > 1000:
            raise ValueError("Batch size exceeds maximum of 1000 sentences (FR-3.2).")

        self._ensure_model_for_pair(src_lang, tgt_lang)
        preprocessor = self._preprocessor
        results: list[str] = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            normalized = [preprocessor.normalize(s) for s in batch]

            encoded = self.model_manager.tokenizer(
                normalized,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            output_ids = self._run_inference(encoded, decoding="beam", beam_size=self.beam_size)
            decoded = self.model_manager.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            results.extend(decoded)

            logger.debug("Batch %d-%d translated.", i, i + len(batch))

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model_for_pair(self, src_lang: str, tgt_lang: str) -> None:
        """Load the best model for the given language pair. (FR-3.5)

        Raises:
            ValueError: If no model supports the requested pair.
        """
        pair = (src_lang.lower(), tgt_lang.lower())

        # Already loaded for this pair?
        info = self.model_manager.get_model_info()
        if info["src_lang"] == src_lang and info["tgt_lang"] == tgt_lang:
            return

        target_model = LANG_PAIR_MODEL_MAP.get(pair)

        if target_model is None:
            # Try M2M as generic fallback
            target_model = M2M_MODEL_ID
            logger.warning(
                "No dedicated model for %s→%s. Falling back to M2M-100.", src_lang, tgt_lang
            )

        if target_model != self.model_manager.model_id:
            self.model_manager.switch_model(target_model)
            self._refresh_preprocessor()

    def _run_inference(self, encoded: dict, decoding: str, beam_size: int):
        """Run model.generate() with the selected decoding strategy. (FR-3.3)

        Args:
            encoded: Tokenizer output dict.
            decoding: One of 'beam', 'greedy', 'top_k'.
            beam_size: Beam width for beam search.

        Returns:
            Generated token IDs tensor.
        """
        model = self.model_manager.model
        model.eval()

        generate_kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
        }

        if decoding == "beam":
            generate_kwargs["num_beams"] = beam_size
            generate_kwargs["early_stopping"] = True
        elif decoding == "greedy":
            generate_kwargs["num_beams"] = 1
        elif decoding == "top_k":
            generate_kwargs["do_sample"] = True
            generate_kwargs["top_k"] = 50

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                **generate_kwargs,
            )

        return output_ids

    def _refresh_preprocessor(self) -> None:
        """Rebuild preprocessor when the tokenizer changes."""
        self._preprocessor = Preprocessor(self.model_manager.tokenizer)

    def __repr__(self) -> str:
        return (
            f"TranslationEngine(model='{self.model_manager.model_id}', "
            f"beam_size={self.beam_size})"
        )
