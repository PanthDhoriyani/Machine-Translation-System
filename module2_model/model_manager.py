"""
ModelManager — Loads, caches, and switches HuggingFace MT models.

FR-2.1: Load models from Hugging Face Hub by model ID
FR-2.2: Support MarianMT, mBART-50, M2M-100
FR-2.3: Cache downloaded models locally
FR-2.4: Support dynamic language pair switching
FR-2.5: Expose GET /model/info returning model metadata
"""

import time
import logging
from typing import Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported model catalogue (FR-2.2)
# ---------------------------------------------------------------------------
SUPPORTED_MODELS: dict[str, dict] = {
    # MarianMT — fast, pair-specific
    "Helsinki-NLP/opus-mt-en-fr": {"src_lang": "en", "tgt_lang": "fr", "type": "MarianMT"},
    "Helsinki-NLP/opus-mt-en-de": {"src_lang": "en", "tgt_lang": "de", "type": "MarianMT"},
    "Helsinki-NLP/opus-mt-en-es": {"src_lang": "en", "tgt_lang": "es", "type": "MarianMT"},
    "Helsinki-NLP/opus-mt-en-hi": {"src_lang": "en", "tgt_lang": "hi", "type": "MarianMT"},
    # mBART-50 — multilingual
    "facebook/mbart-large-50-many-to-many-mmt": {
        "src_lang": None,
        "tgt_lang": None,
        "type": "mBART-50",
    },
    # M2M-100 — multilingual
    "facebook/m2m100_418M": {"src_lang": None, "tgt_lang": None, "type": "M2M-100"},
}


class ModelManager:
    """Manages lifecycle of HuggingFace MT models.

    Attributes:
        model_id (str): Currently loaded model's HuggingFace identifier.
        model (PreTrainedModel): The loaded model instance.
        tokenizer: The loaded tokenizer instance.
    """

    # In-process cache shared across all instances (FR-2.3)
    _cache: dict[str, tuple] = {}

    def __init__(self, model_id: str = "Helsinki-NLP/opus-mt-en-fr"):
        """Initialise with a default model.

        Args:
            model_id (str): HuggingFace model ID to load on startup.
        """
        self.model_id: str = model_id
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer = None
        self._meta: dict = {}

        self.load_model(model_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, model_id: str) -> None:
        """Load a model by its HuggingFace Hub ID with local caching.

        Args:
            model_id (str): HuggingFace model identifier.

        Raises:
            OSError: If the model ID is invalid or cannot be downloaded. (TC-2.2)
        """
        if model_id in ModelManager._cache:
            logger.info("Loading '%s' from in-process cache.", model_id)
            self.model, self.tokenizer = ModelManager._cache[model_id]
            self.model_id = model_id
            self._refresh_meta()
            return

        logger.info("Downloading / loading '%s' from HuggingFace Hub …", model_id)
        t0 = time.perf_counter()

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        except (OSError, EnvironmentError) as exc:
            raise OSError(f"Failed to load model '{model_id}': {exc}") from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Loaded '%s' in %.0f ms.", model_id, elapsed_ms)

        # Store in cache (FR-2.3)
        ModelManager._cache[model_id] = (model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self._refresh_meta()

    def switch_model(self, new_model_id: str) -> None:
        """Dynamically switch to a different model / language pair. (FR-2.4)

        Args:
            new_model_id (str): HuggingFace ID of the target model.
        """
        logger.info("Switching model: '%s' → '%s'", self.model_id, new_model_id)
        self.load_model(new_model_id)

    def get_model_info(self) -> dict:
        """Return metadata about the currently loaded model. (FR-2.5)

        Returns:
            dict with model_id, src_lang, tgt_lang, parameters_M, cached keys.
        """
        return {
            "model_id": self.model_id,
            "src_lang": self._meta.get("src_lang"),
            "tgt_lang": self._meta.get("tgt_lang"),
            "type": self._meta.get("type", "unknown"),
            "parameters_M": self._count_params(),
            "cached": self.model_id in ModelManager._cache,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _refresh_meta(self) -> None:
        """Refresh internal metadata from the catalogue."""
        self._meta = SUPPORTED_MODELS.get(self.model_id, {})

    def _count_params(self) -> Optional[float]:
        """Return model parameter count in millions, or None if unavailable."""
        if self.model is None:
            return None
        total = sum(p.numel() for p in self.model.parameters())
        return round(total / 1_000_000, 1)

    def __repr__(self) -> str:
        return f"ModelManager(model_id='{self.model_id}', cached={len(ModelManager._cache)})"
