"""
Preprocessor — Text normalization, tokenization, and detokenization.

FR-1.3: Normalize text (lowercasing, unicode, punctuation)
FR-1.4: Tokenize using the model's native tokenizer (max 512 tokens)
"""

import re
import unicodedata
import logging
from typing import Union

logger = logging.getLogger(__name__)

# Maximum token length accepted by most transformer models
MAX_TOKENS = 512


class Preprocessor:
    """Handles all text preprocessing for machine translation.

    Wraps a HuggingFace tokenizer to provide normalization, tokenization,
    and detokenization utilities aligned to the SRS requirements.

    Attributes:
        tokenizer: HuggingFace tokenizer instance.
        max_length (int): Maximum number of tokens per input chunk.
    """

    def __init__(self, tokenizer, max_length: int = MAX_TOKENS):
        """Initialise the Preprocessor.

        Args:
            tokenizer: Any HuggingFace AutoTokenizer-compatible object.
            max_length (int): Token cap per input (FR-1.4, TC-1.5).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        """Apply text normalization. (FR-1.3, TC-1.3, TC-1.8)

        Steps applied in order:
          1. Strip HTML tags
          2. Normalize unicode (NFC)
          3. Collapse multiple whitespace
          4. Strip leading/trailing whitespace

        Note: Lowercasing is intentionally skipped to preserve named entities
        and proper nouns that matter for MT quality.

        Args:
            text (str): Raw input text.

        Returns:
            Normalized text string.
        """
        # 1. Strip HTML / script injection (TC-1.8)
        text = re.sub(r"<[^>]+>", "", text)

        # 2. Unicode NFC normalization — e.g. "caf\u00e9" → "café" (TC-1.3)
        text = unicodedata.normalize("NFC", text)

        # 3. Collapse multiple spaces / tabs / newlines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def tokenize(self, text: str) -> dict:
        """Tokenize text using the model-native tokenizer. (FR-1.4, TC-1.5)

        Tokens are truncated to self.max_length. Use chunk_text() for long
        inputs that should be processed without truncation.

        Args:
            text (str): Normalized input text.

        Returns:
            HuggingFace tokenizer output dict (input_ids, attention_mask).
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        token_count = encoded["input_ids"].shape[-1]
        logger.debug("Tokenized to %d tokens (max %d).", token_count, self.max_length)
        return encoded

    def detokenize(self, token_ids) -> str:
        """Convert model output token IDs back to a clean string. (FR-3.6)

        Args:
            token_ids: Tensor or list of generated token IDs.

        Returns:
            Decoded, cleaned translation string.
        """
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return text.strip()

    def chunk_text(self, text: str) -> list[str]:
        """Split long text into chunks that fit within max_length tokens. (TC-3.6)

        Splits on sentence boundaries where possible to preserve semantic units.

        Args:
            text (str): Potentially long input text.

        Returns:
            List of text chunk strings.
        """
        # Quick path — already short enough
        test_enc = self.tokenizer(text, truncation=False)
        if len(test_enc["input_ids"]) <= self.max_length:
            return [text]

        # Split by sentences and re-group into chunks
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(self.tokenizer(sent, truncation=False)["input_ids"])
            if current_len + sent_len > self.max_length and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(sent)
            current_len += sent_len

        if current:
            chunks.append(" ".join(current))

        logger.debug("Text split into %d chunk(s) for long-input handling.", len(chunks))
        return chunks

    def __repr__(self) -> str:
        return f"Preprocessor(max_length={self.max_length})"
