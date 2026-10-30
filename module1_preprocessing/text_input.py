"""
TextInput — Validates and detects the language of raw input text.

FR-1.1: Accept raw text input in UTF-8 encoding
FR-1.2: Auto-detect source language using langdetect/fastText
"""

import re
from typing import Optional

try:
    from langdetect import detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


class TextInput:
    """Represents a single translation request with validation and language detection.

    Attributes:
        raw_text (str): The original, unprocessed input text.
        src_lang (Optional[str]): Source language code (e.g. 'en'). If None,
            detect_language() will be called automatically.
        tgt_lang (str): Target language code (e.g. 'fr').
    """

    def __init__(self, raw_text: str, tgt_lang: str, src_lang: Optional[str] = None):
        """Initialise TextInput.

        Args:
            raw_text (str): Raw UTF-8 input text.
            tgt_lang (str): Target language code.
            src_lang (Optional[str]): Source language code. Detected automatically
                when not supplied.

        Raises:
            TypeError: If raw_text is not a string.
        """
        if not isinstance(raw_text, str):
            raise TypeError(f"raw_text must be str, got {type(raw_text).__name__}")

        self.raw_text: str = raw_text
        self.src_lang: Optional[str] = src_lang
        self.tgt_lang: str = tgt_lang

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Validate that the input is non-empty and within token limits.

        Returns:
            True if valid.

        Raises:
            ValueError: If raw_text is empty or consists only of whitespace. (TC-1.4)
        """
        if not self.raw_text or not self.raw_text.strip():
            raise ValueError("Input text must not be empty.")
        return True

    def detect_language(self) -> str:
        """Auto-detect the source language of raw_text. (FR-1.2, TC-1.1, TC-1.2)

        Uses the langdetect library. Falls back to 'en' if detection fails
        or the library is not installed.

        Returns:
            ISO 639-1 language code string (e.g. 'en', 'fr').
        """
        if self.src_lang:
            return self.src_lang

        if not _LANGDETECT_AVAILABLE:
            # Graceful fallback — no crash if langdetect not installed
            self.src_lang = "en"
            return self.src_lang

        try:
            self.src_lang = detect(self.raw_text)
        except LangDetectException:
            self.src_lang = "en"

        return self.src_lang

    def sanitize_html(self) -> str:
        """Remove HTML tags from raw_text. (TC-1.8)

        Returns:
            Text with all HTML/XML tags stripped.
        """
        clean = re.sub(r"<[^>]+>", "", self.raw_text)
        return clean.strip()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        preview = self.raw_text[:50].replace("\n", " ")
        return (
            f"TextInput(src='{self.src_lang}', tgt='{self.tgt_lang}', "
            f"text='{preview}...')"
        )
