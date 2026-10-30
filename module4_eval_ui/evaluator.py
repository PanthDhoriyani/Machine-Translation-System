"""
Evaluator — Computes BLEU, METEOR, and chrF translation quality metrics.

FR-4.1: Compute BLEU score using sacrebleu
FR-4.2: Compute METEOR score using nltk
FR-4.3: Compute chrF score
"""

import logging
from typing import Optional

import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score

logger = logging.getLogger(__name__)

# Download NLTK data if not already present (required for METEOR)
def _ensure_nltk_data():
    """Download required NLTK corpora silently."""
    for pkg in ("wordnet", "omw-1.4", "punkt"):
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()


class Evaluator:
    """Computes translation quality metrics against reference translations.

    Attributes:
        hypotheses (list[str]): Model-generated translations.
        references (list[str]): Human reference translations.
    """

    def __init__(self, hypotheses: list[str], references: list[str]):
        """Initialise the Evaluator.

        Args:
            hypotheses (list[str]): System translation outputs.
            references (list[str]): Reference (gold) translations.

        Raises:
            ValueError: If hypotheses or references are empty, or lengths differ.
        """
        if not hypotheses or not references:
            raise ValueError("hypotheses and references must not be empty. (TC-4.9)")
        if len(hypotheses) != len(references):
            raise ValueError(
                f"Length mismatch: {len(hypotheses)} hypotheses vs {len(references)} references."
            )

        self.hypotheses: list[str] = hypotheses
        self.references: list[str] = references

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_bleu(self) -> float:
        """Compute corpus-level BLEU score using sacrebleu. (FR-4.1, TC-4.1, TC-4.2)

        Returns:
            BLEU score in range [0, 100]. Higher is better. Target ≥ 25 (NFR-8).
        """
        result = sacrebleu.corpus_bleu(self.hypotheses, [self.references])
        score = round(result.score, 4)
        logger.info("BLEU: %.4f", score)
        return score

    def compute_meteor(self) -> float:
        """Compute average sentence-level METEOR score. (FR-4.2, TC-4.3)

        METEOR considers synonyms, stemming, and paraphrase matching.

        Returns:
            METEOR score in range [0, 1]. Higher is better. Target ≥ 0.35.
        """
        scores = []
        for hyp, ref in zip(self.hypotheses, self.references):
            hyp_tokens = nltk.word_tokenize(hyp.lower())
            ref_tokens = nltk.word_tokenize(ref.lower())
            scores.append(meteor_score([ref_tokens], hyp_tokens))

        avg = round(sum(scores) / len(scores), 4) if scores else 0.0
        logger.info("METEOR: %.4f", avg)
        return avg

    def compute_chrf(self) -> float:
        """Compute corpus-level chrF score using sacrebleu. (FR-4.3, TC-4.4)

        Character n-gram F-score, robust to morphologically rich languages.

        Returns:
            chrF score in range [0, 100]. Higher is better. Target ≥ 45.
        """
        result = sacrebleu.corpus_chrf(self.hypotheses, [self.references])
        score = round(result.score, 4)
        logger.info("chrF: %.4f", score)
        return score

    def generate_report(self) -> dict:
        """Compute all three metrics and return them as a dict. (FR-4.6)

        Returns:
            dict with keys: bleu (float), meteor (float), chrf (float),
            num_sentences (int).
        """
        return {
            "bleu": self.compute_bleu(),
            "meteor": self.compute_meteor(),
            "chrf": self.compute_chrf(),
            "num_sentences": len(self.hypotheses),
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Evaluator(n={len(self.hypotheses)} sentence pairs)"
