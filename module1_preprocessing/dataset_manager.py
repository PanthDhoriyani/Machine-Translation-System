"""
DatasetManager — Load parallel corpora and split into train/val/test sets.

FR-1.5: Load parallel corpora from CSV/JSON/TMX formats
FR-1.6: Split datasets 80/10/10 (train/val/test)
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetManager:
    """Handles loading and splitting of bilingual parallel corpora.

    Supported formats: CSV, JSON, TMX.

    Attributes:
        train (pd.DataFrame): Training split (80 %).
        val (pd.DataFrame):   Validation split (10 %).
        test (pd.DataFrame):  Test split (10 %).
    """

    def __init__(self):
        self.train: Optional[pd.DataFrame] = None
        self.val: Optional[pd.DataFrame] = None
        self.test: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_corpus(self, path: str) -> pd.DataFrame:
        """Load a bilingual parallel corpus from file. (FR-1.5, TC-1.6)

        Supported extensions: .csv, .json, .tmx

        The resulting DataFrame always has columns:
          - ``src``: source language sentence
          - ``tgt``: target language sentence

        Args:
            path (str): Absolute or relative path to the corpus file.

        Returns:
            pd.DataFrame with 'src' and 'tgt' columns.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        ext = file_path.suffix.lower()

        if ext == ".csv":
            df = self._load_csv(file_path)
        elif ext == ".json":
            df = self._load_json(file_path)
        elif ext == ".tmx":
            df = self._load_tmx(file_path)
        else:
            raise ValueError(f"Unsupported corpus format: '{ext}'. Use CSV, JSON, or TMX.")

        if "src" not in df.columns or "tgt" not in df.columns:
            raise ValueError("Loaded DataFrame must contain 'src' and 'tgt' columns.")

        logger.info("Loaded %d sentence pairs from '%s'.", len(df), path)
        return df

    def split_dataset(
        self,
        df: pd.DataFrame,
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a corpus DataFrame into train/val/test splits. (FR-1.6, TC-1.7)

        Args:
            df (pd.DataFrame): Full parallel corpus.
            ratios (tuple): Three floats that sum to 1.0 — (train, val, test).
            seed (int): Random seed for reproducibility.

        Returns:
            Tuple of (train_df, val_df, test_df).

        Raises:
            ValueError: If ratios do not sum to 1.0.
        """
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.4f}")

        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        self.train = df.iloc[:n_train].reset_index(drop=True)
        self.val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
        self.test = df.iloc[n_train + n_val :].reset_index(drop=True)

        logger.info(
            "Split %d pairs → train=%d, val=%d, test=%d",
            n, len(self.train), len(self.val), len(self.test),
        )
        return self.train, self.val, self.test

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load a CSV with 'src' and 'tgt' columns."""
        df = pd.read_csv(path, encoding="utf-8")
        # Flexible column mapping
        df = self._remap_columns(df)
        return df[["src", "tgt"]].dropna()

    def _load_json(self, path: Path) -> pd.DataFrame:
        """Load a JSON file (list of {src, tgt} objects or a dict of lists)."""
        df = pd.read_json(path, encoding="utf-8")
        df = self._remap_columns(df)
        return df[["src", "tgt"]].dropna()

    def _load_tmx(self, path: Path) -> pd.DataFrame:
        """Parse a TMX (Translation Memory eXchange) file."""
        tree = ET.parse(path)
        root = tree.getroot()

        records = []
        for tu in root.iter("tu"):
            tuvs = list(tu.iter("tuv"))
            if len(tuvs) >= 2:
                src_seg = tuvs[0].find("seg")
                tgt_seg = tuvs[1].find("seg")
                if src_seg is not None and tgt_seg is not None:
                    records.append(
                        {"src": src_seg.text or "", "tgt": tgt_seg.text or ""}
                    )

        return pd.DataFrame(records)

    @staticmethod
    def _remap_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to remap common column name variants to 'src' / 'tgt'."""
        remap = {}
        for col in df.columns:
            lc = col.lower()
            if lc in ("source", "en", "src_text") and "src" not in remap.values():
                remap[col] = "src"
            elif lc in ("target", "fr", "de", "es", "hi", "tgt_text") and "tgt" not in remap.values():
                remap[col] = "tgt"
        return df.rename(columns=remap)

    def __repr__(self) -> str:
        sizes = {
            "train": len(self.train) if self.train is not None else 0,
            "val": len(self.val) if self.val is not None else 0,
            "test": len(self.test) if self.test is not None else 0,
        }
        return f"DatasetManager(splits={sizes})"
