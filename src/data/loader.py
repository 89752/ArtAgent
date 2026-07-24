"""
SemArt dataset loader and preprocessing utilities.

Fields: TITLE, DESCRIPTION, AUTHOR, DATE, TECHNIQUE, TYPE, SCHOOL, TIMEFRAME, FILE
"""

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATA_DIR = Path(os.getenv("SEMART_DATA_DIR", "./SemArt"))


class SemArtDataset:
    """
    Loads and provides access to SemArt CSV data.

    Usage:
        dataset = SemArtDataset()
        df = dataset.train
        row = dataset.get_by_title("The Kiss")
    """

    SPLITS = ["train", "val", "test"]

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self._dfs: dict[str, pd.DataFrame] = {}
        self._load_all()

    def _load_all(self):
        for split in self.SPLITS:
            csv_path = self.data_dir / f"semart_{split}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"SemArt CSV not found: {csv_path}")

            df = pd.read_csv(
                csv_path,
                sep="\t",
                encoding="latin-1",
                on_bad_lines="skip",
            )
            df.columns = [c.upper().strip() for c in df.columns]
            df = df.dropna(
                subset=["TITLE", "AUTHOR", "IMAGE_FILE"]
            )  # ← FILE → IMAGE_FILE
            for col in [
                "DESCRIPTION",
                "TECHNIQUE",
                "TYPE",
                "SCHOOL",
                "TIMEFRAME",
                "DATE",
            ]:
                if col in df.columns:
                    df[col] = df[col].fillna("")
            self._dfs[split] = df.reset_index(drop=True)

    @property
    def train(self) -> pd.DataFrame:
        return self._dfs["train"]

    @property
    def val(self) -> pd.DataFrame:
        return self._dfs["val"]

    @property
    def test(self) -> pd.DataFrame:
        return self._dfs["test"]

    @property
    def all(self) -> pd.DataFrame:
        return pd.concat(self._dfs.values(), ignore_index=True)

    def get_by_title(self, title: str, exact: bool = False) -> Optional[pd.Series]:
        df = self.all
        if exact:
            mask = df["TITLE"].str.lower() == title.lower()
        else:
            mask = (
                df["TITLE"]
                .str.lower()
                .str.contains(title.lower(), na=False, regex=False)
            )
        result = df[mask]
        return None if result.empty else result.iloc[0]

    def get_by_author(self, author: str) -> pd.DataFrame:
        df = self.all
        mask = (
            df["AUTHOR"].str.lower().str.contains(author.lower(), na=False, regex=False)
        )
        return df[mask].reset_index(drop=True)

    def get_image_path(self, file_field: str) -> Path:
        return self.data_dir / "Images" / file_field

    def get_stats(self) -> dict:
        return {
            "train_size": len(self.train),
            "val_size": len(self.val),
            "test_size": len(self.test),
            "total": len(self.all),
            "unique_authors": self.all["AUTHOR"].nunique(),
            "unique_schools": self.all["SCHOOL"].nunique(),
            "timeframes": sorted(self.all["TIMEFRAME"].unique().tolist()),
        }


@lru_cache(maxsize=1)
def get_dataset() -> SemArtDataset:
    """Return a globally cached SemArtDataset instance."""
    return SemArtDataset()


def build_document_corpus(dataset: Optional[SemArtDataset] = None) -> list[dict]:
    if dataset is None:
        dataset = get_dataset()

    documents = []
    for _, row in dataset.all.iterrows():
        text_parts = [
            f"Title: {row['TITLE']}",
            f"Artist: {row['AUTHOR']}",
        ]
        for field, label in [
            ("DATE", "Date"),
            ("TECHNIQUE", "Technique"),
            ("TYPE", "Type"),
            ("SCHOOL", "School"),
            ("TIMEFRAME", "Timeframe"),
            ("DESCRIPTION", "Description"),
        ]:
            if row.get(field):
                text_parts.append(f"{label}: {row[field]}")

        documents.append(
            {
                "text": "\n".join(text_parts),
                "metadata": {
                    "title": row["TITLE"],
                    "author": row["AUTHOR"],
                    "date": str(row.get("DATE", "")),
                    "technique": row.get("TECHNIQUE", ""),
                    "type": row.get("TYPE", ""),
                    "school": row.get("SCHOOL", ""),
                    "timeframe": row.get("TIMEFRAME", ""),
                    "file": row["IMAGE_FILE"],  # ← FILE → IMAGE_FILE
                    "description": row.get("DESCRIPTION", ""),
                },
            }
        )

    return documents
