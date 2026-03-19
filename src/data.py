"""Data loading and preprocessing helpers for the pricing challenge."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

TEXT_COLUMN = "catalog_content"
PRICE_COLUMN = "price"
SAMPLE_ID_COLUMN = "sample_id"
IMAGE_LINK_COLUMN = "image_link"


def _resolve_path(path: str | Path) -> Path:
	resolved = Path(path)
	if not resolved.exists():
		raise FileNotFoundError(f"Dataset not found at {resolved}")
	return resolved


def load_training_data(path: str | Path) -> pd.DataFrame:
	"""Load the training CSV and enforce expected column types."""

	df = _load_common(path)
	if PRICE_COLUMN not in df.columns:
		missing = PRICE_COLUMN
		raise ValueError(f"Training data must contain '{missing}' column")

	df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors="coerce")
	df = df.dropna(subset=[PRICE_COLUMN])
	return df


def load_test_data(path: str | Path) -> pd.DataFrame:
	"""Load the test CSV file."""

	return _load_common(path)


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
	"""Return the subset of columns used by the model."""

	expected: Iterable[str] = [TEXT_COLUMN]
	missing = [col for col in expected if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	features = df[list(expected)].copy()
	features[TEXT_COLUMN] = features[TEXT_COLUMN].fillna("")
	return features


def _load_common(path: str | Path) -> pd.DataFrame:
	resolved = _resolve_path(path)
	df = pd.read_csv(resolved)

	if SAMPLE_ID_COLUMN in df.columns:
		df[SAMPLE_ID_COLUMN] = df[SAMPLE_ID_COLUMN].astype(str)
	if TEXT_COLUMN in df.columns:
		df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")

	return df
