"""Utility helpers shared across the training and inference pipeline."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_directory(path: Path | str) -> Path:
	"""Create parent directories for the provided path if needed."""

	path = Path(path)
	if path.is_dir():
		path.mkdir(parents=True, exist_ok=True)
		return path

	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def set_global_seed(seed: int) -> None:
	"""Seed Python, NumPy and hash behaviour for deterministic runs."""

	random.seed(seed)
	np.random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)


def configure_logging(level: int = logging.INFO) -> None:
	"""Initialise a root logger with a concise, uniform format."""

	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


def describe_scores(scores: Iterable[float]) -> str:
	"""Return a compact summary for a collection of scores."""

	values = list(scores)
	if not values:
		return "[]"

	mean_score = float(np.mean(values))
	std_score = float(np.std(values))
	return f"[{mean_score:.4f} ± {std_score:.4f}]"
