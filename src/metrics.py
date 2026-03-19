"""Custom evaluation metrics used in the challenge."""

from __future__ import annotations

import numpy as np


def smape(y_true, y_pred, epsilon: float = 1e-9) -> float:
	"""Compute Symmetric Mean Absolute Percentage Error (SMAPE)."""

	y_true = np.asarray(y_true, dtype=np.float64)
	y_pred = np.asarray(y_pred, dtype=np.float64)

	denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
	safe_denominator = np.where(denominator < epsilon, epsilon, denominator)
	diff = np.abs(y_true - y_pred) / safe_denominator
	return float(np.mean(diff) * 100.0)
