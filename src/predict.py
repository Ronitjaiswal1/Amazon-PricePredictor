"""Inference script to generate predictions for the test dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import SAMPLE_ID_COLUMN, get_feature_frame, load_test_data
from .utils import configure_logging, ensure_directory


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate predictions for the test set")
	parser.add_argument("--test-path", type=str, default="dataset/test.csv")
	parser.add_argument("--model-path", type=str, default="models/pricing_model.joblib")
	parser.add_argument("--output-path", type=str, default="models/test_out.csv")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	configure_logging()
	logger = logging.getLogger("predict")

	logger.info("Loading model from %s", args.model_path)
	artefacts = joblib.load(args.model_path)
	pipeline = artefacts["pipeline"]
	config_dict = artefacts.get("config", {})
	min_prediction = float(config_dict.get("min_prediction", 0.0))

	logger.info("Loading test data from %s", args.test_path)
	test_df = load_test_data(args.test_path)
	features = get_feature_frame(test_df)

	logger.info("Running inference on %d samples", len(features))
	predictions = pipeline.predict(features)
	predictions = np.clip(predictions, min_prediction, None)

	output = pd.DataFrame({
		SAMPLE_ID_COLUMN: test_df[SAMPLE_ID_COLUMN],
		"price": predictions.astype(float),
	})

	output_path = ensure_directory(args.output_path)
	output.to_csv(output_path, index=False)
	logger.info("Saved predictions to %s", output_path)


if __name__ == "__main__":
	main()
