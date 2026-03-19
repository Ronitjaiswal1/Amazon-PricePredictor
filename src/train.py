"""Training script for the Smart Product Pricing challenge."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from .data import get_feature_frame, load_training_data, PRICE_COLUMN
from .metrics import smape
from .model import ModelConfig, build_model
from .utils import configure_logging, describe_scores, ensure_directory, set_global_seed


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train the pricing model")
	parser.add_argument("--train-path", type=str, default="dataset/train.csv")
	parser.add_argument("--model-path", type=str, default="models/pricing_model.joblib")
	parser.add_argument("--metrics-path", type=str, default="models/training_metrics.json")
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--cv-folds", type=int, default=5)
	parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	configure_logging()
	logger = logging.getLogger("train")

	set_global_seed(args.random_state)

	logger.info("Loading training data from %s", args.train_path)
	train_df = load_training_data(args.train_path)
	target = train_df[PRICE_COLUMN].to_numpy(dtype=np.float32)
	features = get_feature_frame(train_df)

	config = ModelConfig(random_state=args.random_state, cv_folds=args.cv_folds)
	model = build_model(config)

	cv_scores: list[float] = []
	if not args.no_cv and config.cv_folds > 1:
		logger.info("Running %d-fold cross-validation", config.cv_folds)
		splitter = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

		for fold, (train_idx, valid_idx) in enumerate(splitter.split(features, target), start=1):
			cv_model = clone(model)
			cv_model.fit(features.iloc[train_idx], target[train_idx])
			preds = cv_model.predict(features.iloc[valid_idx])
			preds = np.clip(preds, config.min_prediction, None)
			score = smape(target[valid_idx], preds)
			cv_scores.append(score)
			logger.info("Fold %d SMAPE: %.4f", fold, score)

		logger.info("CV SMAPE %s", describe_scores(cv_scores))
	else:
		logger.info("Cross-validation disabled. Training on full dataset only.")

	logger.info("Fitting final model on full dataset")
	model.fit(features, target)

	model_path = ensure_directory(args.model_path)
	joblib.dump({"pipeline": model, "config": config.as_dict()}, model_path)
	logger.info("Saved model artefacts to %s", model_path)

	metrics_payload = {
		"cv_scores": cv_scores,
		"cv_mean_smape": float(np.mean(cv_scores)) if cv_scores else None,
		"cv_std_smape": float(np.std(cv_scores)) if cv_scores else None,
		"train_size": int(features.shape[0]),
	}

	metrics_path = ensure_directory(args.metrics_path)
	with open(metrics_path, "w", encoding="utf-8") as handle:
		json.dump(metrics_payload, handle, indent=2)
	logger.info("Stored training metrics at %s", metrics_path)


if __name__ == "__main__":
	main()
