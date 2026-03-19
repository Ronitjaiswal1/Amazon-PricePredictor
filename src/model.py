"""Model definition and feature pipeline for the pricing challenge."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightgbm import LGBMRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import TransformedTargetRegressor

from .data import TEXT_COLUMN


IPQ_REGEX = re.compile(r"(\d+[\d,.]*)\s*(?:pack|pcs|ct|count|pk|pc|piece|pieces|ipq)?", re.IGNORECASE)


def _extract_text_column(frame: pd.DataFrame) -> pd.Series:
	column = frame
	if isinstance(frame, pd.DataFrame):
		column = frame[TEXT_COLUMN]
	return column.fillna("").astype(str)


def _text_stats(frame: pd.DataFrame) -> np.ndarray:
	series = _extract_text_column(frame)
	char_len = series.str.len().to_numpy(dtype=np.float32)
	word_count = series.str.split().map(len).to_numpy(dtype=np.float32)
	digit_count = series.str.count(r"\d").to_numpy(dtype=np.float32)
	upper_count = series.str.count(r"[A-Z]").to_numpy(dtype=np.float32)

	ipq_match = series.str.extract(IPQ_REGEX, expand=False)
	ipq_numeric = ipq_match.str.replace(",", "", regex=False)
	ipq_values = pd.to_numeric(ipq_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

	features = np.vstack(
		[
			char_len,
			word_count,
			digit_count,
			upper_count,
			np.divide(char_len, np.maximum(word_count, 1.0)),
			ipq_values,
		]
	).T
	return features


def _stats_feature_names() -> list[str]:
	return [
		"char_len",
		"word_count",
		"digit_count",
		"upper_count",
		"avg_token_len",
		"ipq",
	]


class NamedFunctionTransformer(FunctionTransformer):
	"""FunctionTransformer that exposes stable feature names."""

	def __init__(self, func, feature_names_out: Optional[list[str]] = None, **kwargs):
		super().__init__(func=func, validate=False, **kwargs)
		self._feature_names_out = feature_names_out

	def get_feature_names_out(self, input_features=None):  # type: ignore[override]
		if self._feature_names_out is None:
			return super().get_feature_names_out(input_features)
		return np.asarray(self._feature_names_out)


@dataclass
class ModelConfig:
	"""Configuration bundle used to instantiate the model pipeline."""

	tfidf_max_features: int = 60000
	tfidf_ngram_range: Tuple[int, int] = (1, 2)
	tfidf_min_df: int = 3
	tfidf_max_df: float = 0.95
	svd_components: Optional[int] = 256
	random_state: int = 42
	cv_folds: int = 5
	min_prediction: float = 0.1
	lightgbm_params: Dict[str, Any] = field(
		default_factory=lambda: {
			"n_estimators": 2000,
			"learning_rate": 0.05,
			"num_leaves": 64,
			"max_depth": -1,
			"subsample": 0.9,
			"colsample_bytree": 0.7,
			"reg_alpha": 0.1,
			"reg_lambda": 0.1,
			"objective": "poisson",
			"n_jobs": -1,
		}
	)

	def __post_init__(self) -> None:
		self.lightgbm_params.setdefault("random_state", self.random_state)

	def as_dict(self) -> Dict[str, Any]:
		return asdict(self)


def _build_text_pipeline(config: ModelConfig) -> Pipeline:
	steps: list[tuple[str, object]] = [
		("extract", NamedFunctionTransformer(_extract_text_column)),
		(
			"tfidf",
			TfidfVectorizer(
				max_features=config.tfidf_max_features,
				ngram_range=config.tfidf_ngram_range,
				min_df=config.tfidf_min_df,
				max_df=config.tfidf_max_df,
				sublinear_tf=True,
				dtype=np.float32,
			),
		),
	]

	if config.svd_components:
		steps.append(
			(
				"svd",
				TruncatedSVD(
					n_components=config.svd_components,
					random_state=config.random_state,
				),
			)
		)

	return Pipeline(steps)


def _build_stat_pipeline() -> Pipeline:
	return Pipeline(
		steps=[
			("stats", NamedFunctionTransformer(_text_stats, feature_names_out=_stats_feature_names())),
			("impute", SimpleImputer(strategy="median")),
			("scale", StandardScaler()),
			("to_sparse", NamedFunctionTransformer(lambda x: sp.csr_matrix(x))),
		]
	)


def build_feature_pipeline(config: ModelConfig) -> FeatureUnion:
	return FeatureUnion(
		transformer_list=[
			("text", _build_text_pipeline(config)),
			("stats", _build_stat_pipeline()),
		]
	)


def build_model(config: Optional[ModelConfig] = None) -> Pipeline:
	config = config or ModelConfig()

	preprocess = build_feature_pipeline(config)
	regressor = LGBMRegressor(**config.lightgbm_params)
	target_transformer = TransformedTargetRegressor(
		regressor=regressor,
		func=lambda y: np.log1p(np.clip(y, a_min=config.min_prediction, a_max=None)),
		inverse_func=np.expm1,
	)

	pipeline = Pipeline(
		steps=[
			("features", preprocess),
			("regressor", target_transformer),
		]
	)
	pipeline.config = config  # type: ignore[attr-defined]
	return pipeline
