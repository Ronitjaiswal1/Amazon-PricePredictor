"""Prepare local numpy arrays from precomputed text embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs = project_root / "outputs"
    features_dir = project_root / "features"
    dataset_dir = project_root / "dataset"

    train_embeddings = outputs / "train_text_embeddings.npy"
    test_embeddings = outputs / "test_text_embeddings.npy"
    train_csv = dataset_dir / "train.csv"

    if not train_embeddings.exists():
        raise FileNotFoundError(train_embeddings)
    if not test_embeddings.exists():
        raise FileNotFoundError(test_embeddings)
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)

    features_dir.mkdir(exist_ok=True)

    train_text = np.load(train_embeddings)
    test_text = np.load(test_embeddings)
    np.save(features_dir / "train_text.npy", train_text)
    np.save(features_dir / "test_text.npy", test_text)

    price = pd.read_csv(train_csv)["price"].to_numpy()
    np.save(features_dir / "train_price.npy", price)
    np.save(features_dir / "train_log_price.npy", np.log1p(price))

    print("Prepared features:")
    print("train_text:", train_text.shape)
    print("test_text:", test_text.shape)


if __name__ == "__main__":
    main()
