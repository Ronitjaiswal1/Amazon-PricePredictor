# Amazon Price Predictor

End-to-end machine learning project for predicting product prices from catalog metadata, text fields, and optional derived features.

## Overview

This repository provides:

- A training pipeline for tabular/text-based regression.
- A prediction pipeline that writes submission-ready output.
- Supporting scripts for feature preparation and image/text embedding workflows.
- Notebooks for experimentation and iterative model development.

## Tech Stack

- Python 3.10+
- NumPy, Pandas, SciPy
- Scikit-learn
- LightGBM
- PyTorch and TorchVision (for embedding generation workflows)

## Quick Start

### 1. Clone and enter project

```powershell
git clone https://github.com/Ronitjaiswal1/Amazon-PricePredictor.git
cd Amazon-PricePredictor
```

### 2. Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Add dataset locally

Place challenge CSV files in the local dataset folder (ignored by git):

- dataset/train.csv
- dataset/test.csv

## Training

```powershell
python -m src.train --train-path dataset/train.csv --model-path models/pricing_model.joblib
```

This command trains the model and saves artifacts to the selected model path.

## Inference

```powershell
python -m src.predict --test-path dataset/test.csv --model-path models/pricing_model.joblib --output-path models/test_out.csv
```

Expected output columns:

- sample_id
- price

## Repository Layout

- src/ - training, prediction, metrics, and utilities
- scripts/ - one-off or batch feature generation scripts
- notebooks/ - experimentation and pipeline notebooks
- dataset/ - local input files (ignored)
- models/ - saved model artifacts (ignored)
- outputs/ - generated embeddings/predictions (ignored)
- features/ - cached feature arrays (ignored)

## Version Control Notes

The repository ignores large, generated, and environment-specific files in .gitignore, including:

- dataset and data directories
- generated .npy and model artifacts
- virtual environments and cache folders
- local notebook metadata

This keeps commits lightweight and source-focused.

## Future Improvements

- Hyperparameter search for LightGBM using Optuna or Bayesian optimization.
- Better feature fusion between text and image embeddings.
- Experiment tracking integration with MLflow or Weights and Biases.
