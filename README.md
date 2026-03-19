# Amazon Price Predictor

Machine learning pipeline for predicting product prices from catalog text and metadata.

## Highlights

- LightGBM-based regression pipeline for price prediction.
- Training and inference entry points under `src/`.
- Utility scripts for text features and image embeddings.
- Notebook workflows for experimentation and model development.

## Tech Stack

- Python 3.10+
- NumPy, Pandas, Scikit-learn, SciPy
- LightGBM
- PyTorch / TorchVision (for embedding workflows)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Training

```powershell
python -m src.train --train-path dataset/train.csv --model-path models/pricing_model.joblib
```

This trains the model and stores artifacts/metrics near the model output path.

## Prediction

```powershell
python -m src.predict --test-path dataset/test.csv --model-path models/pricing_model.joblib --output-path models/test_out.csv
```

Expected output format: `sample_id`, `price`.

## Project Structure

- `src/data.py` - data loading and preprocessing helpers.
- `src/model.py` - feature engineering and model pipeline definition.
- `src/train.py` - training entry point.
- `src/predict.py` - inference entry point.
- `src/metrics.py` - evaluation metrics.
- `src/utils.py` - common utilities.
- `scripts/` - supporting feature generation scripts.
- `notebooks/` - experimentation notebooks.

## Notes on Version Control

Large and generated assets are intentionally ignored in `.gitignore`, including:

- Raw/derived datasets
- Generated embeddings and feature arrays
- Trained model binaries
- Local virtual environments and caches

This keeps the repository clean, lightweight, and easier to clone.
