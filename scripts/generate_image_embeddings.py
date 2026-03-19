"""Generate image embeddings using a ViT backbone.

Usage:
    python scripts/generate_image_embeddings.py --split train --images-dir data/images
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

LOGGER = logging.getLogger("image_embeddings")


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], image_size: int = 224) -> None:
        self.image_paths = image_paths
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.transform = weights.transforms(crop_size=image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        if path.exists():
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
            except Exception:  # noqa: BLE001
                img = Image.new("RGB", (224, 224))
                valid = False
            else:
                valid = True
        else:
            img = Image.new("RGB", (224, 224))
            valid = False

        tensor = self.transform(img)
        return tensor, int(valid)


def load_model(device: torch.device) -> nn.Module:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads = nn.Identity()
    model.eval()
    model.to(device)
    return model


def infer_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    feature_dim: int,
) -> np.ndarray:
    features: List[np.ndarray] = []
    missing = 0

    with torch.no_grad():
        for batch_images, valid_mask in dataloader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            outputs = outputs.cpu().numpy()
            valid_mask = valid_mask.numpy().astype(bool)
            outputs[~valid_mask] = 0.0
            missing += (~valid_mask).sum()
            features.append(outputs)

    stacked = np.concatenate(features, axis=0)
    LOGGER.info("Feature extraction complete. Missing images: %d", missing)
    assert stacked.shape[1] == feature_dim
    return stacked


def prepare_image_paths(df: pd.DataFrame, images_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for link in df["image_link"].fillna(""):
        filename = Path(link.split("?", 1)[0]).name or "missing.jpg"
        paths.append(images_dir / filename)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ViT image embeddings")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory containing train/test CSV",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/images"),
        help="Directory containing downloaded images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Where to store .npy embeddings",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    csv_path = args.dataset_dir / f"{args.split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    image_paths = prepare_image_paths(df, args.images_dir)

    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        shuffle=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    feature_dim = model(torch.zeros(1, 3, 224, 224, device=device)).shape[1]

    features = infer_embeddings(model, dataloader, device, feature_dim)

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{args.split}_image_embeddings.npy"
    np.save(output_path, features)
    LOGGER.info("Saved embeddings to %s", output_path)


if __name__ == "__main__":
    main()
