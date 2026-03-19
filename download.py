from __future__ import annotations

"""Bulk image downloader with retry logic for the challenge datasets.

Run from repository root, e.g.:

    python download.py --split train --output-dir data/images

This script avoids hard-coded paths and keeps terminal runs responsive.
"""

import argparse
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm


LOGGER = logging.getLogger("download")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download product images")
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Which CSV split to read (train/test)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory containing train.csv/test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/images"),
        help="Directory where downloaded images are stored",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(16, cpu_count()),
        help="Parallel download worker count",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that already exist",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retry attempts for transient errors",
    )
    return parser.parse_args()


def load_links(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "image_link" not in df.columns:
        raise ValueError("CSV missing 'image_link' column")
    return df["image_link"].fillna("")


def build_session(retries: int, backoff: float = 0.5) -> requests.Session:
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "AmazonMLChallengeDownloader/1.0 (+https://github.com/)"
        }
    )
    return session


def _download_single(
    link: str,
    output_dir: Path,
    timeout: float,
    retries: int,
    resume: bool,
) -> tuple[str, bool]:
    if not isinstance(link, str) or not link.strip():
        return link, False

    filename = Path(link.split("?")[0]).name or "unknown.jpg"
    target = output_dir / filename

    if resume and target.exists() and target.stat().st_size > 0:
        return str(target), True

    session = build_session(retries=retries)
    try:
        with session.get(link, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
        return str(target), True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to download %s -> %s", link, exc)
        return link, False


def download_all(
    links: Iterable[str],
    output_dir: Path,
    max_workers: int,
    timeout: float,
    retries: int,
    resume: bool,
) -> None:
    freeze_support()
    output_dir.mkdir(parents=True, exist_ok=True)

    worker = partial(
        _download_single,
        output_dir=output_dir,
        timeout=timeout,
        retries=retries,
        resume=resume,
    )

    with Pool(processes=max_workers) as pool:
        results = list(tqdm(pool.imap(worker, links), total=len(links)))

    failed = [link for link, ok in results if not ok]
    if failed:
        LOGGER.error("%d downloads failed. See log for details.", len(failed))
    else:
        LOGGER.info("All downloads completed successfully.")


def main() -> None:
    configure_logging()
    args = parse_args()

    csv_file = args.dataset_dir / f"{args.split}.csv"
    LOGGER.info("Loading image links from %s", csv_file)
    links = load_links(csv_file)

    LOGGER.info(
        "Starting %s downloads to %s with %d workers",
        args.split,
        args.output_dir,
        args.max_workers,
    )

    download_all(
        links.tolist(),
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        timeout=args.timeout,
        retries=args.retries,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
& .venv\Scripts\Activate.ps1
python - <<'PY'
import numpy as np, pandas as pd
train = pd.read_csv("dataset/train.csv")
X_txt = np.load("outputs/train_text_embeddings.npy")
X_test_txt = np.load("outputs/test_text_embeddings.npy")
assert len(train) == X_txt.shape[0]
print("train shape:", X_txt.shape, "| test shape:", X_test_txt.shape)
PY