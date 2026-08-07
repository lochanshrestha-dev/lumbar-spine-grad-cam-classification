#!/usr/bin/env python3
"""
dataset.py – PyTorch Dataset for lumbar spine MRI classification from DICOM files.

This is a slow, on-the-fly preprocessing dataset intended for development
and debugging. For training use the JPEG equivalent.
"""

import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from series_id import classify_study, select_series
from preprocess import build_study_image


# ---------------------------------------------------------------------------
# Label schema – must match the definitions in train.py
# ---------------------------------------------------------------------------

CONDITIONS = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]
LEVELS = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
LABEL_COLS = [f"{c}_{l}" for c in CONDITIONS for l in LEVELS]  # 25 elements

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class LumbarDatasetDICOM(Dataset):
    """
    DICOM-based dataset for lumbar spine MRI classification.

    Runs the full preprocessing pipeline (series classification, intensity
    normalisation, 3-channel composition) on-the-fly. Slow – use for
    development and debugging only. For training use LumbarDatasetJPG.

    Args:
        df: DataFrame with columns ``study_id`` and the 25 label columns.
        dicom_root: Root directory containing subdirectories named by study_id.
        transform: Torchvision transform applied after tensor conversion.
                   If ``None``, a default ``Normalize(mean, std)`` is used.
        image_size: Output image size (default 512).

    Raises:
        RuntimeError: if no valid study directories are found.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dicom_root: Path,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 512,
    ) -> None:
        self.dicom_root = Path(dicom_root)
        self.image_size = image_size

        # Set default transform if none provided
        if transform is None:
            self.transform = transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
        else:
            self.transform = transform

        # Build list of valid samples (study_id + labels)
        self.samples: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            study_id = str(row["study_id"])
            study_dir = self.dicom_root / study_id
            if not study_dir.is_dir():
                warnings.warn(f"Study directory missing, skipping: {study_dir}")
                continue

            # Extract labels, replace NaN with -1
            label_vals = row[LABEL_COLS].values.astype(np.float32)  # NaN stays
            label_vals = np.where(np.isnan(label_vals), -1.0, label_vals)
            label_vals = label_vals.astype(np.int64)

            self.samples.append({"study_id": study_id, "labels": label_vals})

        if not self.samples:
            raise RuntimeError(
                f"No valid study directories found under {self.dicom_root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        study_id = sample["study_id"]
        labels = torch.from_numpy(sample["labels"])  # int64 tensor [25]

        try:
            study_dir = self.dicom_root / study_id
            # Step 1 – classify series
            series_map = classify_study(study_dir)
            # Step 2 – select best series per modality
            series_selection = select_series(series_map)
            # Step 3 – build RGB composite
            img = build_study_image(study_dir, series_selection, self.image_size)
            # img is uint8 [H, W, 3]

            # Convert to float tensor [C, H, W] in [0, 1]
            img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            # Apply normalisation (or any supplied transform)
            img_t = self.transform(img_t)

            return img_t, labels

        except Exception as exc:
            warnings.warn(f"Failed to load study {study_id}: {exc}")
            # Return zero image and all -1 labels so DataLoader doesn't crash
            zero_img = torch.zeros(3, self.image_size, self.image_size)
            fail_labels = torch.full((len(LABEL_COLS),), -1, dtype=torch.int64)
            return zero_img, fail_labels


# ---------------------------------------------------------------------------
# Smoke test (CLI)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke test for LumbarDatasetDICOM"
    )
    parser.add_argument(
        "--csv", type=Path, required=True, help="Path to train.csv (or similar)"
    )
    parser.add_argument(
        "--dicom_root", type=Path, required=True, help="Root directory of DICOM studies"
    )
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Number of samples to test"
    )
    args = parser.parse_args()

    # Load the first few rows
    df = pd.read_csv(args.csv).head(args.num_samples)
    print(f"Loaded {len(df)} rows from {args.csv}")

    # Build dataset
    try:
        dataset = LumbarDatasetDICOM(df, args.dicom_root)
    except RuntimeError as e:
        print(f"Dataset creation failed: {e}")
        return

    print(f"Dataset contains {len(dataset)} valid studies")

    # Iterate over the dataset
    for i in range(len(dataset)):
        start = time.time()
        img_tensor, labels = dataset[i]
        elapsed = time.time() - start

        study_id = dataset.samples[i]["study_id"]
        print(f"\n--- Study {i}: {study_id} ---")
        print(f"Image tensor shape: {img_tensor.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label min: {labels.min().item()}, max: {labels.max().item()}")
        num_missing = (labels == -1).sum().item()
        print(f"Number of missing labels (-1): {num_missing}")
        print(f"Time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()