#!/usr/bin/env python3
"""
preprocess.py – Lumbar spine MRI preprocessing pipeline.

Loads DICOM series, normalises intensity, and composites them into a single
512×512 3‑channel uint8 image per study.
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pydicom

# Import from series_id module (same package)
from series_id import classify_study, select_series


# ---------------------------------------------------------------------------
# Intensity normalisation
# ---------------------------------------------------------------------------

def normalise_series(volume: np.ndarray) -> np.ndarray:
    """
    Percentile clip and scale to uint8.

    Args:
        volume: float32 array of any shape (slices × H × W or single slice H × W).

    Returns:
        uint8 array of same shape, values in [0, 255].
    """
    p1 = np.percentile(volume, 1)
    p99 = np.percentile(volume, 99)
    clipped = np.clip(volume, p1, p99)
    scaled = (clipped - p1) / (p99 - p1 + 1e-8) * 255
    return scaled.astype(np.uint8)


# ---------------------------------------------------------------------------
# Coverage guard
# ---------------------------------------------------------------------------

def check_coverage(channel: np.ndarray, low: float = 0.05, high: float = 0.99) -> bool:
    """
    Check that foreground pixels are within acceptable range.

    Args:
        channel: uint8 2D array [H, W] (single slice or projection).
        low:     minimum foreground fraction (default 0.05).
        high:    maximum foreground fraction (default 0.99).

    Returns:
        True if coverage is acceptable, False otherwise.
    """
    total_pixels = channel.size
    if total_pixels == 0:
        return False
    foreground = np.count_nonzero(channel)
    frac = foreground / total_pixels
    return low <= frac <= high


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------

def load_dicom_volume(series_dir: Path) -> np.ndarray:
    """
    Load all slices from a DICOM series directory, sorted by
    InstanceNumber (fallback: filename sort).

    Args:
        series_dir: Path to directory containing .dcm files.

    Returns:
        float32 array [N, H, W] where N = number of slices.

    Raises:
        RuntimeError: if zero slices could be loaded.
    """
    slices = []
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise RuntimeError(f"No DICOM files found in {series_dir}")

    for fpath in files:
        try:
            ds = pydicom.dcmread(fpath)
            pixel_data = ds.pixel_array.astype(np.float32)
            # Apply rescale if tags are present
            if hasattr(ds, "RescaleSlope"):
                pixel_data = pixel_data * ds.RescaleSlope
            if hasattr(ds, "RescaleIntercept"):
                pixel_data = pixel_data + ds.RescaleIntercept
            # Sort key: InstanceNumber if available, else filename
            # Cast InstanceNumber to int to keep types consistent
            if hasattr(ds, "InstanceNumber"):
                key = int(ds.InstanceNumber)
            else:
                key = fpath.name
            slices.append((key, pixel_data))
        except Exception as exc:
            warnings.warn(f"Could not read {fpath}: {exc}")
            continue

    if not slices:
        raise RuntimeError(f"Failed to load any valid slice from {series_dir}")

    # Sort slices – fall back to string representation if mixed types occur
    try:
        slices.sort(key=lambda x: x[0])
    except TypeError:
        slices.sort(key=lambda x: str(x[0]))

    volume = np.stack([data for _, data in slices], axis=0)
    return volume


# ---------------------------------------------------------------------------
# Channel builders
# ---------------------------------------------------------------------------

def build_sag_t2_channel(series_dir: Path, image_size: int = 512) -> Optional[np.ndarray]:
    """
    Build sagittal T2 channel: mean projection of 3 central slices.

    Args:
        series_dir: Path to SAG_T2 series directory.
        image_size: output size (default 512).

    Returns:
        uint8 [image_size, image_size] or None on failure.
    """
    try:
        volume = load_dicom_volume(series_dir)         # [N, H, W] float32
        volume = normalise_series(volume)               # uint8
        n = volume.shape[0]
        # Clamp indices to valid range [0, n-1]
        idx_lo = max(0, n // 2 - 1)
        idx_hi = min(n - 1, n // 2 + 1)
        # Take three central slices (if n==1 they are all the same)
        central = volume[idx_lo : idx_hi + 1]           # (<=3, H, W)
        proj = np.mean(central, axis=0)                 # float64
        proj = np.round(proj).astype(np.uint8)          # uint8
        proj = cv2.resize(proj, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        if not check_coverage(proj):
            warnings.warn(f"SAG_T2 coverage check failed for {series_dir}")
            return None
        return proj
    except Exception as exc:
        warnings.warn(f"Failed to build SAG_T2 channel from {series_dir}: {exc}")
        return None


def build_sag_t1_channel(series_dir: Path, image_size: int = 512) -> Optional[np.ndarray]:
    """
    Build sagittal T1 channel: middle slice only.

    Args:
        series_dir: Path to SAG_T1 series directory.
        image_size: output size (default 512).

    Returns:
        uint8 [image_size, image_size] or None on failure.
    """
    try:
        volume = load_dicom_volume(series_dir)          # float32
        volume = normalise_series(volume)                # uint8
        middle = volume[volume.shape[0] // 2]            # uint8
        resized = cv2.resize(middle, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        if not check_coverage(resized):
            warnings.warn(f"SAG_T1 coverage check failed for {series_dir}")
            return None
        return resized
    except Exception as exc:
        warnings.warn(f"Failed to build SAG_T1 channel from {series_dir}: {exc}")
        return None


def build_ax_t2_channel(series_dir: Path, image_size: int = 512) -> Optional[np.ndarray]:
    """
    Build axial T2 channel: middle slice, centre-embedded.

    Args:
        series_dir: Path to AX_T2 series directory.
        image_size: output size (default 512).

    Returns:
        uint8 [image_size, image_size] or None on failure.
    """
    try:
        volume = load_dicom_volume(series_dir)          # float32
        volume = normalise_series(volume)                # uint8
        middle = volume[volume.shape[0] // 2]            # [H, W] uint8
        small_size = image_size // 2                     # 256
        small = cv2.resize(middle, (small_size, small_size), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((image_size, image_size), dtype=np.uint8)
        centre = image_size // 2
        half_small = small_size // 2
        r1, r2 = centre - half_small, centre + half_small
        canvas[r1:r2, r1:r2] = small

        if not check_coverage(canvas):
            warnings.warn(f"AX_T2 coverage check failed for {series_dir}")
            return None
        return canvas
    except Exception as exc:
        warnings.warn(f"Failed to build AX_T2 channel from {series_dir}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Composite function
# ---------------------------------------------------------------------------

def build_study_image(
    study_dir: Path,
    series_selection: dict[str, Optional[str]],
    image_size: int = 512,
) -> np.ndarray:
    """
    Build the final [512, 512, 3] uint8 RGB composite for one study.

    Args:
        study_dir:         Path to study directory (contains series subdirectories).
        series_selection:  Output of select_series() from series_id.py.
                           Keys: 'sag_t2', 'sag_t1', 'ax_t2'.
                           Values: series_id str or None.
        image_size:        Output spatial size (default 512).

    Returns:
        uint8 numpy array [image_size, image_size, 3].
    """
    black = np.zeros((image_size, image_size), dtype=np.uint8)

    # Channel 0 – SAG_T2 (required, black fallback)
    sag_t2_id = series_selection.get("sag_t2")
    if sag_t2_id is not None:
        ch0 = build_sag_t2_channel(study_dir / sag_t2_id, image_size)
        if ch0 is None:
            ch0 = black
    else:
        ch0 = black

    # Channel 1 – SAG_T1 (falls back to Ch0)
    sag_t1_id = series_selection.get("sag_t1")
    if sag_t1_id is not None:
        ch1 = build_sag_t1_channel(study_dir / sag_t1_id, image_size)
        if ch1 is None:
            ch1 = ch0
    else:
        ch1 = ch0

    # Channel 2 – AX_T2 (falls back to Ch0)
    ax_t2_id = series_selection.get("ax_t2")
    if ax_t2_id is not None:
        ch2 = build_ax_t2_channel(study_dir / ax_t2_id, image_size)
        if ch2 is None:
            ch2 = ch0
    else:
        ch2 = ch0

    # Stack into RGB image
    rgb = np.stack([ch0, ch1, ch2], axis=2)   # [H, W, 3]
    return rgb


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Lumbar spine MRI preprocessing")
    parser.add_argument("--study_dir", type=Path, required=True,
                        help="Path to the study directory containing series subdirectories.")
    parser.add_argument("--out_path", type=Path, required=True,
                        help="Output JPEG path (e.g., /path/to/out.jpg).")
    args = parser.parse_args()

    series_map = classify_study(args.study_dir)
    series_selection = select_series(series_map)
    img = build_study_image(args.study_dir, series_selection)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.out_path), bgr)
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()