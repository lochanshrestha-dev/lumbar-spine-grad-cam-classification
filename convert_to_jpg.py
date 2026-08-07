#!/usr/bin/env python3
"""
convert_to_jpg.py

One-time batch conversion script that converts all DICOM studies from the
RSNA 2024 lumbar spine dataset to pre-composited JPEG files for fast training.

For each study, the pipeline runs:
  1. series classification (classify_study)
  2. series selection (select_series)
  3. image composition + normalisation (build_study_image)

Output JPEGs are written directly to output_dir (flat structure, no part1/part2 split).

Usage examples:
  # Full conversion (multiprocessing)
  python convert_to_jpg.py --dicom_root /kaggle/input/rsna2024/train \
                           --output_dir /kaggle/input/lumbar-jpg \
                           --image_size 512 --num_workers 8

  # Convert only specific study IDs (for spot checks)
  python convert_to_jpg.py --dicom_root /kaggle/input/rsna2024/train \
                           --output_dir /kaggle/input/lumbar-jpg \
                           --study_ids 1000001 1000002 1000003
"""

import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm

# Local modules (assumed to be in the same repository)
from series_id import classify_study, select_series
from preprocess import build_study_image


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def get_all_study_dirs(dicom_root: Path) -> List[Path]:
    """
    Return sorted list of all study directories under dicom_root.

    Args:
        dicom_root: root directory containing {study_id}/ subdirectories

    Returns:
        List of Path objects, one per study, sorted alphabetically.
        Sorted order ensures deterministic conversion order.

    Behaviour:
        - Includes only directories (not files)
        - Issues a warning if no studies are found
    """
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root directory not found: {dicom_root}")

    study_dirs = [p for p in dicom_root.iterdir() if p.is_dir()]
    if not study_dirs:
        warnings.warn(f"No study directories found in {dicom_root}", UserWarning)
        return []

    study_dirs.sort()  # alphabetical by study_id
    return study_dirs


def convert_study(study_dir: Path, out_path: Path, image_size: int = 512) -> bool:
    """
    Convert a single study to a pre-composited JPEG.

    Args:
        study_dir:  Path to the study directory containing DICOM series
        out_path:   Destination Path for the output JPEG (e.g., /output/12345.jpg)
        image_size: Output image size (width = height), default 512

    Returns:
        True on success, False on failure
    """
    try:
        # 1. Classify series in the study
        series_map = classify_study(study_dir)

        # 2. Select the appropriate series (Sagittal T2, Sagittal T1, Axial T2)
        series_selection = select_series(series_map)

        # 3. Build composite image (uint8 RGB, shape [H,W,3])
        img = build_study_image(study_dir, series_selection, image_size)

        # 4. Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 5. Write JPEG (cv2 expects BGR ordering)
        success = cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            raise IOError(f"cv2.imwrite failed for {out_path}")

        return True

    except Exception as e:
        warnings.warn(f"Failed to convert {study_dir.name}: {e}", UserWarning)
        return False


def convert_all(
    dicom_root: Path,
    output_dir: Path,
    image_size: int = 512,
    num_workers: int = 4,
) -> Dict[str, int]:
    """
    Convert all studies with multiprocessing, writing JPEGs flat into output_dir.

    Args:
        dicom_root:  Root directory of DICOM studies
        output_dir:  Output directory (JPEGs written directly here)
        image_size:  JPEG output size (default 512)
        num_workers: Number of parallel worker processes (default 4)

    Returns:
        dict with keys: 'total', 'success', 'failed'
    """
    # 1. Get all study directories (sorted)
    study_dirs = get_all_study_dirs(dicom_root)
    total = len(study_dirs)
    if total == 0:
        raise RuntimeError("No study directories found; aborting.")

    print(f"Found {total} studies. Output will be written to {output_dir} (flat).")

    # 2. Build task list: (study_dir, out_path, image_size)
    tasks: List[Tuple[Path, Path, int]] = []
    for study_dir in study_dirs:
        out_path = output_dir / f"{study_dir.name}.jpg"
        tasks.append((study_dir, out_path, image_size))

    # 3. Run multiprocessing
    print(f"Converting with {num_workers} workers...")
    success_count = 0
    with Pool(processes=num_workers) as pool:
        results = []
        for result in tqdm(
            pool.starmap(convert_study, tasks),
            total=len(tasks),
            desc="Converting studies",
            unit="study",
        ):
            results.append(result)

        success_count = sum(results)

    failed_count = total - success_count
    print(f"\nConversion finished.")
    print(f"  Total studies: {total}")
    print(f"  Successful:    {success_count}")
    print(f"  Failed:        {failed_count}")

    return {"total": total, "success": success_count, "failed": failed_count}


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert DICOM studies to pre-composited JPEGs (flat output)."
    )
    parser.add_argument(
        "--dicom_root",
        type=Path,
        required=True,
        help="Root directory containing study ID subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory (JPEGs written directly here, no part1/part2 split).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Output JPEG size (width=height). Default 512.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker processes. Default 4.",
    )
    parser.add_argument(
        "--study_ids",
        type=str,
        nargs="+",
        default=None,
        help="Convert only specific study IDs (overrides full conversion). "
             "Useful for spot checks.",
    )
    args = parser.parse_args()

    # --- Spot check mode: convert only a few studies ---
    if args.study_ids is not None:
        print(f"Spot check mode: converting {len(args.study_ids)} specific studies.")
        success_count = 0
        for study_id in args.study_ids:
            study_dir = args.dicom_root / study_id
            if not study_dir.is_dir():
                warnings.warn(f"Study directory not found: {study_dir}", UserWarning)
                continue

            out_path = args.output_dir / f"{study_id}.jpg"
            ok = convert_study(study_dir, out_path, args.image_size)
            success_count += 1 if ok else 0
            print(f"  {study_id} -> {'SUCCESS' if ok else 'FAILED'}")

        print(f"\nSpot check finished: {success_count}/{len(args.study_ids)} succeeded.")
        return

    # --- Full conversion mode ---
    convert_all(
        dicom_root=args.dicom_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()