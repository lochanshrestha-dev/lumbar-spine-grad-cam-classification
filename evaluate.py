%%writefile /kaggle/working/evaluate.py
#!/usr/bin/env python3
"""
Evaluation script for lumbar spine MRI classification model.
Computes metrics (QWK, AUC, sensitivity, specificity, confusion matrices)
on internal validation split and optional external dataset.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration (matches train.py)
# ----------------------------------------------------------------------
CFG = {
    'image_size': 512,
    'batch_size': 32,
    'num_workers': 2,
    'val_frac': 0.15,
    'seed': 42,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

CONDITIONS = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis'
]
LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
LABEL_COLS = [f'{c}_{l}' for c in CONDITIONS for l in LEVELS]  # 25 total

# Map string labels to ordinal integers
LABEL_MAP = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}


# ----------------------------------------------------------------------
# Model definition (must match train.py)
# ----------------------------------------------------------------------
class LumbarModel(nn.Module):
    """EfficientNet-B4 backbone + classification head."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        self.head = nn.Linear(1792, 75)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits.view(-1, 25, 3)


# ----------------------------------------------------------------------
# Dataset (must match evaluate.py exactly)
# ----------------------------------------------------------------------
class LumbarDatasetJPG(Dataset):
    """
    Dataset for lumbar spine MRI JPEGs.
    Searches multiple roots for {study_id}.jpg, uses first found.
    NaN labels are converted to -1.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        jpeg_roots: List[Path],
        transform: Optional[transforms.Compose] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.jpeg_roots = jpeg_roots

        # Initialize transform: if None, use default Normalize
        if transform is None:
            self.transform = transforms.Normalize(mean=CFG['mean'], std=CFG['std'])
        else:
            self.transform = transform

        # Build list of valid indices (those with existing JPEG)
        self.valid_idx = []
        self.image_paths = []
        for idx, row in self.df.iterrows():
            study_id = row['study_id']
            found = False
            for root in self.jpeg_roots:
                img_path = root / f"{study_id}.jpg"
                if img_path.exists():
                    self.image_paths.append(img_path)
                    self.valid_idx.append(idx)
                    found = True
                    break
            if not found:
                warnings.warn(f"JPEG not found for study_id={study_id} in any root: {jpeg_roots}")

        if len(self.valid_idx) == 0:
            raise RuntimeError("No valid JPEGs found in the dataset.")

        # Subset dataframe to valid rows
        self.df_valid = self.df.iloc[self.valid_idx].reset_index(drop=True)

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load image with OpenCV (BGR), convert to RGB
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (CFG['image_size'], CFG['image_size']))

        # Convert to tensor [C,H,W] and normalize to [0,1]
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0

        # Apply transform (always defined)
        img_tensor = self.transform(img_tensor)

        # Labels: convert NaN to -1
        labels = self.df_valid.iloc[idx][LABEL_COLS].values.astype(np.float32)
        labels = np.nan_to_num(labels, nan=-1.0).astype(np.int64)
        return img_tensor, torch.tensor(labels, dtype=torch.long)


# ----------------------------------------------------------------------
# Patient-level split (must match train.py)
# ----------------------------------------------------------------------
def make_patient_split(df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42):
    """
    Patient-level split. Returns (train_df, val_df).
    Steps:
      1. Get unique patient IDs from df['study_id']
      2. Shuffle with seed
      3. Take last (val_frac) fraction as val
      4. Assert no overlap between train and val study_ids (leakage check)
    """
    unique_studies = df['study_id'].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_studies)
    n_val = int(len(shuffled) * val_frac)
    val_studies = set(shuffled[-n_val:] if n_val > 0 else [])
    train_studies = set(shuffled[:-n_val] if n_val > 0 else shuffled)

    train_df = df[df['study_id'].isin(train_studies)].reset_index(drop=True)
    val_df = df[df['study_id'].isin(val_studies)].reset_index(drop=True)

    # Leakage check
    assert set(train_df['study_id']).isdisjoint(set(val_df['study_id'])), \
        "Overlap between train and val study_ids!"
    return train_df, val_df


# ----------------------------------------------------------------------
# Model loading with DataParallel handling
# ----------------------------------------------------------------------
def load_model(ckpt_path: Path, device: torch.device) -> LumbarModel:
    """Load checkpoint, unwrap DataParallel if needed, return eval model."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = LumbarModel().to(device)
    state_dict = checkpoint['model_state']
    # Remove 'module.' prefix if present (DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model inference on the entire dataloader.
    Returns:
        preds: [N, 25] int (argmax over classes 0-2)
        labels: [N, 25] int (ground truth, -1 for missing)
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Inference"):
            images = images.to(device)
            logits = model(images)  # [B, 25, 3]
            preds = torch.argmax(logits, dim=2)  # [B, 25]
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_preds), np.vstack(all_labels)


# ----------------------------------------------------------------------
# Metrics computation
# ----------------------------------------------------------------------
def compute_all_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, dict]:
    """
    Compute per-label and macro metrics.
    Returns dictionary with keys:
        - per label: LABEL_COLS[i] -> dict with kappa, auc, severe_sens, severe_spec, confusion_matrix
        - 'macro_kappa': float
        - 'macro_auc': float
    NaN values are handled gracefully (set to np.nan).
    """
    per_label = {}
    kappas = []
    aucs = []

    for i, col in enumerate(LABEL_COLS):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        mask = y_true != -1
        y_true_valid = y_true[mask]
        y_pred_valid = y_pred[mask]

        if len(y_true_valid) == 0:
            per_label[col] = {
                'kappa': np.nan,
                'auc': np.nan,
                'severe_sens': np.nan,
                'severe_spec': np.nan,
                'confusion_matrix': np.full((3, 3), np.nan, dtype=np.float32),
            }
            continue

        # Quadratic weighted kappa
        try:
            kappa = cohen_kappa_score(y_true_valid, y_pred_valid, weights='quadratic')
        except ValueError:
            kappa = np.nan
        kappas.append(kappa)

        # AUC: Severe (2) vs rest (0/1)
        binary_true = (y_true_valid == 2).astype(int)
        binary_pred = (y_pred_valid == 2).astype(int)
        if len(np.unique(binary_true)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(binary_true, binary_pred)
        aucs.append(auc)

        # Sensitivity and specificity for Severe (class 2) using binary confusion matrix
        binary_cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
        tn, fp, fn, tp = binary_cm.ravel()
        severe_sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        severe_spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        # Full confusion matrix [3,3]
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1, 2])
        per_label[col] = {
            'kappa': float(kappa) if not np.isnan(kappa) else np.nan,
            'auc': float(auc) if not np.isnan(auc) else np.nan,
            'severe_sens': float(severe_sens) if not np.isnan(severe_sens) else np.nan,
            'severe_spec': float(severe_spec) if not np.isnan(severe_spec) else np.nan,
            'confusion_matrix': cm.astype(np.float32),
        }

    # Macro averages (ignore NaN)
    macro_kappa = np.nanmean(kappas) if kappas else np.nan
    macro_auc = np.nanmean(aucs) if aucs else np.nan

    result = {
        **per_label,
        'macro_kappa': float(macro_kappa),
        'macro_auc': float(macro_auc),
    }
    return result


# ----------------------------------------------------------------------
# Printing and saving results
# ----------------------------------------------------------------------
def print_results_table(metrics: Dict, title: str):
    """Print formatted table of per-label metrics."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    header = f"{'Condition':<30} | {'Level':<8} | {'Kappa':<6} | {'AUC':<6} | {'Sev_Sens':<8} | {'Sev_Spec':<8}"
    print(header)
    print("-" * 80)

    for i, col in enumerate(LABEL_COLS):
        condition = CONDITIONS[i // 5]
        level = LEVELS[i % 5]
        m = metrics[col]
        kappa = f"{m['kappa']:.4f}" if not np.isnan(m['kappa']) else "  nan  "
        auc = f"{m['auc']:.4f}" if not np.isnan(m['auc']) else "  nan  "
        sens = f"{m['severe_sens']:.4f}" if not np.isnan(m['severe_sens']) else "  nan  "
        spec = f"{m['severe_spec']:.4f}" if not np.isnan(m['severe_spec']) else "  nan  "
        print(f"{condition:<30} | {level:<8} | {kappa:^6} | {auc:^6} | {sens:^8} | {spec:^8}")

    print("-" * 80)
    macro_kappa = f"{metrics['macro_kappa']:.4f}" if not np.isnan(metrics['macro_kappa']) else "  nan  "
    macro_auc = f"{metrics['macro_auc']:.4f}" if not np.isnan(metrics['macro_auc']) else "  nan  "
    print(f"{'MACRO':<30} | {'-':<8} | {macro_kappa:^6} | {macro_auc:^6} | {'-':^8} | {'-':^8}")
    print("=" * 80 + "\n")


def save_results(metrics: Dict, out_dir: Path):
    """
    Save metrics as JSON (convert ndarray to list, NaN to None)
    and confusion matrices as a 5x5 PNG grid.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON serialization: convert NaN to None and confusion matrices to lists
    json_metrics = {}
    for key, val in metrics.items():
        if key.endswith('_kappa') or key.endswith('_auc'):
            json_metrics[key] = None if np.isnan(val) else val
        elif isinstance(val, dict) and 'confusion_matrix' in val:
            cm = val['confusion_matrix']
            json_metrics[key] = {
                'kappa': None if np.isnan(val['kappa']) else val['kappa'],
                'auc': None if np.isnan(val['auc']) else val['auc'],
                'severe_sens': None if np.isnan(val['severe_sens']) else val['severe_sens'],
                'severe_spec': None if np.isnan(val['severe_spec']) else val['severe_spec'],
                'confusion_matrix': cm.tolist() if isinstance(cm, np.ndarray) else cm,
            }
        else:
            json_metrics[key] = val

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(json_metrics, f, indent=2)

    # Confusion matrices grid: 5 conditions x 5 levels
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    fig.suptitle("Confusion Matrices (rows=conditions, cols=levels)", fontsize=16)

    for i, condition in enumerate(CONDITIONS):
        for j, level in enumerate(LEVELS):
            label = f"{condition}_{level}"
            cm = metrics[label]['confusion_matrix']
            ax = axes[i, j]
            # Display zero-filled version to avoid imshow error on all-NaN matrices
            cm_display = np.nan_to_num(cm, nan=0.0)
            im = ax.imshow(cm_display, cmap='Blues', interpolation='nearest')
            ax.set_title(f"{condition}\n{level}", fontsize=8)
            ax.set_xlabel("Predicted", fontsize=6)
            ax.set_ylabel("Actual", fontsize=6)
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(['N/Mild', 'Mod', 'Sev'], fontsize=5)
            ax.set_yticklabels(['N/Mild', 'Mod', 'Sev'], fontsize=5)
            # Add text annotations from original cm (NaNs become no text)
            for irow in range(3):
                for icol in range(3):
                    val = cm[irow, icol]
                    if not np.isnan(val):
                        ax.text(icol, irow, int(val), ha="center", va="center",
                                color="white" if val > cm_display.max() / 2 else "black", fontsize=6)

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI and main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate lumbar spine MRI classification model")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint file path")
    parser.add_argument("--csv", type=Path, required=True, help="Path to train.csv (for internal validation split)")
    parser.add_argument("--jpeg_roots", type=Path, nargs="+", required=True,
                        help="Directories to search for JPEGs (internal dataset)")
    parser.add_argument("--out_dir", type=Path, default=Path("/kaggle/working/eval_outputs"),
                        help="Output directory for results")
    parser.add_argument("--val_frac", type=float, default=CFG['val_frac'],
                        help="Validation fraction for patient split")
    parser.add_argument("--seed", type=int, default=CFG['seed'], help="Random seed for split")
    parser.add_argument("--batch_size", type=int, default=CFG['batch_size'], help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--ext_csv", type=Path, default=None,
                        help="Optional external validation CSV")
    parser.add_argument("--ext_jpeg_roots", type=Path, nargs="+", default=None,
                        help="JPEG directories for external dataset (required if --ext_csv given)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---- Internal validation ----
    df = pd.read_csv(args.csv)
    if 'study_id' not in df.columns:
        raise ValueError("CSV must contain 'study_id' column")

    # Map string labels to ordinal integers
    for col in LABEL_COLS:
        if col in df.columns:
            df[col] = df[col].map(LABEL_MAP)

    # Ensure study_id remains integer after map() operation
    df['study_id'] = df['study_id'].astype(int)

    _, val_df = make_patient_split(df, val_frac=args.val_frac, seed=args.seed)
    print(f"Internal validation set size: {len(val_df)} studies")

    # LumbarDatasetJPG handles normalization internally
    val_dataset = LumbarDatasetJPG(val_df, args.jpeg_roots, transform=None)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=CFG['num_workers'],
        pin_memory=(device.type == 'cuda'),
    )

    model = load_model(args.ckpt, device)
    preds, labels = run_inference(model, val_loader, device)
    metrics = compute_all_metrics(preds, labels)

    internal_out = args.out_dir / "internal"
    print_results_table(metrics, "=== Internal Validation ===")
    save_results(metrics, internal_out)

    # ---- External validation (if provided) ----
    if args.ext_csv is not None:
        if args.ext_jpeg_roots is None:
            raise ValueError("--ext_jpeg_roots must be provided when --ext_csv is given")
        df_ext = pd.read_csv(args.ext_csv)
        if 'study_id' not in df_ext.columns:
            raise ValueError("External CSV must contain 'study_id' column")

        # Map string labels to ordinal integers for external CSV
        for col in LABEL_COLS:
            if col in df_ext.columns:
                df_ext[col] = df_ext[col].map(LABEL_MAP)

        # Ensure study_id remains integer after map() operation
        df_ext['study_id'] = df_ext['study_id'].astype(int)

        ext_dataset = LumbarDatasetJPG(df_ext, args.ext_jpeg_roots, transform=None)
        ext_loader = DataLoader(
            ext_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=CFG['num_workers'],
            pin_memory=(device.type == 'cuda'),
        )
        print(f"External validation set size: {len(ext_dataset)} studies")
        preds_ext, labels_ext = run_inference(model, ext_loader, device)
        metrics_ext = compute_all_metrics(preds_ext, labels_ext)
        external_out = args.out_dir / "external"
        print_results_table(metrics_ext, "=== External Validation ===")
        save_results(metrics_ext, external_out)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()