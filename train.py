%%writefile /kaggle/working/train.py
#!/usr/bin/env python3
"""
Training script for lumbar spine MRI classification.
Model: EfficientNet-B4 (timm) with 25 condition-level outputs, each with 3 ordinal classes.
Uses patient-level split, class weighting, mixed precision, and cosine annealing.
"""

import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CFG = {
    'image_size': 512,
    'batch_size': 16,          # per GPU; effective 32 with DataParallel
    'epochs': 20,
    'lr': 1e-4,
    'weight_decay': 1e-2,
    'val_frac': 0.15,
    'seed': 42,
    'num_workers': 4,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'T_max': 20,
    'eta_min': 1e-6,
    'train_csv': '/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification/train.csv',
    'jpeg_roots': [
        '/kaggle/input/datasets/drlochanshrestha/lumbar-jpg/kaggle/working/jpgs',
    ],
    'out_dir': '/kaggle/working/',
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
# Patient-level split (must match evaluate.py exactly)
# ----------------------------------------------------------------------
def make_patient_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Patient-level split identical to evaluate.py.

    Steps:
        1. unique_studies = df['study_id'].unique()
        2. rng = np.random.default_rng(seed)
        3. shuffled = rng.permutation(unique_studies)
        4. n_val = int(len(shuffled) * val_frac)
        5. val_studies = set(shuffled[-n_val:]) if n_val>0 else set()
        6. train_studies = set(shuffled[:-n_val]) if n_val>0 else set(shuffled)
        7. train_df = df[df['study_id'].isin(train_studies)]
        8. val_df = df[df['study_id'].isin(val_studies)]
        9. assert no overlap
        10. return train_df, val_df
    """
    unique_studies = df['study_id'].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_studies)
    n_val = int(len(shuffled) * val_frac)
    if n_val > 0:
        val_studies = set(shuffled[-n_val:])
        train_studies = set(shuffled[:-n_val])
    else:
        val_studies = set()
        train_studies = set(shuffled)
    train_df = df[df['study_id'].isin(train_studies)].reset_index(drop=True)
    val_df = df[df['study_id'].isin(val_studies)].reset_index(drop=True)
    # Leakage check
    assert set(train_df['study_id']).isdisjoint(set(val_df['study_id'])), \
        "Data leakage: overlap between train and val"
    return train_df, val_df


# ----------------------------------------------------------------------
# Model definition
# ----------------------------------------------------------------------
class LumbarModel(nn.Module):
    """EfficientNet-B4 backbone + classification head."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(1792, 75)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns logits of shape [B, 25, 3]."""
        features = self.backbone(x)   # [B, 1792]
        logits = self.head(features)  # [B, 75]
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

    def __getitem__(self, idx: int):
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
# Class weights from training set
# ----------------------------------------------------------------------
def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse frequency class weights for classes 0,1,2 using all 25 labels.
    Returns tensor of shape [3] (float32).
    Warns if any class has zero count.
    """
    all_labels = df[LABEL_COLS].values.flatten()
    all_labels = pd.to_numeric(all_labels, errors='coerce')  # force numeric, bad values → NaN
    all_labels = all_labels[~np.isnan(all_labels)].astype(int)
    unique, counts = np.unique(all_labels, return_counts=True)
    total = counts.sum()
    class_counts = np.zeros(3, dtype=np.float32)
    for u, c in zip(unique, counts):
        class_counts[u] = c
    weights = total / (3 * class_counts)
    zero_classes = np.where(class_counts == 0)[0]
    if len(zero_classes) > 0:
        warnings.warn(f"Zero count for class(es) {zero_classes}. Setting weight to 1.0.")
        weights[zero_classes] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


# ----------------------------------------------------------------------
# Training one epoch (no scheduler.step() inside)
# ----------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    loss_fn: nn.CrossEntropyLoss,
    epoch: int,
) -> float:
    """
    Train for one epoch with mixed precision.
    Returns average loss.
    """
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)   # [B, 25]

        optimizer.zero_grad()
        # Use device.type (cuda/cpu) for autocast
        with autocast(device.type):
            logits = model(images)                     # [B, 25, 3]
            # Reshape for CrossEntropyLoss
            loss = loss_fn(logits.view(-1, 3), labels.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    return avg_loss


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.CrossEntropyLoss,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Validate the model and compute:
        - val_loss
        - macro kappa (average over 25 labels)
        - per_label_kappas dict
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)                     # [B, 25, 3]
            loss = loss_fn(logits.view(-1, 3), labels.view(-1))
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=2)        # [B, 25]
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            pbar.set_postfix(loss=loss.item())

    val_loss = total_loss / len(loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute per-label quadratic weighted kappa
    per_label_kappas = {}
    kappas = []
    for i, col in enumerate(LABEL_COLS):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]
        mask = y_true != -1
        y_true_valid = y_true[mask]
        y_pred_valid = y_pred[mask]
        if len(y_true_valid) == 0 or len(np.unique(y_true_valid)) < 2:
            kappa = np.nan
        else:
            try:
                kappa = cohen_kappa_score(y_true_valid, y_pred_valid, weights='quadratic')
            except ValueError:
                kappa = np.nan
        per_label_kappas[col] = kappa
        if not np.isnan(kappa):
            kappas.append(kappa)
    macro_kappa = float(np.nanmean(kappas)) if kappas else np.nan
    return val_loss, macro_kappa, per_label_kappas


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    random.seed(CFG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    out_dir = Path(CFG['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_csv_path = Path(CFG['train_csv'])
    df = pd.read_csv(train_csv_path)
    if 'study_id' not in df.columns:
        raise ValueError("CSV must contain 'study_id' column")

    # Map string labels to ordinal integers
    for col in LABEL_COLS:
        if col in df.columns:
            df[col] = df[col].map(LABEL_MAP)

    # Ensure study_id remains integer after map() operation
    df['study_id'] = df['study_id'].astype(int)

    # Patient split
    train_df, val_df = make_patient_split(df, val_frac=CFG['val_frac'], seed=CFG['seed'])
    print(f"Train studies: {len(train_df)}")
    print(f"Val studies: {len(val_df)}")
    print(f"Leakage check passed.")

    # Compute class weights on training set
    class_weights = compute_class_weights(train_df).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Transforms: train with augmentation, val only normalization
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=CFG['mean'], std=CFG['std']),
    ])
    # For val, we pass transform=None to fall back to default normalization (now handled in __init__)
    val_transform = None

    # JPEG roots as Path objects
    jpeg_roots = [Path(root) for root in CFG['jpeg_roots']]

    # Datasets
    train_dataset = LumbarDatasetJPG(train_df, jpeg_roots, transform=train_transform)
    val_dataset = LumbarDatasetJPG(val_df, jpeg_roots, transform=val_transform)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    # Model
    model = LumbarModel(pretrained=True)
    # DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG['lr'],
        weight_decay=CFG['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG['T_max'],
        eta_min=CFG['eta_min'],
    )
    # GradScaler requires device argument (cuda/cpu)
    scaler = GradScaler(device.type if device.type == 'cuda' else 'cpu')
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    best_kappa = -1.0
    # Training loop
    for epoch in range(CFG['epochs']):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            loss_fn, epoch
        )
        val_loss, macro_kappa, per_label_kappas = validate(
            model, val_loader, device, loss_fn
        )
        # Step scheduler once per epoch after validation
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Kappa: {macro_kappa:.4f} | LR: {current_lr:.2e}")

        # Save checkpoint (model.state_dict() works directly, no manual module. prefix)
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_kappa': macro_kappa,
            'cfg': CFG,
        }
        # always save last checkpoint
        torch.save(checkpoint, out_dir / 'last_model.pth')
        if macro_kappa > best_kappa:
            best_kappa = macro_kappa
            torch.save(checkpoint, out_dir / 'best_model.pth')
            print(f"  -> New best model (kappa {best_kappa:.4f})")

    print(f"Training complete. Best kappa: {best_kappa:.4f}")


if __name__ == '__main__':
    main()