import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import gc
import random
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Clear any leftover GPU memory
gc.collect()
torch.cuda.empty_cache()

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CFG = dict(
    train_csv     = '/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification/train.csv',
    jpeg_roots    = [
        '/kaggle/input/datasets/drlochanshrestha/lumbar-jpg/kaggle/working/jpgs',
    ],
    image_size    = 512,
    val_frac      = 0.15,
    test_frac     = 0.15,   # NEW: held-out test set, never used for checkpoint selection
    seed          = 42,
    backbone      = 'efficientnet_b4',
    pretrained    = True,
    drop_rate     = 0.3,
    epochs        = 35,
    warmup_epochs = 3,
    batch_size    = 8,        # reduced from 16 to avoid OOM
    lr            = 3e-5,
    weight_decay  = 1e-2,
    num_workers   = 4,
    pin_memory    = True,
    crop_lumbar   = True,
    out_dir       = '/kaggle/working',
    ckpt_name     = 'lumbar_best_v3_nohflip.pth',   # renamed: v3, no-hflip fix
    mean          = [0.485, 0.456, 0.406],
    std           = [0.229, 0.224, 0.225],
)

CONDITIONS = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis',
]
LEVELS     = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
LABEL_COLS = [f'{c}_{l}' for c in CONDITIONS for l in LEVELS]
LABEL_MAP  = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}

# Which LABEL_COLS are laterality-specific (left/right) vs not.
# Used post-hoc to check whether removing the flip specifically helped
# the outputs that were affected by the labeling bug (PLOS ONE reviewer #2).
LATERALITY_COLS = [c for c in LABEL_COLS if c.startswith(('left_', 'right_'))]
NON_LATERALITY_COLS = [c for c in LABEL_COLS if c not in LATERALITY_COLS]


# ----------------------------------------------------------------------
# Ordinal-aware loss
# ----------------------------------------------------------------------
class OrdinalCrossEntropyLoss(nn.Module):
    """
    CrossEntropy scaled by ordinal distance between predicted and true class.

    Distance weight matrix (symmetric):
        pred:  0    1    2
    true: 0 [1.0, 1.5, 2.0]
          1 [1.5, 1.0, 1.5]
          2 [2.0, 1.5, 1.0]
    """
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.ignore_index  = ignore_index
        self.class_weights = class_weights
        w = torch.tensor([[1.0, 1.5, 2.0],
                          [1.5, 1.0, 1.5],
                          [2.0, 1.5, 1.0]])
        self.register_buffer('dist_weights', w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [N, 3]
            targets: [N]
        Returns:
            scalar loss
        """
        mask = targets != self.ignore_index
        if mask.sum() == 0:
            return logits.sum() * 0.0
        lg = logits[mask]
        tg = targets[mask]
        ce       = F.cross_entropy(lg, tg, weight=self.class_weights, reduction='none')
        pred_cls = lg.argmax(dim=1)
        dist_w   = self.dist_weights.to(tg.device)[tg, pred_cls]
        return (ce * dist_w).mean()


# ----------------------------------------------------------------------
# Class weights (exponent 1.0)
# ----------------------------------------------------------------------
def compute_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Compute inverse frequency class weights.

    Args:
        df:     training DataFrame with LABEL_COLS already mapped to int
        device: torch device

    Returns:
        float32 tensor [3]
    """
    all_labels = pd.to_numeric(df[LABEL_COLS].values.flatten(), errors='coerce')
    all_labels = all_labels[~np.isnan(all_labels)].astype(int)
    unique, counts = np.unique(all_labels, return_counts=True)
    total = counts.sum()
    class_counts = np.ones(3, dtype=np.float32)
    for u, c in zip(unique, counts):
        class_counts[u] = c
    weights = (total / (3 * class_counts)) ** 1.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ----------------------------------------------------------------------
# Patient-level split — NOW THREE-WAY (train / val / test)
# ----------------------------------------------------------------------
def make_patient_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level three-way split. Returns (train_df, val_df, test_df).

    val_df:  used ONLY for checkpoint/epoch selection during training.
             Never used for headline metrics.
    test_df: held out completely from training and model selection.
             ALL headline performance metrics reported in the manuscript
             (AUC, QWK, sensitivity, specificity, and their confidence
             intervals) are computed on this set only.

    This separation addresses PLOS ONE reviewer #2, comment 1: selecting
    the checkpoint by best validation QWK and then reporting headline
    metrics on that same validation set introduces optimistic bias.
    """
    unique_studies = df['study_id'].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_studies)

    n_total = len(shuffled)
    n_val   = int(n_total * val_frac)
    n_test  = int(n_total * test_frac)

    test_studies  = set(shuffled[:n_test])
    val_studies   = set(shuffled[n_test:n_test + n_val])
    train_studies = set(shuffled[n_test + n_val:])

    train_df = df[df['study_id'].isin(train_studies)].reset_index(drop=True)
    val_df   = df[df['study_id'].isin(val_studies)].reset_index(drop=True)
    test_df  = df[df['study_id'].isin(test_studies)].reset_index(drop=True)

    assert set(train_df['study_id']).isdisjoint(set(val_df['study_id'])), \
        "Data leakage: overlap between train and val"
    assert set(train_df['study_id']).isdisjoint(set(test_df['study_id'])), \
        "Data leakage: overlap between train and test"
    assert set(val_df['study_id']).isdisjoint(set(test_df['study_id'])), \
        "Data leakage: overlap between val and test"

    print(f"Train: {len(train_df)}, Val (checkpoint selection): {len(val_df)}, "
          f"Test (held out, headline metrics): {len(test_df)}")

    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        severe_counts = (split_df[LABEL_COLS] == 2).sum().sum()
        print(f"  {name}: {severe_counts} Severe-class labels across all outputs")

    return train_df, val_df, test_df


# ----------------------------------------------------------------------
# Lumbar crop
# ----------------------------------------------------------------------
def crop_lumbar_region(
    img: np.ndarray,
    top_frac: float = 0.20,
    bot_frac: float = 0.80,
) -> np.ndarray:
    """
    Crop to central lumbar zone. Removes top 20% and bottom 20%.

    Args:
        img:      uint8 [H, W, 3]
        top_frac: fraction to remove from top
        bot_frac: fraction to keep up to

    Returns:
        uint8 [H', W, 3]
    """
    H = img.shape[0]
    return img[int(H * top_frac):int(H * bot_frac), :, :]


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class LumbarModel(nn.Module):
    """EfficientNet-B4 backbone + dropout + classification head."""
    def __init__(self, pretrained: bool = True, drop_rate: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=pretrained, num_classes=0)
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(1792, 75),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, 25, 3]."""
        return self.head(self.backbone(x)).view(-1, 25, 3)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class LumbarDatasetJPG(Dataset):
    """
    JPEG-based dataset. Searches jpeg_roots in order for {study_id}.jpg.
    Optionally crops to lumbar region before resize.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        jpeg_roots: List[Path],
        transform: Optional[transforms.Compose] = None,
        crop_lumbar: bool = False,
    ):
        self.df          = df.reset_index(drop=True)
        self.jpeg_roots  = [Path(r) for r in jpeg_roots]
        self.crop_lumbar = crop_lumbar
        self.transform   = (transforms.Normalize(mean=CFG['mean'], std=CFG['std'])
                            if transform is None else transform)
        self.valid_idx   = []
        self.image_paths = []
        for idx, row in self.df.iterrows():
            study_id = int(row['study_id'])
            for root in self.jpeg_roots:
                img_path = root / f"{study_id}.jpg"
                if img_path.exists():
                    self.image_paths.append(img_path)
                    self.valid_idx.append(idx)
                    break
            else:
                warnings.warn(f"JPEG not found for study_id={study_id}")
        if len(self.valid_idx) == 0:
            raise RuntimeError("No valid JPEGs found.")
        self.df_valid = self.df.iloc[self.valid_idx].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_bgr = cv2.imread(str(self.image_paths[idx]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.crop_lumbar:
            img_rgb = crop_lumbar_region(img_rgb)
        img_resized = cv2.resize(
            img_rgb, (CFG['image_size'], CFG['image_size']),
            interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        img_tensor = self.transform(img_tensor)
        labels = self.df_valid.iloc[idx][LABEL_COLS].values.astype(np.float32)
        labels = np.nan_to_num(labels, nan=-1.0).astype(np.int64)
        return img_tensor, torch.tensor(labels, dtype=torch.long)

    def study_id_at(self, idx: int) -> int:
        return int(self.df_valid.iloc[idx]['study_id'])


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def print_kappa_breakdown(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
) -> float:
    """Print per condition-level kappa and class distribution."""
    print(f"\n{'Condition-level':45s}  {'kappa':>7}  {'dist 0/1/2':>15}")
    print("-" * 75)
    kappas = []
    for j, col in enumerate(LABEL_COLS):
        mask   = all_labels[:, j] >= 0
        y_true = all_labels[mask, j]
        y_pred = all_preds[mask, j]
        if len(np.unique(y_true)) < 2:
            continue
        k    = cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=[0, 1, 2])
        dist = np.bincount(y_true, minlength=3)
        kappas.append(k)
        print(f"{col:45s}  {k:7.3f}  {dist[0]:5d}/{dist[1]:5d}/{dist[2]:5d}")
    macro = float(np.mean(kappas)) if kappas else float('nan')
    print(f"\nMACRO KAPPA: {macro:.4f}")
    return macro


def check_severe_predictions(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
) -> None:
    """Warn if model never predicts Severe (class collapse)."""
    mask = all_labels >= 0
    total_severe_true = (all_labels[mask] == 2).sum()
    total_severe_pred = (all_preds[mask]  == 2).sum()
    print(f"Severe ground truth: {total_severe_true}")
    print(f"Severe predicted:    {total_severe_pred}")
    if total_severe_pred == 0:
        warnings.warn("Model is never predicting Severe — class collapse!")


def print_laterality_comparison(all_preds: np.ndarray, all_labels: np.ndarray) -> None:
    """
    NEW: compare mean kappa on laterality-specific outputs (left/right —
    previously affected by the horizontal-flip label bug) vs non-laterality
    outputs (spinal canal stenosis — unaffected). Run this on the held-out
    test set after training to check whether removing RandomHorizontalFlip
    specifically helped the affected outputs, for the PLOS ONE response letter.
    """
    def _group_kappa(cols):
        kappas = []
        for col in cols:
            j = LABEL_COLS.index(col)
            mask = all_labels[:, j] >= 0
            y_true, y_pred = all_labels[mask, j], all_preds[mask, j]
            if len(np.unique(y_true)) < 2:
                continue
            kappas.append(cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=[0, 1, 2]))
        return float(np.mean(kappas)) if kappas else float('nan'), len(kappas)

    lat_kappa, lat_n = _group_kappa(LATERALITY_COLS)
    nonlat_kappa, nonlat_n = _group_kappa(NON_LATERALITY_COLS)
    print(f"\nLaterality-specific outputs (n={lat_n}, previously flip-bug-affected): mean kappa = {lat_kappa:.4f}")
    print(f"Non-laterality outputs (n={nonlat_n}, unaffected):                      mean kappa = {nonlat_kappa:.4f}")


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    loss_fn: nn.Module,
    epoch: int,
) -> float:
    """Train for one epoch with mixed precision. Returns average loss."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device.type):
            logits = model(images)
            loss   = loss_fn(logits.view(-1, 3), labels.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


# ----------------------------------------------------------------------
# Validation / evaluation loop
# ----------------------------------------------------------------------
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    return_probs: bool = False,
):
    """
    Validate/evaluate model.

    Returns:
        val_loss, macro_kappa, all_preds [N,25], all_labels [N,25]
        and, if return_probs=True, also all_probs [N,25,3]
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += loss_fn(logits.view(-1, 3), labels.view(-1)).item()
            probs = F.softmax(logits, dim=2)
            all_preds.append(torch.argmax(logits, dim=2).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if return_probs:
                all_probs.append(probs.cpu().numpy())
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    kappas = []
    for i in range(len(LABEL_COLS)):
        y_true, y_pred = all_labels[:, i], all_preds[:, i]
        mask = y_true != -1
        y_t, y_p = y_true[mask], y_pred[mask]
        if len(y_t) == 0 or len(np.unique(y_t)) < 2:
            continue
        try:
            kappas.append(cohen_kappa_score(y_t, y_p, weights='quadratic'))
        except ValueError:
            pass
    macro = float(np.nanmean(kappas)) if kappas else float('nan')
    if return_probs:
        all_probs = np.concatenate(all_probs, axis=0)
        return total_loss / len(loader), macro, all_preds, all_labels, all_probs
    return total_loss / len(loader), macro, all_preds, all_labels


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
random.seed(CFG['seed'])

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_dir = Path(CFG['out_dir'])
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Device: {device} — {torch.cuda.get_device_name(0)}")
print(f"GPU memory free: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB")

# Load and map labels
df = pd.read_csv(CFG['train_csv'])
for col in LABEL_COLS:
    if col in df.columns:
        df[col] = df[col].map(LABEL_MAP)
df['study_id'] = df['study_id'].astype(int)

# THREE-WAY split (was two-way). test_df is held out from training and
# checkpoint selection entirely — see make_patient_split docstring.
train_df, val_df, test_df = make_patient_split(
    df, val_frac=CFG['val_frac'], test_frac=CFG['test_frac'], seed=CFG['seed'])
print("Leakage check passed.")

class_weights = compute_class_weights(train_df, device)
print(f"Class weights: {class_weights.cpu().numpy()}")

# ------------------------------------------------------------------
# Transforms — RandomHorizontalFlip REMOVED following peer review
# (PLOS ONE reviewer #2, comment 5).
#
# A horizontal flip reverses left/right laterality on axial images but
# has a different anatomical meaning on sagittal images, since the three
# MRI sequences (sagittal T2, sagittal T1, axial T2) are stacked as
# channels of a single composite image. Because 20 of 25 outputs are
# laterality-specific (left/right foraminal narrowing, left/right
# subarticular stenosis), flipping without swapping the corresponding
# labels injected anatomically inconsistent training signal for these
# outputs. Rather than build a plane-aware flip+label-swap transform
# under review deadline pressure, horizontal flip is removed entirely;
# vertical flip was never used. RandomRotation and ColorJitter are
# retained as they do not have this laterality confound.
# ------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=CFG['mean'], std=CFG['std']),
])

jpeg_roots = [Path(r) for r in CFG['jpeg_roots']]

train_dataset = LumbarDatasetJPG(
    train_df, jpeg_roots,
    transform=train_transform,
    crop_lumbar=CFG['crop_lumbar'],
)
val_dataset = LumbarDatasetJPG(
    val_df, jpeg_roots,
    transform=None,
    crop_lumbar=CFG['crop_lumbar'],
)
# NEW: held-out test set, same treatment as val (no augmentation)
test_dataset = LumbarDatasetJPG(
    test_df, jpeg_roots,
    transform=None,
    crop_lumbar=CFG['crop_lumbar'],
)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, "
      f"Test samples: {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset, batch_size=CFG['batch_size'], shuffle=True,
    num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'], drop_last=True)
val_loader = DataLoader(
    val_dataset, batch_size=CFG['batch_size'], shuffle=False,
    num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'])
test_loader = DataLoader(
    test_dataset, batch_size=CFG['batch_size'], shuffle=False,
    num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'])

# Model
model = LumbarModel(
    pretrained=CFG['pretrained'],
    drop_rate=CFG['drop_rate'],
).to(device)

# Optimizer + scheduler with warmup
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG['lr'],
    weight_decay=CFG['weight_decay'],
)
warmup = LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=CFG['warmup_epochs'],
)
cosine = CosineAnnealingLR(
    optimizer,
    T_max=CFG['epochs'] - CFG['warmup_epochs'],
    eta_min=CFG['lr'] * 1e-2,
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[CFG['warmup_epochs']],
)
scaler  = GradScaler(device.type)
loss_fn = OrdinalCrossEntropyLoss(
    class_weights=class_weights,
    ignore_index=-1,
)

best_kappa = -1.0
for epoch in range(CFG['epochs']):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scaler, device, loss_fn, epoch)
    # Checkpoint selection uses val_loader ONLY — never test_loader.
    val_loss, macro_kappa, all_preds, all_labels = validate(
        model, val_loader, device, loss_fn)
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Kappa: {macro_kappa:.4f} | LR: {lr:.2e}")

    # Diagnostics every 5 epochs
    if (epoch + 1) % 5 == 0:
        print_kappa_breakdown(all_preds, all_labels)
        check_severe_predictions(all_preds, all_labels)

    checkpoint = {
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'val_kappa':       macro_kappa,
        'cfg':             CFG,
    }
    torch.save(checkpoint, out_dir / 'last_model.pth')
    if macro_kappa > best_kappa:
        best_kappa = macro_kappa
        torch.save(checkpoint, out_dir / CFG['ckpt_name'])
        print(f"  -> New best model saved (kappa {best_kappa:.4f})")

# ------------------------------------------------------------------
# FINAL EVALUATION — on held-out test set, using the best checkpoint
# selected by val_kappa above. These are the numbers to report as
# headline results in the revised manuscript (addresses reviewer #2,
# comment 1: checkpoint selection and headline metrics must not share
# the same data).
# ------------------------------------------------------------------
print("\n" + "=" * 75)
print("FINAL TEST-SET EVALUATION (held out, never used for training or "
      "checkpoint selection)")
print("=" * 75)

best_ckpt = torch.load(out_dir / CFG['ckpt_name'], map_location=device)
model.load_state_dict(best_ckpt['model_state'])
print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']+1} "
      f"(val kappa {best_ckpt['val_kappa']:.4f})")

test_loss, test_macro_kappa, test_preds, test_labels, test_probs = validate(
    model, test_loader, device, loss_fn, return_probs=True)

print(f"\nTest Loss: {test_loss:.4f} | Test Macro Kappa: {test_macro_kappa:.4f}")
print_kappa_breakdown(test_preds, test_labels)
check_severe_predictions(test_preds, test_labels)
print_laterality_comparison(test_preds, test_labels)

# ------------------------------------------------------------------
# Save per-study, per-output predictions with class probabilities.
# This CSV is what's needed to compute bootstrapped 95% confidence
# intervals, precision/PPV/F1, and exact per-output denominators for
# the PLOS ONE response letter (reviewer #2, comments 2 and 3).
# ------------------------------------------------------------------
records = []
for i in range(len(test_dataset)):
    study_id = test_dataset.study_id_at(i)
    for j, col in enumerate(LABEL_COLS):
        true_label = test_labels[i, j]
        if true_label == -1:
            continue  # missing annotation, excluded as in training
        records.append({
            'study_id':          study_id,
            'condition_level':   col,
            'is_laterality':     col in LATERALITY_COLS,
            'true_label':        int(true_label),
            'pred_label':        int(test_preds[i, j]),
            'prob_normal_mild':  float(test_probs[i, j, 0]),
            'prob_moderate':     float(test_probs[i, j, 1]),
            'prob_severe':       float(test_probs[i, j, 2]),
        })

test_results_df = pd.DataFrame(records)
test_results_csv = out_dir / 'test_set_predictions.csv'
test_results_df.to_csv(test_results_csv, index=False)
print(f"\nSaved {len(test_results_df)} test-set predictions to {test_results_csv}")
