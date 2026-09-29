#!/usr/bin/env python3
"""
evaluate.py — Reproduce the held-out test-set results reported in the
PLOS ONE manuscript (LumbarNet-25, EfficientNet-B4, corrected pipeline).

What this script does
---------------------
1. Rebuilds the exact three-way patient-level split used in training
   (seed 42; 70% train / 15% validation / 15% test -> 1,383 / 296 / 296).
2. Loads the best checkpoint from the corrected training run
   (lumbar_best_v3_nohflip.pth; epoch 35; validation QWK 0.2738).
3. Applies the same preprocessing as training (lumbar crop: remove top and
   bottom 20%; resize to 512x512; ImageNet normalization; no augmentation).
4. Evaluates the TEST split (default) and reports:
     - per-output QWK, AUC (Severe vs rest), Severe sensitivity/specificity/PPV
     - macro QWK (all outputs with >=2 grades present, as in training)
     - macro AUC (outputs with >=1 Severe case)
     - pooled (micro-averaged) Severe sensitivity/specificity/PPV/F1
     - optional patient-level bootstrap 95% CIs
5. Compares the headline numbers with those reported in the manuscript and
   prints PASS/MISMATCH.

This replaces an earlier evaluate.py that used the pre-revision model head,
a two-way split and no lumbar crop, and therefore could not reproduce the
revised results.

Usage (Kaggle)
--------------
    !python evaluate.py                         # auto-finds checkpoint
    !python evaluate.py --ckpt /path/to/lumbar_best_v3_nohflip.pth
    !python evaluate.py --bootstrap 2000        # add 95% CIs (~2-4 min)
    !python evaluate.py --split val             # checkpoint-selection split
"""

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration — must match train.py exactly
# ----------------------------------------------------------------------
CFG = dict(
    train_csv   = '/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification/train.csv',
    jpeg_roots  = ['/kaggle/input/datasets/drlochanshrestha/lumbar-jpg/kaggle/working/jpgs'],
    image_size  = 512,
    val_frac    = 0.15,
    test_frac   = 0.15,
    seed        = 42,
    drop_rate   = 0.3,
    crop_lumbar = True,
    ckpt_name   = 'lumbar_best_v3_nohflip.pth',
    mean        = [0.485, 0.456, 0.406],
    std         = [0.229, 0.224, 0.225],
    batch_size  = 8,
    num_workers = 2,
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

# Expected values (from the training log and the manuscript) used as a
# reproducibility check.
EXPECTED = {
    'split_sizes':     (1383, 296, 296),
    'severe_counts':   (2146, 438, 505),
    'ckpt_epoch':      35,
    'ckpt_val_kappa':  0.2738,
    'test': dict(macro_qwk=0.3255, macro_auc=0.8211, n_auc=23,
                 sens=0.653, spec=0.862, ppv=0.260, f1=0.372, n_labels=7285),
}


# ----------------------------------------------------------------------
# Split, crop, model, dataset — identical to train.py
# ----------------------------------------------------------------------
def make_patient_split(df, val_frac=0.15, test_frac=0.15, seed=42):
    unique_studies = df['study_id'].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_studies)
    n_total = len(shuffled)
    n_val = int(n_total * val_frac)
    n_test = int(n_total * test_frac)
    test_studies = set(shuffled[:n_test])
    val_studies = set(shuffled[n_test:n_test + n_val])
    train_studies = set(shuffled[n_test + n_val:])
    train_df = df[df['study_id'].isin(train_studies)].reset_index(drop=True)
    val_df = df[df['study_id'].isin(val_studies)].reset_index(drop=True)
    test_df = df[df['study_id'].isin(test_studies)].reset_index(drop=True)
    assert train_studies.isdisjoint(val_studies)
    assert train_studies.isdisjoint(test_studies)
    assert val_studies.isdisjoint(test_studies)
    return train_df, val_df, test_df


def crop_lumbar_region(img, top_frac=0.20, bot_frac=0.80):
    H = img.shape[0]
    return img[int(H * top_frac):int(H * bot_frac), :, :]


class LumbarModel(nn.Module):
    """EfficientNet-B4 backbone + dropout + linear head (75 logits -> [B,25,3])."""
    def __init__(self, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(1792, 75))

    def forward(self, x):
        return self.head(self.backbone(x)).view(-1, 25, 3)


def find_jpeg(study_id, roots):
    for r in roots:
        p = Path(r) / f'{int(study_id)}.jpg'
        if p.exists():
            return p
    return None


def load_image(path, crop=True, size=512):
    """Returns (normalized tensor [3,H,W], uint8 RGB array [H,W,3]) exactly as in training."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if crop:
        img_rgb = crop_lumbar_region(img_rgb)
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    t = transforms.Normalize(mean=CFG['mean'], std=CFG['std'])(t)
    return t, img_resized


class EvalDataset(Dataset):
    def __init__(self, df, roots):
        self.rows, self.paths = [], []
        for _, row in df.iterrows():
            p = find_jpeg(row['study_id'], roots)
            if p is None:
                warnings.warn(f"JPEG not found for study_id={row['study_id']}")
                continue
            self.rows.append(row)
            self.paths.append(p)
        if not self.paths:
            raise RuntimeError('No JPEGs found — check CFG["jpeg_roots"].')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        t, _ = load_image(self.paths[i], CFG['crop_lumbar'], CFG['image_size'])
        labels = self.rows[i][LABEL_COLS].values.astype(np.float32)
        labels = np.nan_to_num(labels, nan=-1.0).astype(np.int64)
        return t, torch.tensor(labels), int(self.rows[i]['study_id'])


def find_checkpoint(explicit=None):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f'Checkpoint not found: {p}')
        return p
    known = Path('/kaggle/input/models/drlochanshrestha/lumbarnet-25/pytorch/default/2') / CFG['ckpt_name']
    if known.exists():
        return known
    # Search small locations only (rglob over /kaggle/input would walk the
    # competition's DICOM files and can take minutes).
    for base in [Path('/kaggle/working'), Path('/kaggle/input/models'), Path('.')]:
        if base.exists():
            hits = sorted(base.rglob(CFG['ckpt_name']))
            if hits:
                return hits[0]
    sys.exit(f"Could not find {CFG['ckpt_name']} under /kaggle/working, /kaggle/input or '.'. "
             "Pass --ckpt explicitly (e.g. from your Kaggle Model 'lumbarnet-25').")


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model = LumbarModel(CFG['drop_rate']).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    epoch = ckpt.get('epoch')
    vk = ckpt.get('val_kappa')
    return model, (epoch + 1 if epoch is not None else None), vk


def load_splits():
    df = pd.read_csv(CFG['train_csv'])
    for col in LABEL_COLS:
        if col in df.columns:
            df[col] = df[col].map(LABEL_MAP)
    df['study_id'] = df['study_id'].astype(int)
    return make_patient_split(df, CFG['val_frac'], CFG['test_frac'], CFG['seed'])


@torch.no_grad()
def predict(model, df, device):
    """Full-precision inference (matches validate() in train.py)."""
    ds = EvalDataset(df, CFG['jpeg_roots'])
    dl = DataLoader(ds, batch_size=CFG['batch_size'], shuffle=False,
                    num_workers=CFG['num_workers'], pin_memory=device.type == 'cuda')
    probs, labels, ids = [], [], []
    for x, y, sid in tqdm(dl, desc='Evaluating'):
        logits = model(x.to(device))
        probs.append(F.softmax(logits.float(), dim=2).cpu().numpy())
        labels.append(y.numpy())
        ids.extend(sid.tolist())
    return np.concatenate(probs), np.concatenate(labels), np.array(ids)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def to_long(probs, labels, ids):
    preds = probs.argmax(axis=2)
    rec = []
    for i, sid in enumerate(ids):
        for j, col in enumerate(LABEL_COLS):
            if labels[i, j] < 0:
                continue
            rec.append(dict(study_id=int(sid), condition_level=col,
                            is_laterality=col.startswith(('left_', 'right_')),
                            true_label=int(labels[i, j]), pred_label=int(preds[i, j]),
                            prob_normal_mild=float(probs[i, j, 0]),
                            prob_moderate=float(probs[i, j, 1]),
                            prob_severe=float(probs[i, j, 2])))
    return pd.DataFrame(rec)


def per_output(g):
    y, p, ps = g.true_label.values, g.pred_label.values, g.prob_severe.values
    k = (cohen_kappa_score(y, p, labels=[0, 1, 2], weights='quadratic')
         if len(np.unique(y)) >= 2 else np.nan)
    ys, pp = (y == 2), (p == 2)
    auc = roc_auc_score(ys, ps) if 0 < ys.sum() < len(ys) else np.nan
    tp, fn = (ys & pp).sum(), (ys & ~pp).sum()
    fp, tn = (~ys & pp).sum(), (~ys & ~pp).sum()
    return dict(kappa=k, auc=auc,
                sens=tp / (tp + fn) if tp + fn else np.nan,
                spec=tn / (tn + fp) if tn + fp else np.nan,
                ppv=tp / (tp + fp) if tp + fp else np.nan,
                n_severe=int(ys.sum()), n=len(y))


def summarize(long_df):
    rows = {col: per_output(long_df[long_df.condition_level == col]) for col in LABEL_COLS}
    per = pd.DataFrame(rows).T.loc[LABEL_COLS]
    ys, pp = long_df.true_label.values == 2, long_df.pred_label.values == 2
    tp, fn = (ys & pp).sum(), (ys & ~pp).sum()
    fp, tn = (~ys & pp).sum(), (~ys & ~pp).sum()
    sens, spec, ppv = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp)
    head = dict(macro_qwk=float(np.nanmean(per.kappa.astype(float))),
                n_qwk=int(per.kappa.notna().sum()),
                macro_auc=float(np.nanmean(per.auc.astype(float))),
                n_auc=int(per.auc.notna().sum()),
                sens=sens, spec=spec, ppv=ppv, f1=2 * ppv * sens / (ppv + sens),
                n_labels=len(long_df), n_severe=int(ys.sum()))
    return per, head


def bootstrap(long_df, n_macro, n_per, seed=42):
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in long_df.groupby('study_id')}
    studies = np.array(list(groups))
    macro = []
    for _ in tqdm(range(n_macro), desc='Bootstrap (macro/pooled)'):
        samp = pd.concat([groups[s] for s in rng.choice(studies, len(studies))])
        _, h = summarize(samp)
        macro.append([h['macro_qwk'], h['macro_auc'], h['sens'], h['spec'], h['ppv'], h['f1']])
    macro = np.array(macro)
    ci = {k: np.nanpercentile(macro[:, i], [2.5, 97.5])
          for i, k in enumerate(['macro_qwk', 'macro_auc', 'sens', 'spec', 'ppv', 'f1'])}
    per_ci = {}
    if n_per:
        for col in tqdm(LABEL_COLS, desc='Bootstrap (per output)'):
            g = long_df[long_df.condition_level == col]
            sg = {s: x for s, x in g.groupby('study_id')}
            ss = np.array(list(sg))
            ks, aus = [], []
            for _ in range(n_per):
                b = pd.concat([sg[s] for s in rng.choice(ss, len(ss))])
                r = per_output(b)
                ks.append(r['kappa']); aus.append(r['auc'])
            per_ci[col] = (np.nanpercentile(ks, [2.5, 97.5]),
                           np.nanpercentile(aus, [2.5, 97.5]) if np.isfinite(aus).any() else (np.nan, np.nan))
    return ci, per_ci


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--split', choices=['test', 'val'], default='test')
    ap.add_argument('--bootstrap', type=int, default=0, help='resamples for macro/pooled CIs (paper: 2000)')
    ap.add_argument('--bootstrap_per_output', type=int, default=0, help='resamples per output (paper: 1000)')
    ap.add_argument('--compare_csv',
                    default='/kaggle/input/models/drlochanshrestha/lumbarnet-25/pytorch/default/2/test_set_predictions.csv',
                    help='original test_set_predictions.csv to compare against (skipped if missing)')
    ap.add_argument('--out_dir', default='/kaggle/working/eval_outputs')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args, _ = ap.parse_known_args()

    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    checks = []

    train_df, val_df, test_df = load_splits()
    sizes = (len(train_df), len(val_df), len(test_df))
    sev = tuple(int((d[LABEL_COLS] == 2).sum().sum()) for d in (train_df, val_df, test_df))
    print(f'Split sizes (train/val/test): {sizes}   Severe labels: {sev}')
    checks.append(('Split sizes', sizes == EXPECTED['split_sizes']))
    checks.append(('Severe counts per split', sev == EXPECTED['severe_counts']))

    ckpt_path = find_checkpoint(args.ckpt)
    model, epoch, vk = load_model(ckpt_path, device)
    print(f'Loaded {ckpt_path}  (epoch {epoch}, stored val kappa {vk})')
    checks.append(('Checkpoint epoch = 35', epoch == EXPECTED['ckpt_epoch']))
    checks.append(('Checkpoint val kappa = 0.2738',
                   vk is not None and abs(float(vk) - EXPECTED['ckpt_val_kappa']) < 5e-4))

    df = test_df if args.split == 'test' else val_df
    probs, labels, ids = predict(model, df, device)
    long_df = to_long(probs, labels, ids)
    per, head = summarize(long_df)

    pred_csv = out / f'{args.split}_set_predictions.csv'
    long_df.to_csv(pred_csv, index=False)
    per.to_csv(out / f'{args.split}_per_output_metrics.csv')

    print(f"\n{'Condition-level':42s} {'QWK':>7} {'AUC':>7} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'nSev/n':>9}")
    print('-' * 90)
    for col, r in per.iterrows():
        f = lambda v: '   —  ' if pd.isna(v) else f'{float(v):6.3f}'
        print(f"{col:42s} {f(r.kappa):>7} {f(r.auc):>7} {f(r.sens)} {f(r.spec)} {f(r.ppv)} "
              f"{int(r.n_severe):>4}/{int(r.n)}")
    print(f"\nMacro QWK {head['macro_qwk']:.4f} (n={head['n_qwk']})   "
          f"Macro AUC {head['macro_auc']:.4f} (n={head['n_auc']})")
    print(f"Pooled Severe: sens {head['sens']:.3f}  spec {head['spec']:.3f}  "
          f"PPV {head['ppv']:.3f}  F1 {head['f1']:.3f}   "
          f"({head['n_severe']} Severe / {head['n_labels']} labels)")

    ref = Path(args.compare_csv) if args.compare_csv else None
    if args.split == 'test' and ref is not None and ref.exists():
        old = pd.read_csv(ref)
        m = long_df.merge(old, on=['study_id', 'condition_level'], suffixes=('', '_orig'))
        same_pred = (m.pred_label == m.pred_label_orig).mean()
        max_dp = (m.prob_severe - m.prob_severe_orig).abs().max()
        print(f'\nCompared with original run ({ref}): {len(m)}/{len(old)} rows matched, '
              f'identical predicted grade {same_pred:.2%}, max |ΔP(Severe)| {max_dp:.1e}')
        checks.append(('Predictions identical to original training-run CSV',
                       len(m) == len(old) == len(long_df) and same_pred == 1.0))

    if args.split == 'val':
        checks.append(('Val macro QWK reproduces 0.2738', abs(head['macro_qwk'] - EXPECTED['ckpt_val_kappa']) < 5e-4))
    else:
        e = EXPECTED['test']
        checks += [
            ('Test macro QWK = 0.3255', abs(head['macro_qwk'] - e['macro_qwk']) < 5e-4),
            ('Test macro AUC = 0.8211 over 23 outputs',
             abs(head['macro_auc'] - e['macro_auc']) < 5e-4 and head['n_auc'] == e['n_auc']),
            ('Pooled sens/spec/PPV/F1 = 0.653/0.862/0.260/0.372',
             all(abs(head[k] - e[k]) < 5e-4 for k in ('sens', 'spec', 'ppv', 'f1'))),
            ('7,285 annotated test labels', head['n_labels'] == e['n_labels']),
        ]

    if args.bootstrap:
        ci, per_ci = bootstrap(long_df, args.bootstrap, args.bootstrap_per_output)
        print('\n95% CIs (patient-level bootstrap; Monte Carlo variation of ±0.003 vs the paper is expected):')
        for k, (lo, hi) in ci.items():
            print(f'  {k:10s} {lo:.3f}–{hi:.3f}')
        if per_ci:
            rows = [dict(condition_level=c, kappa_lo=k[0], kappa_hi=k[1], auc_lo=a[0], auc_hi=a[1])
                    for c, (k, a) in per_ci.items()]
            pd.DataFrame(rows).to_csv(out / f'{args.split}_per_output_CIs.csv', index=False)

    print('\n' + '=' * 60 + '\nREPRODUCIBILITY CHECKS\n' + '=' * 60)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'MISMATCH'}] {name}")
    print(f'\nOutputs written to {out}')
    if not all(ok for _, ok in checks):
        print('\nOne or more checks failed — results do NOT reproduce the manuscript. '
              'Check the checkpoint file and train.csv path.')


if __name__ == '__main__':
    main()
