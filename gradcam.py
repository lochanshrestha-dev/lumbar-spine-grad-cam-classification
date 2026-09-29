#!/usr/bin/env python3
"""
gradcam.py — Grad-CAM for the corrected LumbarNet-25 model (PLOS ONE revision).

Requires evaluate.py in the same folder (shared split / model / preprocessing
code, so Grad-CAM uses exactly the same pipeline as training and evaluation).

What this script does
---------------------
1. Rebuilds the three-way split (1,383 / 296 / 296; seed 42) and loads
   lumbar_best_v3_nohflip.pth.
2. SAFETY CHECK: re-scores the validation split and requires macro QWK to
   reproduce the training log value (0.2738). If it doesn't, the script stops,
   so heatmaps can never come from the wrong checkpoint or preprocessing.
3. Selects the N validation-split studies with the most Severe labels
   (default 10). The held-out test split is not touched.
4. For every study and all 25 outputs, computes Grad-CAM for the SEVERE class
   logit (class index 2) at backbone.bn2 (final layer before global pooling;
   primary analysis), irrespective of the predicted class. Other layers via
   --target_layer (sensitivity analyses: blocks.6, conv_head).
5. Normalizes the 25 maps of each study on a SHARED scale (divided by the
   study's maximum), so activation magnitude is comparable across outputs.
6. Writes:
     gradcam_outputs/<study>_gradcam.png   review grid per study
     gradcam_outputs/Fig2.tif              publication figure (top study; PLOS TIFF, 300 dpi;
                                           heatmaps on the grayscale sagittal T2 channel)
     gradcam_outputs/Fig2_preview.png
     gradcam_outputs/S1_channel_view.png   heatmap shown on each input channel separately
     gradcam_outputs/cam_summary.csv       per study x output: grades, P(Severe), peak/mean activation, centroid
     gradcam_outputs/run_summary.txt       everything printed below

Differences from the earlier gradcam.py (pre-revision):
  - model head is Dropout+Linear (the old Linear-only head cannot load the new checkpoint)
  - lumbar crop applied, as in training
  - explains the Severe class (old script explained the predicted class)
  - shared per-study normalization (old script normalized each panel separately)

Usage (Kaggle)
--------------
    !python gradcam.py
    !python gradcam.py --ckpt /path/to/lumbar_best_v3_nohflip.pth --n_studies 10
    !python gradcam.py --target_layer blocks.6 --out_dir /kaggle/working/gradcam_blocks6
"""

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (CFG, CONDITIONS, LEVELS, LABEL_COLS, EXPECTED,  # noqa: E402
                      find_checkpoint, find_jpeg, load_image, load_model, resolve_jpeg_roots, resolve_train_csv,
                      load_splits, predict, summarize, to_long)

COND_SHORT = {
    'spinal_canal_stenosis': 'SCS',
    'left_neural_foraminal_narrowing': 'Left NFN',
    'right_neural_foraminal_narrowing': 'Right NFN',
    'left_subarticular_stenosis': 'Left SS',
    'right_subarticular_stenosis': 'Right SS',
}
LEVEL_SHORT = {'l1_l2': 'L1–L2', 'l2_l3': 'L2–L3', 'l3_l4': 'L3–L4', 'l4_l5': 'L4–L5', 'l5_s1': 'L5–S1'}
GRADE_SHORT = {0: 'N/M', 1: 'Mod', 2: 'Sev', -1: '—'}
SEVERE = 2


class GradCAM:
    """Grad-CAM with one forward pass and one backward pass per output."""
    def __init__(self, model, layer):
        self.model, self.act = model, None
        self.h = layer.register_forward_hook(lambda m, i, o: setattr(self, 'act', o))

    def all_outputs(self, x, class_idx=SEVERE):
        """Returns raw CAMs [25, H, W] (not normalized) and softmax probs [25, 3]."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)                     # [1, 25, 3]
            act = self.act                             # [1, C, h, w]
            cams = []
            for k in range(25):
                g = torch.autograd.grad(logits[0, k, class_idx], act, retain_graph=True)[0]
                w = g.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((w * act).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
                cams.append(cam[0, 0].detach().cpu().numpy())
        probs = F.softmax(logits.detach().float(), dim=2)[0].cpu().numpy()
        return np.stack(cams), probs

    def remove(self):
        self.h.remove()


def overlay(bg_rgb, cam01, alpha=0.45):
    heat = plt.get_cmap('jet')(np.clip(cam01, 0, 1))[..., :3]
    base = bg_rgb.astype(np.float32) / 255.0
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def centroid(cam):
    s = cam.sum()
    if s <= 0:
        return np.nan, np.nan
    ys, xs = np.indices(cam.shape)
    return float((ys * cam).sum() / s / cam.shape[0]), float((xs * cam).sum() / s / cam.shape[1])


def review_grid(study_id, img, cams01, truth, preds, psev, n_sev, path):
    fig, axes = plt.subplots(5, 5, figsize=(16, 17))
    for i, c in enumerate(CONDITIONS):
        for j, lv in enumerate(LEVELS):
            k = LABEL_COLS.index(f'{c}_{lv}')
            ax = axes[i, j]
            ax.imshow(overlay(img, cams01[k]))
            ax.set_title(f'{COND_SHORT[c]} {LEVEL_SHORT[lv]}\n'
                         f'pred {GRADE_SHORT[preds[k]]} | true {GRADE_SHORT[truth[k]]} | '
                         f'P(sev) {psev[k]:.2f}', fontsize=8)
            ax.axis('off')
    fig.suptitle(f'Study {study_id} — {n_sev} Severe labels — Grad-CAM for Severe class '
                 f'(shared scale within study)', fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def publication_figure(img, cams01, truth, preds, out_tif, out_png):
    """PLOS ONE: TIFF, RGB, 300 dpi, max 7.5 in wide."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
    fig = plt.figure(figsize=(7.5, 8.4))
    # Background: sagittal T2 channel only (grayscale). The fused RGB composite
    # mixes sagittal and axial series, whose pixels do not correspond spatially.
    sag_t2 = np.repeat(img[..., 0:1], 3, axis=2)
    gs = fig.add_gridspec(6, 6, width_ratios=[0.55, 1, 1, 1, 1, 1],
                          height_ratios=[0.18, 1, 1, 1, 1, 1], wspace=0.04, hspace=0.18)
    for j, lv in enumerate(LEVELS):
        ax = fig.add_subplot(gs[0, j + 1]); ax.axis('off')
        ax.text(0.5, 0.2, LEVEL_SHORT[lv], ha='center', va='center', fontsize=8, fontweight='bold')
    for i, c in enumerate(CONDITIONS):
        ax = fig.add_subplot(gs[i + 1, 0]); ax.axis('off')
        ax.text(0.95, 0.5, COND_SHORT[c], ha='right', va='center', fontsize=8, fontweight='bold')
        for j, lv in enumerate(LEVELS):
            k = LABEL_COLS.index(f'{c}_{lv}')
            ax = fig.add_subplot(gs[i + 1, j + 1])
            ax.imshow(overlay(sag_t2, cams01[k]))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f'Pred {GRADE_SHORT[preds[k]]} · True {GRADE_SHORT[truth[k]]}', fontsize=6, labelpad=1)
    cax = fig.add_axes([0.30, 0.052, 0.5, 0.012])
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('Grad-CAM activation for the Severe class (normalized to the study maximum)', fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.095)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, facecolor='white')
    buf.seek(0)
    from PIL import Image
    Image.open(buf).convert('RGB').save(out_tif, compression='tiff_lzw', dpi=(300, 300))  # PLOS: RGB, no alpha
    fig.savefig(out_png, dpi=120, facecolor='white')
    plt.close(fig)


def channel_view(study_id, img, cams01, truth, preds, path, max_rows=3):
    names = ['Ch0: sagittal T2', 'Ch1: sagittal T1', 'Ch2: axial T2']
    idx = [k for k in np.argsort([-c.max() for c in cams01]) if truth[k] == SEVERE][:max_rows] \
        or list(np.argsort([-c.max() for c in cams01])[:max_rows])
    fig, axes = plt.subplots(len(idx), 3, figsize=(10, 3.4 * len(idx)), squeeze=False)
    for r, k in enumerate(idx):
        for ch in range(3):
            gray = np.repeat(img[..., ch:ch + 1], 3, axis=2)
            ax = axes[r, ch]
            ax.imshow(overlay(gray, cams01[k]))
            ax.axis('off')
            c, lv = LABEL_COLS[k].rsplit('_', 2)[0], '_'.join(LABEL_COLS[k].rsplit('_', 2)[1:])
            ax.set_title(f'{names[ch]}\n{COND_SHORT[c]} {LEVEL_SHORT[lv]} '
                         f'(pred {GRADE_SHORT[preds[k]]}, true {GRADE_SHORT[truth[k]]})', fontsize=8)
    fig.suptitle(f'Study {study_id}: the same Grad-CAM map shown on each fused input channel', fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(args):
    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    CFG['train_csv'] = resolve_train_csv(args.train_csv)
    CFG['jpeg_roots'] = resolve_jpeg_roots(args.jpeg_dir)

    train_df, val_df, test_df = load_splits()
    sizes = (len(train_df), len(val_df), len(test_df))
    print(f'Split sizes (train/val/test): {sizes}')
    if sizes != EXPECTED['split_sizes']:
        sys.exit('STOP: split does not match training (expected 1383/296/296).')

    ckpt = find_checkpoint(args.ckpt)
    model, epoch, vk = load_model(ckpt, device)
    print(f'Loaded {ckpt} (epoch {epoch}, stored val kappa {vk})')

    if not args.skip_val_check:
        probs, labels, ids = predict(model, val_df, device)
        _, head = summarize(to_long(probs, labels, ids))
        print(f"Validation macro QWK recomputed: {head['macro_qwk']:.4f} (training log: 0.2738)")
        if abs(head['macro_qwk'] - EXPECTED['ckpt_val_kappa']) > 5e-4:
            sys.exit('STOP: validation QWK does not reproduce — wrong checkpoint or preprocessing. '
                     'No heatmaps generated.')
        print('PASS: checkpoint and preprocessing reproduce the training run.')

    v = val_df.copy()
    v['severe_count'] = (v[LABEL_COLS] == 2).sum(axis=1)
    if args.study_ids:
        sel = v[v.study_id.isin([int(s) for s in args.study_ids])]
    else:
        sel = v.nlargest(args.n_studies, 'severe_count')
    print(f'\nSelected {len(sel)} validation-split studies (Severe labels per study: '
          f"{sel.severe_count.min()}–{sel.severe_count.max()}):")
    print(sel[['study_id', 'severe_count']].to_string(index=False))

    layer = model.backbone
    for part in args.target_layer.split('.'):
        layer = layer[int(part)] if part.isdigit() else getattr(layer, part)
    print(f'Grad-CAM target layer: backbone.{args.target_layer} ({type(layer).__name__})')
    gc = GradCAM(model, layer)
    rows, per_study = [], {}
    for _, row in sel.iterrows():
        sid = int(row.study_id)
        path = find_jpeg(sid, CFG['jpeg_roots'])
        if path is None:
            print(f'  missing JPEG for {sid}, skipped'); continue
        t, img = load_image(path, CFG['crop_lumbar'], CFG['image_size'])
        raw, probs = gc.all_outputs(t.unsqueeze(0).to(device))
        vmax = raw.max()
        cams01 = raw / vmax if vmax > 0 else raw
        truth = np.nan_to_num(row[LABEL_COLS].values.astype(float), nan=-1).astype(int)
        preds = probs.argmax(1)
        review_grid(sid, img, cams01, truth, preds, probs[:, 2], int(row.severe_count),
                    out / f'{sid}_gradcam.png')
        per_study[sid] = (img, cams01, truth, preds)
        for k, col in enumerate(LABEL_COLS):
            cy, cx = centroid(raw[k])
            c, lv = col.rsplit('_', 2)[0], '_'.join(col.rsplit('_', 2)[1:])
            rows.append(dict(study_id=sid, severe_labels=int(row.severe_count), condition=c, level=lv,
                             true=int(truth[k]), pred=int(preds[k]), prob_severe=float(probs[k, 2]),
                             peak_raw=float(raw[k].max()), mean_raw=float(raw[k].mean()),
                             peak_rel_study=float(cams01[k].max()), centroid_y=cy, centroid_x=cx))
        print(f'  done {sid}')
    gc.remove()

    cs = pd.DataFrame(rows)
    cs.to_csv(out / 'cam_summary.csv', index=False)

    top = int(sel.iloc[0].study_id)
    img, cams01, truth, preds = per_study[top]
    publication_figure(img, cams01, truth, preds, out / 'Fig2.tif', out / 'Fig2_preview.png')
    channel_view(top, img, cams01, truth, preds, out / 'S1_channel_view.png')

    # ---- Quantitative summaries for the manuscript text --------------------
    print('\n' + '=' * 70 + '\nSUMMARY FOR MANUSCRIPT\n' + '=' * 70)
    print(f'Figure 2 study: {top} ({int(sel.iloc[0].severe_count)} Severe labels)')
    grp = cs.assign(true_grade=cs.true.map(GRADE_SHORT)).groupby('true_grade')
    print('\nMean relative peak activation by TRUE grade (shared within-study scale, 0–1):')
    print(grp.peak_rel_study.agg(['mean', 'median', 'count']).round(3).to_string())
    grp = cs.assign(pred_grade=cs.pred.map(GRADE_SHORT)).groupby('pred_grade')
    print('\nMean relative peak activation by PREDICTED grade:')
    print(grp.peak_rel_study.agg(['mean', 'median', 'count']).round(3).to_string())
    print('\nMean vertical centroid by disc level (0 = top of cropped image, 1 = bottom):')
    lv = cs.groupby('level').centroid_y.agg(['mean', 'std']).reindex(LEVELS).round(3)
    print(lv.to_string())
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(cs.level.map(LEVELS.index), cs.centroid_y, nan_policy='omit')
        print(f'Spearman rho (level order vs vertical centroid): {rho:.3f} (p = {p:.2g}, '
              f'n = {cs.centroid_y.notna().sum()} maps)')
    except Exception as e:  # scipy missing
        print(f'Spearman not computed: {e}')
    print('Note: the axial channel carries no craniocaudal information, so vertical centroids '
          'mainly reflect the sagittal channels.')
    print(f'\nFiles written to {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--train_csv', default=None, help='competition train.csv (auto-detected if omitted)')
    ap.add_argument('--jpeg_dir', default=None, help='folder containing <study_id>.jpg (auto-detected if omitted)')
    ap.add_argument('--n_studies', type=int, default=10)
    ap.add_argument('--study_ids', nargs='*', default=None, help='override automatic selection (validation split)')
    ap.add_argument('--target_layer', default='bn2',
                    help="backbone sub-module, e.g. bn2 (primary), conv_head, blocks.6")
    ap.add_argument('--out_dir', default='/kaggle/working/gradcam_bn2')
    ap.add_argument('--skip_val_check', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args, _ = ap.parse_known_args()

    buf = io.StringIO()

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s); buf.write(s)

        def flush(self):
            sys.__stdout__.flush()

    with redirect_stdout(Tee()):
        try:
            run(args)
        finally:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.out_dir) / 'run_summary.txt').write_text(buf.getvalue())


if __name__ == '__main__':
    main()
