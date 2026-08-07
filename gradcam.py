#!/usr/bin/env python3
"""
Grad-CAM visualization for lumbar spine MRI classification model.
Generates heatmaps for 25 condition-level pairs.
"""

import argparse
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torchvision import transforms
from tqdm import tqdm

# Constants (as in the original partial)
CONDITIONS = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis'
]
LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
LABEL_COLS = [f'{c}_{l}' for c in CONDITIONS for l in LEVELS]  # 25 total
LABEL_INDEX = {label: idx for idx, label in enumerate(LABEL_COLS)}
SEVERITY_NAMES = ['Normal/Mild', 'Moderate', 'Severe']


def get_output_idx(condition: str, level: str) -> int:
    """Return index (0-24) for given condition and level."""
    return LABEL_INDEX[f'{condition}_{level}']


# ----------------------------------------------------------------------
# GradCAM class (partial already written, fully implemented here)
# ----------------------------------------------------------------------
class GradCAM:
    """Grad-CAM heatmap generator for a target layer."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_hook = None
        self._backward_hook = None
        self._register_hooks()
    
    def _save_features(self, module, input, output):
        self.activations = output.detach()
    
    def _save_grads(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def _register_hooks(self):
        self._forward_hook = self.target_layer.register_forward_hook(self._save_features)
        # Use register_full_backward_hook (PyTorch 2.x compatible)
        self._backward_hook = self.target_layer.register_full_backward_hook(self._save_grads)
    
    def remove(self):
        if self._forward_hook:
            self._forward_hook.remove()
        if self._backward_hook:
            self._backward_hook.remove()
    
    def generate(self, img_tensor: torch.Tensor, output_idx: int, class_idx: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            img_tensor: [1, 3, H, W] normalized tensor on correct device
            output_idx: which of the 25 outputs (0-24)
            class_idx: which of the 3 severity classes (0,1,2)
        
        Returns:
            float32 numpy array [H, W] heatmap (512x512)
        """
        self.model.zero_grad()
        logits = self.model(img_tensor)  # [1, 25, 3]
        target_score = logits[0, output_idx, class_idx]
        target_score.backward()
        
        # Global average pooling of gradients
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])  # [C]
        # Weight activations
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_grads[i]
        # Average over channels, ReLU
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        # Normalize to [0,1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        # Resize to input size (512x512)
        heatmap = heatmap.cpu().numpy()
        heatmap = cv2.resize(heatmap, (512, 512))
        return heatmap.astype(np.float32)


# ----------------------------------------------------------------------
# Overlay function
# ----------------------------------------------------------------------
def overlay_heatmap(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap (JET colormap) on RGB image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay.astype(np.uint8)


# ----------------------------------------------------------------------
# save_gradcam_grid - COMPLETED with severity names
# ----------------------------------------------------------------------
def save_gradcam_grid(
    study_id: str,
    img_rgb: np.ndarray,                     # [512,512,3] uint8
    cams: dict[str, tuple[np.ndarray, int]], # {label: (cam, pred_class)}
    out_dir: Path,
) -> None:
    """Create 5x5 grid of Grad-CAM overlays with severity labels and save PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    fig.suptitle(f"Grad-CAM for {study_id}", fontsize=16)
    
    for i, condition in enumerate(CONDITIONS):
        for j, level in enumerate(LEVELS):
            label = f"{condition}_{level}"
            ax = axes[i, j]
            data = cams.get(label)
            if data is not None:
                cam, pred_class = data
                severity_name = SEVERITY_NAMES[pred_class]
                overlay = overlay_heatmap(img_rgb, cam, alpha=0.4)
                ax.imshow(overlay)
                ax.set_title(f"{condition}\n{level}\n{severity_name}", fontsize=8)
            else:
                ax.imshow(img_rgb)
                ax.set_title(f"{condition}\n{level}\n(no cam)", fontsize=8)
            ax.axis('off')
    
    # Add original image as inset in top-left subplot
    ax0 = axes[0, 0]
    inset = inset_axes(ax0, width="30%", height="30%", loc='lower left')
    inset.imshow(img_rgb)
    inset.axis('off')
    
    plt.tight_layout()
    out_path = out_dir / f"{study_id}_gradcam.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_jpeg_for_gradcam(jpeg_path: Path, image_size: int = 512):
    """Load JPEG, resize, normalize, return tensor and RGB array."""
    img_bgr = cv2.imread(str(jpeg_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read {jpeg_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor).unsqueeze(0)
    return img_tensor, img_resized


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


def load_model(ckpt_path: Path, device: torch.device):
    """Load checkpoint, handle DataParallel, return model in eval mode."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = LumbarModel().to(device)
    state_dict = checkpoint['model_state']
    # Strip 'module.' prefix if present
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
# CLI and main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Grad-CAM for lumbar spine MRI')
    parser.add_argument('--ckpt', type=Path, required=True, help='Checkpoint path')
    parser.add_argument('--jpeg_dir', type=Path, required=True, help='Directory with JPEGs')
    parser.add_argument('--study_ids', nargs='+', help='Specific study IDs (default: all)')
    parser.add_argument('--out_dir', type=Path, default=Path('/kaggle/working/gradcam_outputs'))
    parser.add_argument('--target_layer', type=str, default='blocks.6',
                        help='Target layer name (e.g., blocks.6)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    
    # Extract target layer from model backbone
    if args.target_layer.startswith('blocks.'):
        block_idx = int(args.target_layer.split('.')[1])
        target_layer = model.backbone.blocks[block_idx]
    else:
        target_layer = model.backbone.blocks[-1]
    
    gradcam = GradCAM(model, target_layer)
    
    # Determine study IDs
    if args.study_ids:
        study_ids = args.study_ids
    else:
        study_ids = [p.stem for p in args.jpeg_dir.glob('*.jpg')]
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    for study_id in tqdm(study_ids, desc="Generating Grad-CAM"):
        jpeg_path = args.jpeg_dir / f"{study_id}.jpg"
        if not jpeg_path.exists():
            warnings.warn(f"JPEG not found: {jpeg_path}")
            continue
        try:
            img_tensor, img_rgb = load_jpeg_for_gradcam(jpeg_path, 512)
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                logits = model(img_tensor)  # [1, 25, 3]
            cams = {}
            for output_idx in range(25):
                pred_class = torch.argmax(logits[0, output_idx]).item()
                cam = gradcam.generate(img_tensor, output_idx, pred_class)
                key = LABEL_COLS[output_idx]
                cams[key] = (cam, pred_class)   # store tuple for severity display
            save_gradcam_grid(study_id, img_rgb, cams, args.out_dir)
            print(f"Saved: {study_id}")
        except Exception as e:
            warnings.warn(f"Failed for {study_id}: {e}")
    
    gradcam.remove()


if __name__ == '__main__':
    main()