"""Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explainability of CNN predictions.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
    IEEE International Conference on Computer Vision (ICCV), 2017.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Computes Grad-CAM spatial attention heatmaps from a target convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        self.activations = out.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Detach forward and backward hooks cleanly."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """Compute the Grad-CAM heatmap for an input tensor of shape (1, C, H, W).

        Returns:
            heatmap: 2D float32 array in [0, 1] resized to (H, W)
            pred_class: predicted class index
            confidence: top-1 softmax probability
            probs: 1D numpy array of all class probabilities
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)
        probs_tensor = F.softmax(logits, dim=1)
        probs = probs_tensor.detach().cpu().numpy()[0]

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        class_idx = pred_class if target_class is None else int(target_class)

        score = logits[0, class_idx]
        score.backward()

        assert self.activations is not None and self.gradients is not None, "Grad-CAM hooks did not capture tensors."

        # Global average pooling over spatial gradient dimensions -> channel weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, K, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input spatial resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap.astype(np.float32), pred_class, confidence, probs


def overlay_heatmap(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Overlay a normalized [0, 1] Grad-CAM heatmap onto an RGB uint8 image using the specified colormap."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if rgb_image.shape[:2] != heatmap.shape[:2]:
        heatmap_pil = Image.fromarray(np.uint8(255 * np.clip(heatmap, 0.0, 1.0))).resize(
            (rgb_image.shape[1], rgb_image.shape[0]),
            resample=Image.BILINEAR,
        )
        heatmap = np.asarray(heatmap_pil, dtype=np.float32) / 255.0

    cmap_key = colormap_name if colormap_name in {"jet", "inferno", "viridis", "plasma", "magma", "turbo"} else "jet"
    colormap = matplotlib.colormaps[cmap_key] if hasattr(matplotlib, "colormaps") else cm.get_cmap(cmap_key)
    colored_heatmap = colormap(heatmap)[:, :, :3]  # Drop alpha channel -> (H, W, 3) float in [0, 1]
    base_float = rgb_image.astype(np.float32) / 255.0

    blended = (1.0 - alpha) * base_float + alpha * colored_heatmap
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)
