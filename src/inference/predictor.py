"""Single-image inference engine with ecological & environmental advisory insights,
counterfactual Grad-CAM, Dihedral Test-Time Augmentation (TTA), and uncertainty quantification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.inference.gradcam import GradCAM, overlay_heatmap
from src.models.classifier import EuroSATClassifier
from src.utils import get_device, load_config, setup_logger

logger = setup_logger("eurosat.predictor")

ENVIRONMENTAL_INSIGHTS: Dict[str, Dict[str, Any]] = {
    "AnnualCrop": {
        "category": "Agricultural Land-Use",
        "ecological_indicator": "Seasonal cropland requiring irrigation & soil nutrient monitoring.",
        "actionable_advisory": "Monitor NDVI seasonal harvest cycles and check for fertilizer runoff near waterways.",
        "carbon_density_t_ha": 12.0,
        "impervious_fraction": 0.04,
        "biodiversity_index": 0.35,
    },
    "Forest": {
        "category": "High Carbon-Sink Ecosystem",
        "ecological_indicator": "Dense tree canopy crucial for carbon sequestration and biodiversity.",
        "actionable_advisory": "Priority conservation zone: trigger alert if consecutive passes show canopy fragmentation.",
        "carbon_density_t_ha": 145.0,
        "impervious_fraction": 0.01,
        "biodiversity_index": 0.95,
    },
    "HerbaceousVegetation": {
        "category": "Natural Shrubland & Meadow",
        "ecological_indicator": "Transitional vegetative cover and pollinator habitat.",
        "actionable_advisory": "Monitor dry-season wildfire fuel load and ecological succession patterns.",
        "carbon_density_t_ha": 34.0,
        "impervious_fraction": 0.02,
        "biodiversity_index": 0.78,
    },
    "Highway": {
        "category": "Transportation Infrastructure",
        "ecological_indicator": "Impervious linear corridor causing habitat fragmentation.",
        "actionable_advisory": "Track roadside buffer encroachment and urban heat-island contribution.",
        "carbon_density_t_ha": 3.0,
        "impervious_fraction": 0.78,
        "biodiversity_index": 0.10,
    },
    "Industrial": {
        "category": "High-Intensity Built Environment",
        "ecological_indicator": "Large impervious commercial/industrial roofs with high thermal footprint.",
        "actionable_advisory": "Assess rooftop solar photovoltaic (PV) potential and industrial emission buffers.",
        "carbon_density_t_ha": 2.0,
        "impervious_fraction": 0.88,
        "biodiversity_index": 0.06,
    },
    "Pasture": {
        "category": "Managed Grazing Land",
        "ecological_indicator": "Low-intensity agricultural grassland used for livestock.",
        "actionable_advisory": "Evaluate rotational grazing density to prevent soil compaction and overgrazing.",
        "carbon_density_t_ha": 24.0,
        "impervious_fraction": 0.03,
        "biodiversity_index": 0.52,
    },
    "PermanentCrop": {
        "category": "Perennial Agriculture (Orchards/Vineyards)",
        "ecological_indicator": "Long-term woody crop rows with structured soil retention.",
        "actionable_advisory": "Monitor micro-climate frost risk and perennial water-use efficiency.",
        "carbon_density_t_ha": 48.0,
        "impervious_fraction": 0.05,
        "biodiversity_index": 0.58,
    },
    "Residential": {
        "category": "Urban & Suburban Settlement",
        "ecological_indicator": "Dense human habitation with mixed roof, road, and garden textures.",
        "actionable_advisory": "Track urban sprawl rate and green-space-to-built-area ratio for sustainable planning.",
        "carbon_density_t_ha": 9.0,
        "impervious_fraction": 0.68,
        "biodiversity_index": 0.22,
    },
    "River": {
        "category": "Freshwater Hydrological Corridor",
        "ecological_indicator": "Flowing surface water body and riparian zone.",
        "actionable_advisory": "Monitor channel turbidity, flood-plain morphological shifts, and riparian buffer health.",
        "carbon_density_t_ha": 18.0,
        "impervious_fraction": 0.0,
        "biodiversity_index": 0.86,
    },
    "SeaLake": {
        "category": "Open Water Body (Marine / Lacustrine)",
        "ecological_indicator": "Deep water reservoir or coastal zone with low visible reflectance.",
        "actionable_advisory": "Track shoreline erosion, water surface extent, and algal bloom anomalies.",
        "carbon_density_t_ha": 10.0,
        "impervious_fraction": 0.0,
        "biodiversity_index": 0.80,
    },
}

TILE_AREA_HECTARES = 40.96  # 64 px * 10 m/px = 640m x 640m = 409,600 m^2 = 40.96 ha


def compute_spectral_indices(rgb_uint8: np.ndarray) -> Dict[str, float]:
    """Compute optical remote-sensing indices from an RGB image array in [0, 255]."""
    arr = rgb_uint8.astype(np.float32) / 255.0
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    r_mean, g_mean, b_mean = float(r.mean()), float(g.mean()), float(b.mean())
    r_std, g_std, b_std = float(r.std()), float(g.std()), float(b.std())

    # Normalized Green-Red Difference Index (proxy for optical chlorophyll reflectance)
    ngrdi = float(np.mean((g - r) / (g + r + 1e-6)))
    # Excess Green Index (ExG = 2G - R - B)
    exg = float(np.mean(2.0 * g - r - b))
    # Normalized Blue-Red Water Proxy Index
    nbri_water = float(np.mean((b - r) / (b + r + 1e-6)))
    # Spatial Gradient Texture Energy
    gx = float(np.abs(np.diff(arr, axis=1)).mean())
    gy = float(np.abs(np.diff(arr, axis=0)).mean())
    texture_energy = round((gx + gy) * 0.5, 4)

    return {
        "red_mean": round(r_mean, 4),
        "green_mean": round(g_mean, 4),
        "blue_mean": round(b_mean, 4),
        "red_std": round(r_std, 4),
        "green_std": round(g_std, 4),
        "blue_std": round(b_std, 4),
        "ngrdi_vegetation_index": round(ngrdi, 4),
        "excess_green_index": round(exg, 4),
        "water_blue_red_index": round(nbri_water, 4),
        "spatial_texture_energy": texture_energy,
    }


def compute_uncertainty_metrics(probs: np.ndarray) -> Dict[str, Any]:
    """Compute Shannon predictive entropy, normalized uncertainty, and top-2 margin."""
    clipped = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    clipped = clipped / clipped.sum()
    num_classes = len(clipped)

    entropy_bits = float(-np.sum(clipped * np.log2(clipped)))
    max_entropy_bits = float(np.log2(max(num_classes, 2)))
    norm_uncertainty = float(entropy_bits / max_entropy_bits)

    sorted_probs = np.sort(clipped)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top1 - top2

    if top1 >= 0.80 and margin >= 0.50:
        tier = "High Confidence (Operational)"
    elif top1 >= 0.55:
        tier = "Moderate Confidence (Advisory)"
    else:
        tier = "Low Confidence / Mixed Boundary Tile (Human Review Recommended)"

    return {
        "shannon_entropy_bits": round(entropy_bits, 4),
        "max_entropy_bits": round(max_entropy_bits, 4),
        "normalized_uncertainty": round(norm_uncertainty, 4),
        "top2_margin": round(margin, 4),
        "reliability_tier": tier,
    }


class EuroSATPredictor:
    """High-level inference wrapper loading a trained checkpoint and running classification + Grad-CAM."""

    def __init__(
        self,
        checkpoint_path: str | Path = "outputs/checkpoints/best_eurosat_model.pth",
        config_path: str | Path = "configs/default_config.yaml",
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = load_config(config_path)
        self.device = device or get_device()
        self.checkpoint_path = Path(checkpoint_path)
        self.class_names = list(self.config["data"]["class_names"])
        self.input_size = int(self.config["data"]["model_input_size"])
        self.val_acc = 0.0

        mean = self.config["augmentation"]["imagenet_norm"]["mean"]
        std = self.config["augmentation"]["imagenet_norm"]["std"]
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self._load_checkpoint_into_model()

    def _load_checkpoint_into_model(self) -> None:
        """Load or reload the trained PyTorch model checkpoint and initialize Grad-CAM hooks."""
        arch = self.config["model"]["architecture"]
        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            arch = ckpt.get("architecture", arch)
            self.class_names = ckpt.get("class_names", self.class_names)
            self.input_size = int(ckpt.get("model_input_size", self.input_size))
            self.val_acc = float(ckpt.get("val_acc", 0.0))
            self.model = EuroSATClassifier(
                architecture=arch,
                num_classes=len(self.class_names),
                pretrained=False,
                dropout_rate=float(self.config["model"].get("dropout_rate", 0.3)),
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                f"Loaded trained checkpoint from {self.checkpoint_path} "
                f"(Val Acc: {self.val_acc*100:.2f}%)"
            )
        else:
            logger.warning(
                f"Checkpoint {self.checkpoint_path} not found! Initializing untrained {arch} for demo fallback."
            )
            self.model = EuroSATClassifier(
                architecture=arch,
                num_classes=len(self.class_names),
                pretrained=False,
            ).to(self.device)

        self.model.eval()
        self.gradcam = GradCAM(self.model, self.model.get_gradcam_target_layer())

    @staticmethod
    def _coerce_pil(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
        """Convert path, numpy array, or PIL image into a validated RGB PIL Image."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Satellite image file not found: {path}")
            return Image.open(path).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            return Image.fromarray(np.uint8(np.clip(image, 0, 255))).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image)}")

    @torch.no_grad()
    def _predict_tta(self, pil_img: Image.Image) -> Tuple[np.ndarray, float]:
        """Run 4-orientational Dihedral Test-Time Augmentation (0, 90, 180, 270 deg).

        Returns:
            mean_probs: 1D numpy array of averaged class probabilities across 4 rotations
            agreement_ratio: fraction of the 4 rotations predicting the consensus top-1 class
        """
        rotations = [0, 90, 180, 270]
        tensors = []
        for angle in rotations:
            rotated = pil_img.rotate(angle) if angle > 0 else pil_img
            tensors.append(self.transform(rotated))
        batch = torch.stack(tensors, dim=0).to(self.device)
        logits = self.model(batch)
        probs_batch = F.softmax(logits, dim=1).cpu().numpy()

        mean_probs = probs_batch.mean(axis=0)
        consensus_cls = int(np.argmax(mean_probs))
        per_rot_preds = np.argmax(probs_batch, axis=1)
        agreement_ratio = float(np.mean(per_rot_preds == consensus_cls))
        return mean_probs, agreement_ratio

    def predict_image(
        self,
        image: Image.Image | np.ndarray | str | Path,
        heatmap_alpha: float = 0.45,
        target_class_name: Optional[str] = None,
        use_tta: bool = False,
        colormap: str = "jet",
    ) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
        """Run inference, uncertainty quantification, and Grad-CAM on a single satellite image.

        Args:
            image: Input satellite image (PIL, numpy array, or file path).
            heatmap_alpha: Opacity of the Grad-CAM overlay in [0, 1].
            target_class_name: Optional class name for counterfactual Grad-CAM attribution.
                If None or 'Auto (Top-1 Predicted Class)', explains the top-1 predicted class.
            use_tta: If True, averages probabilities across 4 dihedral rotations (0, 90, 180, 270 deg).
            colormap: Matplotlib colormap name for Grad-CAM ('jet', 'inferno', 'viridis', 'plasma').

        Returns:
            class_probabilities: dict mapping class_name -> probability float
            gradcam_overlay_rgb: uint8 numpy image (256, 256, 3) with Grad-CAM heatmap overlay
            report: structured metadata dictionary including ecological advisory & uncertainty metrics
        """
        pil_img = self._coerce_pil(image)
        raw_np = np.asarray(pil_img, dtype=np.uint8)

        display_pil = pil_img.resize((256, 256), resample=Image.BILINEAR)
        display_np = np.asarray(display_pil, dtype=np.uint8)

        target_idx: Optional[int] = None
        if target_class_name and target_class_name in self.class_names:
            target_idx = self.class_names.index(target_class_name)

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        heatmap, pred_idx, confidence, probs = self.gradcam.generate(
            input_tensor,
            target_class=target_idx,
        )

        tta_agreement = 1.0
        if use_tta:
            probs, tta_agreement = self._predict_tta(pil_img)
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            if target_idx is None:
                heatmap, _, _, _ = self.gradcam.generate(input_tensor, target_class=pred_idx)

        explained_idx = pred_idx if target_idx is None else target_idx
        explained_class = self.class_names[explained_idx]

        overlay_rgb = overlay_heatmap(
            display_np,
            heatmap,
            alpha=heatmap_alpha,
            colormap_name=colormap,
        )
        pred_class = self.class_names[pred_idx]

        prob_dict = {
            self.class_names[i]: round(float(probs[i]), 4)
            for i in range(len(self.class_names))
        }

        insight = ENVIRONMENTAL_INSIGHTS.get(
            pred_class,
            {
                "category": "General Land-Cover",
                "ecological_indicator": "Standard satellite surface patch.",
                "actionable_advisory": "Continue routine geospatial monitoring.",
                "carbon_density_t_ha": 15.0,
                "impervious_fraction": 0.10,
                "biodiversity_index": 0.50,
            },
        )

        spectral = compute_spectral_indices(raw_np)
        uncertainty = compute_uncertainty_metrics(probs)

        est_carbon_tonnes = round(float(insight["carbon_density_t_ha"]) * TILE_AREA_HECTARES, 1)
        est_impervious_ha = round(float(insight["impervious_fraction"]) * TILE_AREA_HECTARES, 2)

        self.last_heatmap = heatmap
        self.last_display_rgb = display_np

        report = {
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": round(confidence, 4),
            "gradcam_target_class": explained_class,
            "is_counterfactual_gradcam": bool(explained_class != pred_class),
            "gradcam_focus_intensity": round(float(np.mean(heatmap >= 0.5)), 4),
            "tta_enabled": bool(use_tta),
            "tta_agreement_ratio": round(tta_agreement, 4),
            "category": insight["category"],
            "ecological_indicator": insight["ecological_indicator"],
            "actionable_advisory": insight["actionable_advisory"],
            "tile_area_hectares": TILE_AREA_HECTARES,
            "estimated_carbon_stock_tonnes": est_carbon_tonnes,
            "estimated_impervious_area_ha": est_impervious_ha,
            "biodiversity_habitat_score": insight["biodiversity_index"],
            "uncertainty": uncertainty,
            "spectral_indices": spectral,
            "probabilities": prob_dict,
        }
        return prob_dict, overlay_rgb, report
