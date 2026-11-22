"""Single-image inference engine with ecological & environmental advisory insights and Grad-CAM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.inference.gradcam import GradCAM, overlay_heatmap
from src.models.classifier import EuroSATClassifier
from src.utils import get_device, load_config, setup_logger

logger = setup_logger("eurosat.predictor")

ENVIRONMENTAL_INSIGHTS: Dict[str, Dict[str, str]] = {
    "AnnualCrop": {
        "category": "Agricultural Land-Use",
        "ecological_indicator": "Seasonal cropland requiring irrigation & soil nutrient monitoring.",
        "actionable_advisory": "Monitor NDVI seasonal harvest cycles and check for fertilizer runoff near waterways.",
    },
    "Forest": {
        "category": "High Carbon-Sink Ecosystem",
        "ecological_indicator": "Dense tree canopy crucial for carbon sequestration and biodiversity.",
        "actionable_advisory": "Priority conservation zone: trigger alert if consecutive passes show canopy fragmentation.",
    },
    "HerbaceousVegetation": {
        "category": "Natural Shrubland & Meadow",
        "ecological_indicator": "Transitional vegetative cover and pollinator habitat.",
        "actionable_advisory": "Monitor dry-season wildfire fuel load and ecological succession patterns.",
    },
    "Highway": {
        "category": "Transportation Infrastructure",
        "ecological_indicator": "Impervious linear corridor causing habitat fragmentation.",
        "actionable_advisory": "Track roadside buffer encroachment and urban heat-island contribution.",
    },
    "Industrial": {
        "category": "High-Intensity Built Environment",
        "ecological_indicator": "Large impervious commercial/industrial roofs with high thermal footprint.",
        "actionable_advisory": "Assess rooftop solar photovoltaic (PV) potential and industrial emission buffers.",
    },
    "Pasture": {
        "category": "Managed Grazing Land",
        "ecological_indicator": "Low-intensity agricultural grassland used for livestock.",
        "actionable_advisory": "Evaluate rotational grazing density to prevent soil compaction and overgrazing.",
    },
    "PermanentCrop": {
        "category": "Perennial Agriculture (Orchards/Vineyards)",
        "ecological_indicator": "Long-term woody crop rows with structured soil retention.",
        "actionable_advisory": "Monitor micro-climate frost risk and perennial water-use efficiency.",
    },
    "Residential": {
        "category": "Urban & Suburban Settlement",
        "ecological_indicator": "Dense human habitation with mixed roof, road, and garden textures.",
        "actionable_advisory": "Track urban sprawl rate and green-space-to-built-area ratio for sustainable planning.",
    },
    "River": {
        "category": "Freshwater Hydrological Corridor",
        "ecological_indicator": "Flowing surface water body and riparian zone.",
        "actionable_advisory": "Monitor channel turbidity, flood-plain morphological shifts, and riparian buffer health.",
    },
    "SeaLake": {
        "category": "Open Water Body (Marine / Lacustrine)",
        "ecological_indicator": "Deep water reservoir or coastal zone with low visible reflectance.",
        "actionable_advisory": "Track shoreline erosion, water surface extent, and algal bloom anomalies.",
    },
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
        self.class_names = list(self.config["data"]["class_names"])
        self.input_size = int(self.config["data"]["model_input_size"])

        mean = self.config["augmentation"]["imagenet_norm"]["mean"]
        std = self.config["augmentation"]["imagenet_norm"]["std"]
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        ckpt_file = Path(checkpoint_path)
        arch = self.config["model"]["architecture"]
        if ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            arch = ckpt.get("architecture", arch)
            self.class_names = ckpt.get("class_names", self.class_names)
            self.input_size = int(ckpt.get("model_input_size", self.input_size))
            self.model = EuroSATClassifier(
                architecture=arch,
                num_classes=len(self.class_names),
                pretrained=False,
                dropout_rate=float(self.config["model"].get("dropout_rate", 0.3)),
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded trained checkpoint from {ckpt_file} (Val Acc: {ckpt.get('val_acc', 0)*100:.2f}%)")
        else:
            logger.warning(f"Checkpoint {ckpt_file} not found! Initializing untrained {arch} for demo fallback.")
            self.model = EuroSATClassifier(
                architecture=arch,
                num_classes=len(self.class_names),
                pretrained=False,
            ).to(self.device)

        self.model.eval()
        self.gradcam = GradCAM(self.model, self.model.get_gradcam_target_layer())

    def predict_image(
        self,
        image: Image.Image | np.ndarray | str | Path,
        heatmap_alpha: float = 0.45,
    ) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
        """Run inference and Grad-CAM on a single image.

        Returns:
            class_probabilities: dict mapping class_name -> probability float
            gradcam_overlay_rgb: uint8 numpy image (H, W, 3) with Grad-CAM heatmap overlay
            report: structured metadata dictionary including ecological advisory insights
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(np.uint8(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        display_pil = pil_img.resize((256, 256), resample=Image.BILINEAR)
        display_np = np.asarray(display_pil, dtype=np.uint8)

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        heatmap, pred_idx, confidence, probs = self.gradcam.generate(input_tensor)

        overlay_rgb = overlay_heatmap(display_np, heatmap, alpha=heatmap_alpha)
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
            },
        )

        report = {
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": round(confidence, 4),
            "category": insight["category"],
            "ecological_indicator": insight["ecological_indicator"],
            "actionable_advisory": insight["actionable_advisory"],
            "probabilities": prob_dict,
        }
        return prob_dict, overlay_rgb, report
