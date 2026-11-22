"""CLI script to run single-image inference and save a Grad-CAM explainability visualization.

Usage:
    python scripts/predict_single.py --image path/to/patch.jpg --output outputs/figures/demo_prediction.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import EuroSATPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-image inference + Grad-CAM on a satellite patch.")
    parser.add_argument("--image", type=str, required=True, help="Path to input satellite image.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_eurosat_model.pth",
        help="Path to trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/figures/single_inference_gradcam.png",
        help="Output path to save comparison figure.",
    )
    args = parser.parse_args()

    predictor = EuroSATPredictor(checkpoint_path=args.checkpoint)
    prob_dict, overlay_rgb, report = predictor.predict_image(args.image)

    orig_img = Image.open(args.image).convert("RGB").resize((256, 256))
    top5 = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)[:5]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=180)
    ax1.imshow(orig_img)
    ax1.set_title("Input Sentinel-2 Patch", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(overlay_rgb)
    ax2.set_title(
        f"Grad-CAM Overlay\nPred: {report['predicted_class']} ({report['confidence']*100:.1f}%)",
        fontsize=11,
        fontweight="bold",
        color="#08519c",
    )
    ax2.axis("off")

    classes_top = [k for k, _ in top5][::-1]
    scores_top = [v * 100 for _, v in top5][::-1]
    ax3.barh(classes_top, scores_top, color="#2ca02c", edgecolor="black")
    ax3.set_xlim(0, 105)
    ax3.set_xlabel("Confidence (%)", fontsize=10)
    ax3.set_title("Top-5 Class Probabilities", fontsize=11, fontweight="bold")
    ax3.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print(json.dumps(report, indent=2))
    print(f"Saved visual prediction & Grad-CAM output to: {out_path}")


if __name__ == "__main__":
    main()
