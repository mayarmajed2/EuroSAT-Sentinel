"""Interactive Gradio web application for EuroSAT-Sentinel Phase 1 demonstration.

Launch command:
    python app/gradio_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import EuroSATPredictor
from src.utils import setup_logger

logger = setup_logger("eurosat.app")


def load_sample_examples() -> List[List[Any]]:
    """Retrieve up to 6 diverse real test sample image paths for one-click interactive demo testing."""
    test_csv = PROJECT_ROOT / "data" / "splits" / "test.csv"
    examples: List[List[Any]] = []
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        for cls_name in ["Forest", "Residential", "SeaLake", "Highway", "Industrial", "AnnualCrop"]:
            sub = df[df["class_name"] == cls_name]
            if not sub.empty:
                fp = sub.iloc[0]["filepath"]
                if Path(fp).exists():
                    examples.append([fp, 0.45])
    return examples


def create_demo_interface(predictor: EuroSATPredictor):
    """Build and return the Gradio Blocks interactive interface."""
    import gradio as gr

    def analyze_patch(image: Image.Image | np.ndarray, alpha: float) -> Tuple[Dict[str, float], np.ndarray, str]:
        if image is None:
            return {}, np.zeros((256, 256, 3), dtype=np.uint8), "Please upload or select a satellite image patch."
        prob_dict, overlay_rgb, report = predictor.predict_image(image, heatmap_alpha=float(alpha))
        md_report = (
            f"### Predicted Land-Use Class: **{report['predicted_class']}** "
            f"(`{report['confidence']*100:.2f}%` confidence)\n\n"
            f"- **Functional Category:** {report['category']}\n"
            f"- **Ecological Indicator:** {report['ecological_indicator']}\n"
            f"- **Actionable Environmental Advisory:** {report['actionable_advisory']}\n"
        )
        return prob_dict, overlay_rgb, md_report

    examples = load_sample_examples()

    with gr.Blocks(title="EuroSAT-Sentinel | Explainable Satellite Land-Use AI") as demo:
        gr.Markdown(
            "# EuroSAT-Sentinel: Explainable Deep Learning for Satellite Land-Use & Environmental Monitoring\n"
            "### University Deep Learning & Computer Vision Project — Phase 1 Interactive Demo\n"
            "Upload a Sentinel-2 RGB satellite patch (or click an example below) to classify land-use across 10 "
            "categories, inspect top-10 softmax probabilities, and visualize **Grad-CAM spatial attention heatmaps**."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Input Sentinel-2 RGB Satellite Patch")
                alpha_slider = gr.Slider(
                    minimum=0.15,
                    maximum=0.80,
                    value=0.45,
                    step=0.05,
                    label="Grad-CAM Heatmap Overlay Opacity (Alpha)",
                )
                run_btn = gr.Button("Analyze Satellite Patch (Classify + Grad-CAM)", variant="primary")

            with gr.Column(scale=1):
                gradcam_out = gr.Image(type="numpy", label="Grad-CAM Spatial Attention Heatmap Overlay")
                label_out = gr.Label(num_top_classes=5, label="Top-5 Land-Cover Class Probabilities")

        advisory_md = gr.Markdown(label="Ecological & Environmental Intelligence Report")

        run_btn.click(
            fn=analyze_patch,
            inputs=[input_img, alpha_slider],
            outputs=[label_out, gradcam_out, advisory_md],
        )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[input_img, alpha_slider],
                outputs=[label_out, gradcam_out, advisory_md],
                fn=analyze_patch,
                cache_examples=False,
                label="Real Sentinel-2 Test Split Examples (Click to Run)",
            )

    return demo


def main() -> None:
    """Entry point to launch the Gradio web server."""
    predictor = EuroSATPredictor(
        checkpoint_path=PROJECT_ROOT / "outputs" / "checkpoints" / "best_eurosat_model.pth",
        config_path=PROJECT_ROOT / "configs" / "default_config.yaml",
    )
    demo = create_demo_interface(predictor)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
