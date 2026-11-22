"""End-to-end Phase 1 execution script for EuroSAT-Sentinel.

Executes:
1. Reproducible seed initialization & hardware verification (CUDA / CPU)
2. EuroSAT dataset download/verification & leakage-free stratified splitting (Train 70% / Val 15% / Test 15%)
3. Exploratory Data Analysis (EDA) & figure generation
4. Classical Machine Learning Baseline (Color Histogram + Spatial Gradients -> Random Forest)
5. Deep Learning Transfer Learning (ResNet-18 two-stage fine-tuning with Cosine Annealing & Checkpointing)
6. Held-out Test Set Evaluation, Confusion Matrix, Failure Analysis, and Grad-CAM Explainability Gallery

Usage:
    python scripts/run_phase1_pipeline.py --config configs/default_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_dataloaders, create_stratified_splits
from src.data.eda import run_eda
from src.evaluation.evaluator import evaluate_model, generate_gradcam_gallery
from src.models.classical_baseline import run_classical_baseline
from src.models.classifier import build_model
from src.training.trainer import Trainer
from src.utils import get_device, load_config, set_seed, setup_logger

logger = setup_logger("eurosat.pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EuroSAT-Sentinel Phase 1 End-to-End Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of training epochs.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional override for stratified samples per class.",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Force recreating stratified train/val/test CSV splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.max_samples_per_class is not None:
        config["data"]["max_samples_per_class"] = args.max_samples_per_class

    seed = int(config["project"]["seed"])
    set_seed(seed)
    device = get_device()
    logger.info(f"=== EuroSAT-Sentinel Phase 1 Pipeline | Seed: {seed} | Device: {device} ===")

    # Step 1: Dataset Loading & Stratified Splitting
    splits = create_stratified_splits(
        config,
        force_recreate=args.force_resplit,
        allow_synthetic_fallback=False,
    )

    # Step 2: Exploratory Data Analysis
    eda_summary = run_eda(config, splits)
    logger.info(
        f"EDA Summary: {eda_summary['total_images_analyzed']} images across {eda_summary['num_classes']} classes "
        f"(Imbalance Ratio: {eda_summary['imbalance_ratio_max_to_min']})"
    )

    # Step 3: Classical ML Baseline (Supporting comparison)
    classical_metrics = run_classical_baseline(config, splits)

    # Step 4: Build DataLoaders & Deep Learning Model
    loaders, _ = build_dataloaders(config, splits=splits)
    model = build_model(config)

    # Step 5: Train & Validate Deep Learning Model
    trainer = Trainer(model=model, config=config, device=device)
    training_summary = trainer.fit(loaders["train"], loaders["val"])

    # Step 6: Evaluate Best Checkpoint on Held-Out Test Split + Grad-CAM Gallery
    eval_metrics = evaluate_model(
        model=trainer.model,
        test_loader=loaders["test"],
        test_records=splits["test"],
        config=config,
        device=device,
        classical_metrics=classical_metrics,
    )
    generate_gradcam_gallery(
        model=trainer.model,
        test_records=splits["test"],
        config=config,
        device=device,
    )

    logger.info("=== Phase 1 Pipeline Completed Successfully ===")
    logger.info(
        f"Classical RF Test Acc: {classical_metrics['test_accuracy']*100:.2f}% vs. "
        f"Deep Learning ({model.architecture}) Test Acc: {eval_metrics['test_accuracy_top1']*100:.2f}% "
        f"(Macro F1: {eval_metrics['test_f1_macro']:.4f})"
    )


if __name__ == "__main__":
    main()
