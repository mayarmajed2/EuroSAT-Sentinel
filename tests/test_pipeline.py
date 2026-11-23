"""Automated unit and integration tests for EuroSAT-Sentinel pipeline and interactive analytics.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import (
    EuroSATSubsetDataset,
    create_stratified_splits,
    generate_synthetic_eurosat_if_needed,
    get_transforms,
)
from src.inference.geospatial_analytics import (
    analyze_bitemporal_change,
    analyze_regional_mosaic,
    create_patch_diagnostic_figure,
    filter_test_set_predictions,
    format_patch_intelligence_markdown,
)
from src.inference.gradcam import GradCAM, overlay_heatmap
from src.inference.predictor import (
    TILE_AREA_HECTARES,
    EuroSATPredictor,
    compute_spectral_indices,
    compute_uncertainty_metrics,
)
from src.models.classical_baseline import extract_handcrafted_features
from src.models.classifier import EuroSATClassifier, count_parameters
from src.utils import load_config, set_seed


@pytest.fixture(scope="module")
def sample_config(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """Create an isolated test configuration pointing to a temporary directory."""
    tmp_dir = tmp_path_factory.mktemp("eurosat_test")
    cfg = load_config("configs/default_config.yaml")
    cfg["data"]["root_dir"] = str(tmp_dir / "raw")
    cfg["data"]["splits_dir"] = str(tmp_dir / "splits")
    cfg["data"]["max_samples_per_class"] = 20
    cfg["data"]["model_input_size"] = 64
    cfg["training"]["batch_size"] = 8
    cfg["training"]["epochs"] = 1
    cfg["training"]["checkpoint_dir"] = str(tmp_dir / "checkpoints")
    cfg["evaluation"]["figures_dir"] = str(tmp_dir / "figures")
    cfg["evaluation"]["metrics_dir"] = str(tmp_dir / "metrics")
    return cfg


def test_config_integrity() -> None:
    """Verify default YAML configuration contains all required keys and valid split ratios."""
    cfg = load_config("configs/default_config.yaml")
    assert "project" in cfg and "data" in cfg and "model" in cfg and "training" in cfg
    assert len(cfg["data"]["class_names"]) == cfg["data"]["num_classes"] == 10
    total_ratio = cfg["data"]["train_ratio"] + cfg["data"]["val_ratio"] + cfg["data"]["test_ratio"]
    assert np.isclose(total_ratio, 1.0)


def test_stratified_splits_zero_leakage(sample_config: Dict[str, Any]) -> None:
    """Verify stratified train/val/test split creation guarantees zero file overlap across splits."""
    set_seed(42)
    generate_synthetic_eurosat_if_needed(
        root_dir=sample_config["data"]["root_dir"],
        class_names=sample_config["data"]["class_names"],
        samples_per_class=20,
        image_size=64,
    )
    splits = create_stratified_splits(sample_config, force_recreate=True, allow_synthetic_fallback=True)
    assert len(splits["train"]) == 140
    assert len(splits["val"]) == 30
    assert len(splits["test"]) == 30

    train_set = {r["filepath"] for r in splits["train"]}
    val_set = {r["filepath"] for r in splits["val"]}
    test_set = {r["filepath"] for r in splits["test"]}

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_dataset_and_transforms(sample_config: Dict[str, Any]) -> None:
    """Verify EuroSATSubsetDataset and domain transforms output normalized tensors of expected shape."""
    splits = create_stratified_splits(sample_config, force_recreate=False, allow_synthetic_fallback=True)
    tfms = get_transforms(sample_config)
    ds = EuroSATSubsetDataset(splits["train"], transform=tfms["train"])

    tensor_img, label = ds[0]
    assert isinstance(tensor_img, torch.Tensor)
    assert tensor_img.shape == (3, 64, 64)
    assert 0 <= label < 10


@pytest.mark.parametrize("arch", ["resnet18", "mobilenet_v3_small", "custom_cnn"])
def test_model_architectures_forward_and_freeze(arch: str) -> None:
    """Test forward pass shapes and backbone freezing across all supported CNN architectures."""
    model = EuroSATClassifier(architecture=arch, num_classes=10, pretrained=False, dropout_rate=0.2)
    x = torch.randn(4, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (4, 10)

    total_params, trainable_all = count_parameters(model)
    assert total_params == trainable_all > 0

    model.set_backbone_trainable(False)
    _, trainable_frozen = count_parameters(model)
    assert 0 < trainable_frozen < total_params

    model.set_backbone_trainable(True)
    _, trainable_unfrozen = count_parameters(model)
    assert trainable_unfrozen == total_params


def test_gradcam_explainability() -> None:
    """Test Grad-CAM hook capture, heatmap normalization in [0, 1], and RGB overlay blending."""
    model = EuroSATClassifier(architecture="custom_cnn", num_classes=10, pretrained=False)
    gradcam = GradCAM(model, model.get_gradcam_target_layer())

    x = torch.randn(1, 3, 64, 64)
    heatmap, pred_cls, conf, probs = gradcam.generate(x)
    gradcam.remove_hooks()

    assert heatmap.shape == (64, 64)
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0 + 1e-5
    assert 0 <= pred_cls < 10
    assert 0.0 <= conf <= 1.0
    assert np.isclose(probs.sum(), 1.0, atol=1e-4)

    dummy_rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
    blended = overlay_heatmap(dummy_rgb, heatmap, alpha=0.45)
    assert blended.shape == (64, 64, 3)
    assert blended.dtype == np.uint8


def test_classical_feature_extractor(tmp_path: Path) -> None:
    """Test handcrafted feature extraction vector dimensionality and finiteness."""
    img_path = tmp_path / "patch.jpg"
    arr = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    feats = extract_handcrafted_features(img_path, bins=16)
    assert feats.shape == (60,)
    assert np.all(np.isfinite(feats))


@pytest.mark.parametrize("cmap", ["jet", "inferno", "viridis", "plasma", "invalid_fallback"])
def test_gradcam_colormaps_and_alpha_clipping(cmap: str) -> None:
    """Verify overlay_heatmap supports multiple colormaps and safely clips alpha bounds."""
    rgb = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    heatmap = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    out = overlay_heatmap(rgb, heatmap, alpha=1.5, colormap_name=cmap)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_spectral_indices_and_uncertainty_metrics() -> None:
    """Verify optical spectral index calculation and Shannon entropy / reliability tier logic."""
    green_patch = np.zeros((64, 64, 3), dtype=np.uint8)
    green_patch[:, :, 0] = 40
    green_patch[:, :, 1] = 180
    green_patch[:, :, 2] = 50

    spec = compute_spectral_indices(green_patch)
    assert spec["ngrdi_vegetation_index"] > 0.5
    assert spec["excess_green_index"] > 0.5

    peaked_probs = np.array([0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    unc_high = compute_uncertainty_metrics(peaked_probs)
    assert unc_high["shannon_entropy_bits"] < 0.8
    assert unc_high["top2_margin"] == pytest.approx(0.90, abs=1e-3)
    assert "High Confidence" in unc_high["reliability_tier"]

    uniform_probs = np.full(10, 0.10)
    unc_low = compute_uncertainty_metrics(uniform_probs)
    assert unc_low["normalized_uncertainty"] == pytest.approx(1.0, abs=1e-3)
    assert "Low Confidence" in unc_low["reliability_tier"]


def test_predictor_counterfactual_and_tta(tmp_path: Path) -> None:
    """Verify EuroSATPredictor supports Dihedral TTA, counterfactual target classes, and JSON serialization."""
    predictor = EuroSATPredictor(
        checkpoint_path=tmp_path / "nonexistent.pth",
        config_path="configs/default_config.yaml",
        device=torch.device("cpu"),
    )
    dummy_img = Image.fromarray(np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8))

    probs, overlay, report = predictor.predict_image(
        dummy_img,
        heatmap_alpha=0.5,
        target_class_name="Highway",
        use_tta=True,
        colormap="inferno",
    )
    assert len(probs) == 10
    assert overlay.shape == (256, 256, 3)
    assert report["gradcam_target_class"] == "Highway"
    assert report["tta_enabled"] is True
    assert 0.0 <= report["tta_agreement_ratio"] <= 1.0
    # Ensure report is 100% JSON serializable (critical for CLI & export)
    serialized = json.dumps(report)
    assert isinstance(serialized, str) and len(serialized) > 50


def test_patch_diagnostic_figure_and_markdown(tmp_path: Path) -> None:
    """Verify 3-panel scientific diagnostic figure and Markdown formatter execute cleanly."""
    predictor = EuroSATPredictor(
        checkpoint_path=tmp_path / "nonexistent.pth",
        config_path="configs/default_config.yaml",
        device=torch.device("cpu"),
    )
    dummy_img = Image.fromarray(np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8))
    _, _, report = predictor.predict_image(dummy_img)

    diag_rgb = create_patch_diagnostic_figure(
        predictor.last_display_rgb,
        predictor.last_heatmap,
        report,
    )
    assert diag_rgb.ndim == 3 and diag_rgb.shape[2] == 3
    assert diag_rgb.dtype == np.uint8

    md = format_patch_intelligence_markdown(report)
    assert "Predicted Land-Use Class" in md
    assert "Shannon Predictive Entropy" in md


def test_bitemporal_change_detection(tmp_path: Path) -> None:
    """Verify bi-temporal change detection engine produces valid overlays, plots, and JSON report."""
    predictor = EuroSATPredictor(
        checkpoint_path=tmp_path / "nonexistent.pth",
        config_path="configs/default_config.yaml",
        device=torch.device("cpu"),
    )
    img_t1 = Image.fromarray(np.full((64, 64, 3), 40, dtype=np.uint8))
    img_t2 = Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8))

    ov1, ov2, comp_fig, md_summary, change_rep = analyze_bitemporal_change(
        predictor, img_t1, img_t2, alpha=0.45, colormap="jet", use_tta=False
    )
    assert ov1.shape == (256, 256, 3)
    assert ov2.shape == (256, 256, 3)
    assert comp_fig.ndim == 3 and comp_fig.shape[2] == 3
    assert "Estimated Carbon Stock" in md_summary
    assert "deltas" in change_rep
    assert 0.0 <= change_rep["deltas"]["jensen_shannon_divergence"] <= 1.0 + 1e-5


def test_regional_mosaic_cartography_scanner(tmp_path: Path) -> None:
    """Verify sliding-window regional mosaic scanner computes accurate hectare census and cartography map."""
    predictor = EuroSATPredictor(
        checkpoint_path=tmp_path / "nonexistent.pth",
        config_path="configs/default_config.yaml",
        device=torch.device("cpu"),
    )
    mosaic_img = Image.fromarray(np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8))
    cart_fig, census_df, md_summary, summary_dict = analyze_regional_mosaic(
        predictor, mosaic_img, grid_rows=2, grid_cols=2
    )
    assert cart_fig.ndim == 3 and cart_fig.shape[2] == 3
    assert summary_dict["total_sectors"] == 4
    assert summary_dict["total_area_hectares"] == pytest.approx(4 * TILE_AREA_HECTARES, abs=1e-2)
    assert int(census_df["Tiles"].sum()) == 4
    assert "Regional Geospatial Census Complete" in md_summary


def test_filter_test_set_predictions(tmp_path: Path) -> None:
    """Verify interactive test-set prediction filtering by class, outcome, and confidence threshold."""
    csv_path = tmp_path / "test_predictions.csv"
    df = pd.DataFrame([
        {
            "filepath": "a.jpg",
            "filename": "Forest_1.jpg",
            "true_label": 1,
            "true_class": "Forest",
            "pred_label": 1,
            "pred_class": "Forest",
            "confidence": 0.99,
            "correct": True,
        },
        {
            "filepath": "b.jpg",
            "filename": "River_2.jpg",
            "true_label": 8,
            "true_class": "River",
            "pred_label": 3,
            "pred_class": "Highway",
            "confidence": 0.62,
            "correct": False,
        },
    ])
    df.to_csv(csv_path, index=False)

    all_df, _ = filter_test_set_predictions(csv_path)
    assert len(all_df) == 2

    err_df, _ = filter_test_set_predictions(
        csv_path, outcome_filter="Misclassified / Hard Cases Only (17)"
    )
    assert len(err_df) == 1
    assert err_df.iloc[0]["True Class"] == "River"

    high_conf_df, _ = filter_test_set_predictions(csv_path, min_confidence=0.90)
    assert len(high_conf_df) == 1
