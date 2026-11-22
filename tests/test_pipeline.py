"""Automated unit and integration tests for EuroSAT-Sentinel Phase 1 pipeline.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import (
    EuroSATSubsetDataset,
    create_stratified_splits,
    generate_synthetic_eurosat_if_needed,
    get_transforms,
)
from src.inference.gradcam import GradCAM, overlay_heatmap
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
