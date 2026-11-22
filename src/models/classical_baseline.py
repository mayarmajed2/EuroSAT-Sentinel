"""Classical Machine Learning Baseline (Color Histogram + Texture Statistics -> Random Forest)
used exclusively as a scientific comparison benchmark to justify Deep Learning necessity.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils import ensure_dirs, setup_logger

logger = setup_logger("eurosat.classical_ml")


def extract_handcrafted_features(filepath: str | Path, bins: int = 16) -> np.ndarray:
    """Extract a 60-dimensional handcrafted feature vector per image:
    - 16-bin normalized color histograms for R, G, and B channels (48 features)
    - Channel mean, standard deviation, skewness proxy, and spatial gradient magnitude (12 features)
    """
    with Image.open(filepath) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    feats = []
    for c in range(3):
        channel = arr[:, :, c]
        hist, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0), density=True)
        hist = hist / (np.sum(hist) + 1e-8)
        feats.extend(hist.tolist())

        # Statistical moments & Sobel-like spatial gradient energies
        gx = np.abs(np.diff(channel, axis=1)).mean()
        gy = np.abs(np.diff(channel, axis=0)).mean()
        feats.extend([
            float(channel.mean()),
            float(channel.std()),
            float(gx),
            float(gy),
        ])

    return np.array(feats, dtype=np.float32)


def build_feature_matrix(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of split records into (X, y) feature arrays."""
    X_list, y_list = [], []
    for rec in records:
        X_list.append(extract_handcrafted_features(rec["filepath"]))
        y_list.append(int(rec["label"]))
    return np.vstack(X_list), np.array(y_list, dtype=np.int64)


def run_classical_baseline(
    config: Dict[str, Any],
    splits: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Train and evaluate a Random Forest classifier on handcrafted spectral-texture features."""
    seed = int(config["project"]["seed"])
    metrics_dir = Path(config["evaluation"]["metrics_dir"])
    ensure_dirs(metrics_dir)

    logger.info("Extracting handcrafted color-histogram & spatial-gradient features for Classical ML baseline...")
    t0 = time.time()
    X_train, y_train = build_feature_matrix(splits["train"])
    X_val, y_val = build_feature_matrix(splits["val"])
    X_test, y_test = build_feature_matrix(splits["test"])

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0

    val_preds = rf.predict(X_val)
    test_preds = rf.predict(X_test)

    results = {
        "model": "RandomForestClassifier (200 trees, Color Histogram + Spatial Gradient Features)",
        "feature_dimension": int(X_train.shape[1]),
        "training_time_sec": round(elapsed, 2),
        "val_accuracy": round(float(accuracy_score(y_val, val_preds)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
        "test_precision_macro": round(float(precision_score(y_test, test_preds, average="macro", zero_division=0)), 4),
        "test_recall_macro": round(float(recall_score(y_test, test_preds, average="macro", zero_division=0)), 4),
        "test_f1_macro": round(float(f1_score(y_test, test_preds, average="macro", zero_division=0)), 4),
    }

    with open(metrics_dir / "classical_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"Classical ML Baseline Completed | Test Accuracy: {results['test_accuracy']*100:.2f}% | "
        f"Macro F1: {results['test_f1_macro']:.4f}"
    )
    return results
