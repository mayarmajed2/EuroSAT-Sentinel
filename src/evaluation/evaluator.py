"""Comprehensive model evaluation, metric computation, and diagnostic plotting for EuroSAT-Sentinel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.inference.gradcam import GradCAM, overlay_heatmap
from src.models.classifier import EuroSATClassifier
from src.utils import ensure_dirs, setup_logger

logger = setup_logger("eurosat.evaluator")


@torch.no_grad()
def evaluate_model(
    model: EuroSATClassifier,
    test_loader: DataLoader,
    test_records: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
    classical_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate trained model on the held-out test set, save metrics, and generate diagnostic figures."""
    model.eval().to(device)
    figures_dir = Path(config["evaluation"]["figures_dir"])
    metrics_dir = Path(config["evaluation"]["metrics_dir"])
    ensure_dirs(figures_dir, metrics_dir)

    class_names = list(config["data"]["class_names"])
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.numpy().tolist())

    probs_matrix = np.vstack(all_probs)
    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    # Compute top-1 and top-3 accuracy, macro/weighted precision, recall, F1
    top1_acc = float(accuracy_score(y_true, y_pred))
    top3_indices = np.argsort(probs_matrix, axis=1)[:, -3:]
    top3_acc = float(np.mean([y_true[i] in top3_indices[i] for i in range(len(y_true))]))

    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    # Identify top confused class pairs
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "count": int(cm[i, j]),
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    eval_results = {
        "architecture": model.architecture,
        "test_samples": len(y_true),
        "test_accuracy_top1": round(top1_acc, 4),
        "test_accuracy_top3": round(top3_acc, 4),
        "test_precision_macro": round(precision_macro, 4),
        "test_recall_macro": round(recall_macro, 4),
        "test_f1_macro": round(f1_macro, 4),
        "test_f1_weighted": round(f1_weighted, 4),
        "per_class_metrics": {
            cls_name: {
                "precision": round(float(cls_report[cls_name]["precision"]), 4),
                "recall": round(float(cls_report[cls_name]["recall"]), 4),
                "f1_score": round(float(cls_report[cls_name]["f1-score"]), 4),
                "support": int(cls_report[cls_name]["support"]),
            }
            for cls_name in class_names
        },
        "top_confused_pairs": confused_pairs[:10],
        "confusion_matrix": cm.tolist(),
    }

    if classical_metrics is not None:
        eval_results["classical_ml_comparison"] = {
            "classical_test_accuracy": classical_metrics["test_accuracy"],
            "deep_learning_test_accuracy": round(top1_acc, 4),
            "absolute_accuracy_gain": round(top1_acc - classical_metrics["test_accuracy"], 4),
            "classical_f1_macro": classical_metrics["test_f1_macro"],
            "deep_learning_f1_macro": round(f1_macro, 4),
            "absolute_f1_gain": round(f1_macro - classical_metrics["test_f1_macro"], 4),
        }

    with open(metrics_dir / "test_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # Save detailed per-sample predictions CSV
    pred_rows = []
    for idx, rec in enumerate(test_records):
        pred_label = int(y_pred[idx])
        true_label = int(y_true[idx])
        conf = float(probs_matrix[idx, pred_label])
        pred_rows.append({
            "filepath": rec["filepath"],
            "filename": rec["filename"],
            "true_label": true_label,
            "true_class": class_names[true_label],
            "pred_label": pred_label,
            "pred_class": class_names[pred_label],
            "confidence": round(conf, 4),
            "correct": bool(pred_label == true_label),
        })
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(metrics_dir / "test_predictions.csv", index=False)

    # Generate all required plots
    history_csv = metrics_dir / "training_history.csv"
    if history_csv.exists():
        plot_training_curves(pd.read_csv(history_csv), figures_dir / "training_curves.png")

    plot_confusion_matrix(cm, class_names, figures_dir / "confusion_matrix.png")
    plot_per_class_metrics(eval_results["per_class_metrics"], classical_metrics, top1_acc, figures_dir / "per_class_f1_and_baseline.png")
    plot_prediction_gallery(pred_df, figures_dir / "sample_predictions_correct_vs_failure.png")

    logger.info(
        f"Test Evaluation Complete | Top-1 Acc: {top1_acc*100:.2f}% | "
        f"Top-3 Acc: {top3_acc*100:.2f}% | Macro F1: {f1_macro:.4f}"
    )
    return eval_results


def plot_training_curves(history_df: pd.DataFrame, save_path: Path) -> None:
    """Plot training and validation loss and accuracy curves across epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    epochs = history_df["epoch"]

    ax1.plot(epochs, history_df["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history_df["val_loss"], "s--", label="Validation Loss", color="#d62728", linewidth=2)
    ax1.set_title("Cross-Entropy Loss Curve (Train vs. Validation)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss (with Label Smoothing = 0.05)", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(epochs, history_df["train_acc"] * 100.0, "o-", label="Train Accuracy", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, history_df["val_acc"] * 100.0, "s--", label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    best_idx = history_df["val_acc"].idxmax()
    best_ep = history_df.loc[best_idx, "epoch"]
    best_acc = history_df.loc[best_idx, "val_acc"] * 100.0
    ax2.scatter([best_ep], [best_acc], color="red", s=90, zorder=5, label=f"Best Val ({best_acc:.1f}% @ Ep {best_ep})")

    ax2.set_title("Classification Accuracy Curve (Train vs. Validation)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
    """Plot normalized and raw count confusion matrix on the held-out test split."""
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(10.5, 8.8), dpi=180)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-Normalized Recall Proportion", fontsize=10.5)

    ax.set_title("EuroSAT Held-Out Test Set Confusion Matrix (Counts & Row %)", fontsize=13, fontweight="bold")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticklabels(class_names, fontsize=9.5)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            text = f"{count}\n({pct:.0f}%)" if count > 0 else "0"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )

    ax.set_ylabel("True Land-Cover Class", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Land-Cover Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_per_class_metrics(
    per_class: Dict[str, Dict[str, Any]],
    classical_metrics: Optional[Dict[str, Any]],
    dl_top1_acc: float,
    save_path: Path,
) -> None:
    """Plot per-class Precision, Recall, and F1-Score alongside Classical ML vs. Deep Learning comparison."""
    classes = list(per_class.keys())
    prec = [per_class[c]["precision"] * 100 for c in classes]
    rec = [per_class[c]["recall"] * 100 for c in classes]
    f1 = [per_class[c]["f1_score"] * 100 for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.4, 1]}, dpi=180)

    x = np.arange(len(classes))
    w = 0.26
    ax1.bar(x - w, prec, w, label="Precision (%)", color="#1f77b4")
    ax1.bar(x, rec, w, label="Recall (%)", color="#ff7f0e")
    ax1.bar(x + w, f1, w, label="F1-Score (%)", color="#2ca02c")
    ax1.set_title("Per-Class Precision, Recall, and F1-Score on Held-Out Test Set", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Score (%)", fontsize=10.5)
    ax1.legend(frameon=True, loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Right subplot: Classical Random Forest vs Deep Learning Transfer Learning
    if classical_metrics is not None:
        models_lbl = ["Classical ML\n(Random Forest)", "Deep Learning\n(ResNet-18 CNN)"]
        acc_vals = [classical_metrics["test_accuracy"] * 100, dl_top1_acc * 100]
        bars = ax2.bar(models_lbl, acc_vals, color=["#7f7f7f", "#2ca02c"], width=0.5, edgecolor="black")
        ax2.set_ylim(0, 108)
        ax2.set_ylabel("Test Top-1 Accuracy (%)", fontsize=10.5)
        ax2.set_title("Classical ML vs. Deep Learning", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(
                f"{h:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
            )

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_prediction_gallery(pred_df: pd.DataFrame, save_path: Path) -> None:
    """Visualize 6 high-confidence correct predictions and 6 instructive failure cases."""
    correct_df = pred_df[pred_df["correct"]].sort_values("confidence", ascending=False).head(6)
    incorrect_df = pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False).head(6)

    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6), dpi=180)

    # Top row: Correct predictions
    for i in range(6):
        ax = axes[0, i]
        if i < len(correct_df):
            row = correct_df.iloc[i]
            with Image.open(row["filepath"]) as img:
                ax.imshow(np.asarray(img.convert("RGB")))
            ax.set_title(
                f"CORRECT\nTrue: {row['true_class']}\nPred: {row['pred_class']} ({row['confidence']*100:.1f}%)",
                fontsize=8.5,
                color="#1b7837",
                fontweight="bold",
            )
        ax.axis("off")

    # Bottom row: Failure cases (or lowest-confidence predictions if <6 failures)
    if len(incorrect_df) < 6:
        extra = pred_df[pred_df["correct"]].sort_values("confidence", ascending=True).head(6 - len(incorrect_df))
        bottom_df = pd.concat([incorrect_df, extra], ignore_index=True)
    else:
        bottom_df = incorrect_df

    for i in range(6):
        ax = axes[1, i]
        if i < len(bottom_df):
            row = bottom_df.iloc[i]
            with Image.open(row["filepath"]) as img:
                ax.imshow(np.asarray(img.convert("RGB")))
            status = "MISCLASSIFIED" if not row["correct"] else "LOW-CONF CORRECT"
            color = "#d73027" if not row["correct"] else "#e08214"
            ax.set_title(
                f"{status}\nTrue: {row['true_class']}\nPred: {row['pred_class']} ({row['confidence']*100:.1f}%)",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )
        ax.axis("off")

    fig.suptitle(
        "Held-Out Test Set Qualitative Evaluation: High-Confidence Correct Predictions (Top) vs. Failure/Hard Cases (Bottom)",
        fontsize=12.5,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def generate_gradcam_gallery(
    model: EuroSATClassifier,
    test_records: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> Path:
    """Generate a Grad-CAM explainability gallery overlaying spatial heatmaps on test satellite images."""
    from src.data.dataset import get_transforms

    figures_dir = Path(config["evaluation"]["figures_dir"])
    ensure_dirs(figures_dir)
    save_path = figures_dir / "gradcam_explainability_gallery.png"

    tfms = get_transforms(config)
    eval_tfm = tfms["test"]
    class_names = list(config["data"]["class_names"])
    input_size = int(config["data"]["model_input_size"])

    gradcam = GradCAM(model, model.get_gradcam_target_layer())

    # Pick 6 diverse classes from test_records
    selected_records = []
    seen_classes = set()
    for rec in test_records:
        if rec["class_name"] not in seen_classes:
            selected_records.append(rec)
            seen_classes.add(rec["class_name"])
        if len(selected_records) == 6:
            break

    fig, axes = plt.subplots(2, 6, figsize=(15, 5.4), dpi=180)
    for idx, rec in enumerate(selected_records):
        with Image.open(rec["filepath"]) as img:
            pil_rgb = img.convert("RGB").resize((input_size, input_size))
        orig_np = np.asarray(pil_rgb, dtype=np.uint8)
        input_tensor = eval_tfm(pil_rgb).unsqueeze(0).to(device)

        heatmap, pred_idx, conf, _ = gradcam.generate(input_tensor)
        overlay_np = overlay_heatmap(orig_np, heatmap, alpha=0.45)

        axes[0, idx].imshow(orig_np)
        axes[0, idx].set_title(f"Original Patch\nTrue: {rec['class_name']}", fontsize=9, fontweight="bold")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(overlay_np)
        axes[1, idx].set_title(
            f"Grad-CAM Overlay\nPred: {class_names[pred_idx]} ({conf*100:.1f}%)",
            fontsize=9,
            color="#08519c",
            fontweight="bold",
        )
        axes[1, idx].axis("off")

    gradcam.remove_hooks()
    fig.suptitle(
        "Grad-CAM Spatial Attention Explainability on Sentinel-2 Satellite Patches (Layer 4 Activations)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Grad-CAM explainability gallery to {save_path}")
    return save_path
