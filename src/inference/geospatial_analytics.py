"""Advanced Geospatial Analytics, Bi-Temporal Change Detection, Regional Mosaic Cartography,
and Spectral-Uncertainty Diagnostic Visualizations for EuroSAT-Sentinel.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.inference.predictor import ENVIRONMENTAL_INSIGHTS, TILE_AREA_HECTARES, EuroSATPredictor

# Distinct, colorblind-aware cartographic palette for the 10 EuroSAT LULC classes
LULC_COLOR_PALETTE: Dict[str, str] = {
    "AnnualCrop": "#eab308",           # Golden amber
    "Forest": "#15803d",               # Deep forest green
    "HerbaceousVegetation": "#84cc16", # Lime meadow green
    "Highway": "#64748b",              # Slate grey
    "Industrial": "#9333ea",           # Industrial purple
    "Pasture": "#22c55e",              # Emerald pasture green
    "PermanentCrop": "#d97706",        # Orchard terracotta
    "Residential": "#ef4444",          # Urban crimson
    "River": "#0ea5e9",                # Cyan river blue
    "SeaLake": "#1d4ed8",              # Deep marine blue
}


def _fig_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    """Render a Matplotlib Figure cleanly into an RGB uint8 numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def create_patch_diagnostic_figure(
    rgb_uint8: np.ndarray,
    heatmap: np.ndarray,
    report: Dict[str, Any],
) -> np.ndarray:
    """Create a 3-panel scientific diagnostic figure for single-patch XAI inspection:
    1. Per-channel RGB Spectral Reflectance Histogram & Optical Indices
    2. Raw Grad-CAM Spatial Attention Activation Map with Contours
    3. Top-5 Softmax Probabilities & Shannon Predictive Entropy Breakdown
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: RGB Spectral Reflectance Histogram
    arr_norm = rgb_uint8.astype(np.float32) / 255.0
    bins = np.linspace(0.0, 1.0, 32)
    colors = [("#dc2626", "Red (B04)"), ("#16a34a", "Green (B03)"), ("#2563eb", "Blue (B02)")]
    for c_idx, (hex_col, lbl) in enumerate(colors):
        channel_vals = arr_norm[:, :, c_idx].ravel()
        ax1.hist(
            channel_vals,
            bins=bins,
            density=True,
            alpha=0.35,
            color=hex_col,
            label=f"{lbl} (μ={channel_vals.mean():.2f})",
        )
        hist_vals, bin_edges = np.histogram(channel_vals, bins=bins, density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax1.plot(centers, hist_vals, color=hex_col, linewidth=1.8)

    spec = report.get("spectral_indices", {})
    ax1.set_title(
        f"RGB Spectral Reflectance Profile\n"
        f"NGRDI={spec.get('ngrdi_vegetation_index', 0):.3f} | "
        f"ExG={spec.get('excess_green_index', 0):.3f} | "
        f"Tex={spec.get('spatial_texture_energy', 0):.3f}",
        fontsize=10,
        fontweight="bold",
    )
    ax1.set_xlabel("Normalized Reflectance [0, 1]", fontsize=9.5)
    ax1.set_ylabel("Spectral Density", fontsize=9.5)
    ax1.legend(fontsize=8, loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Panel 2: Raw Grad-CAM Spatial Activation Intensity + Contour Isolines
    im = ax2.imshow(heatmap, cmap="inferno", vmin=0.0, vmax=1.0)
    try:
        if float(heatmap.max() - heatmap.min()) > 0.15:
            ax2.contour(
                heatmap,
                levels=[0.50, 0.75],
                colors=["#38bdf8", "#ffffff"],
                linewidths=1.2,
                 alpha=0.85,
            )
    except Exception:
        pass
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Activation α_k · A^k", fontsize=8.5)
    target_cls = report.get("gradcam_target_class", report["predicted_class"])
    mode_tag = "Counterfactual" if report.get("is_counterfactual_gradcam") else "Top-1 Attribution"
    ax2.set_title(
        f"Grad-CAM Activation Topology ({mode_tag})\n"
        f"Target: {target_cls} | High-Focus Area: {report.get('gradcam_focus_intensity', 0)*100:.1f}%",
        fontsize=10,
        fontweight="bold",
    )
    ax2.axis("off")

    # Panel 3: Top-5 Class Probabilities & Uncertainty Metrics
    probs_dict = report["probabilities"]
    top5 = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)[:5]
    cls_names = [k for k, _ in top5][::-1]
    cls_scores = [v * 100.0 for _, v in top5][::-1]
    bar_colors = [LULC_COLOR_PALETTE.get(c, "#2563eb") for c in cls_names]

    bars = ax3.barh(cls_names, cls_scores, color=bar_colors, edgecolor="#1e293b", height=0.58)
    for bar in bars:
        w = bar.get_width()
        ax3.text(
            min(w + 1.5, 96.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )
    unc = report.get("uncertainty", {})
    ax3.set_xlim(0, 112)
    ax3.set_xlabel("Softmax Probability (%)", fontsize=9.5)
    ax3.set_title(
        f"Top-5 Posteriors & Predictive Uncertainty\n"
        f"Entropy H(p)={unc.get('shannon_entropy_bits', 0):.2f} bits | "
        f"Margin Δp={unc.get('top2_margin', 0)*100:.1f}%",
        fontsize=10,
        fontweight="bold",
    )
    ax3.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    return _fig_to_rgb_array(fig)


def format_patch_intelligence_markdown(report: Dict[str, Any]) -> str:
    """Format a comprehensive Markdown environmental & XAI report for a single satellite patch."""
    unc = report.get("uncertainty", {})
    spec = report.get("spectral_indices", {})
    tta_badge = (
        f"Enabled (`{report.get('tta_agreement_ratio', 1.0)*100:.0f}%` rotational consensus across $0^\\circ, 90^\\circ, 180^\\circ, 270^\\circ$)"
        if report.get("tta_enabled")
        else "Disabled (Single $0^\\circ$ orientation)"
    )
    cf_note = (
        f"⚠️ **Counterfactual Mode Active:** Grad-CAM is visualizing spatial evidence for **`{report['gradcam_target_class']}`** "
        f"(while top-1 predicted class is **`{report['predicted_class']}`**)."
        if report.get("is_counterfactual_gradcam")
        else f"Grad-CAM highlights physical landforms driving the top-1 prediction (**`{report['predicted_class']}`**)."
    )

    return (
        f"### 🛰️ Predicted Land-Use Class: **{report['predicted_class']}** "
        f"(`{report['confidence']*100:.2f}%` confidence)\n\n"
        f"{cf_note}\n\n"
        f"| Geospatial & Ecological Metric | Measured Value | Scientific Interpretation |\n"
        f"| :--- | :--- | :--- |\n"
        f"| **Functional Category** | **{report['category']}** | {report['ecological_indicator']} |\n"
        f"| **Prediction Reliability Tier** | **{unc.get('reliability_tier', 'N/A')}** | Top-2 Margin: `{unc.get('top2_margin', 0)*100:.2f}%` |\n"
        f"| **Shannon Predictive Entropy** | `{unc.get('shannon_entropy_bits', 0):.3f} bits` | Normalized Uncertainty: `{unc.get('normalized_uncertainty', 0)*100:.1f}%` (max `3.322 bits`) |\n"
        f"| **Dihedral TTA Status** | {tta_badge} | Rotational invariance verification |\n"
        f"| **Grad-CAM High-Attention Area** | `{report.get('gradcam_focus_intensity', 0)*100:.1f}%` of tile | Fraction of tile with normalized activation $\\ge 0.50$ |\n"
        f"| **Est. Above-Ground Carbon Pool** | `{report.get('estimated_carbon_stock_tonnes', 0):,.1f} t C` | Across `{report.get('tile_area_hectares', 40.96)} ha` ($640\\text{{ m}} \\times 640\\text{{ m}}$ at $10\\text{{ m/px}}$) |\n"
        f"| **Est. Impervious Surface Cover** | `{report.get('estimated_impervious_area_ha', 0):.2f} ha` | Sealed artificial/built surface footprint |\n"
        f"| **Optical Spectral Indices** | `NGRDI = {spec.get('ngrdi_vegetation_index', 0):+.3f}` | `ExG = {spec.get('excess_green_index', 0):+.3f}`, `Water Index = {spec.get('water_blue_red_index', 0):+.3f}` |\n\n"
        f"#### 🌱 Actionable Environmental & Planning Advisory\n"
        f"> **{report['actionable_advisory']}**"
    )


def analyze_bitemporal_change(
    predictor: EuroSATPredictor,
    image_t1: Image.Image | np.ndarray | str | Path,
    image_t2: Image.Image | np.ndarray | str | Path,
    alpha: float = 0.45,
    colormap: str = "jet",
    use_tta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """Compare two temporal satellite acquisitions (t1 vs. t2) to detect land-use transitions,
    deforestation, urban sprawl, and hydrological shifts.

    Returns:
        overlay_t1: Grad-CAM overlay image at baseline t1
        overlay_t2: Grad-CAM overlay image at comparison t2
        comparison_figure_rgb: 3-panel visual change analytics plot
        markdown_summary: Formatted executive change detection report
        change_report: Full JSON-serializable dictionary of bi-temporal metrics
    """
    probs_t1, overlay_t1, rep_t1 = predictor.predict_image(
        image_t1, heatmap_alpha=alpha, use_tta=use_tta, colormap=colormap
    )
    disp_t1 = predictor.last_display_rgb.copy()

    probs_t2, overlay_t2, rep_t2 = predictor.predict_image(
        image_t2, heatmap_alpha=alpha, use_tta=use_tta, colormap=colormap
    )
    disp_t2 = predictor.last_display_rgb.copy()

    cls_t1 = rep_t1["predicted_class"]
    cls_t2 = rep_t2["predicted_class"]
    transition_detected = cls_t1 != cls_t2

    # Probability vector comparison (Jensen-Shannon Divergence & Cosine Similarity)
    class_names = predictor.class_names
    p1 = np.array([probs_t1[c] for c in class_names], dtype=np.float64)
    p2 = np.array([probs_t2[c] for c in class_names], dtype=np.float64)
    p1 = np.clip(p1, 1e-12, 1.0) / np.sum(p1)
    p2 = np.clip(p2, 1e-12, 1.0) / np.sum(p2)
    m = 0.5 * (p1 + p2)
    kl_1 = float(np.sum(p1 * np.log2(p1 / m)))
    kl_2 = float(np.sum(p2 * np.log2(p2 / m)))
    js_divergence = round(0.5 * (kl_1 + kl_2), 4)
    cosine_sim = round(float(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-12)), 4)

    # Environmental deltas across the 40.96 ha Sentinel-2 tile
    carbon_t1 = float(rep_t1["estimated_carbon_stock_tonnes"])
    carbon_t2 = float(rep_t2["estimated_carbon_stock_tonnes"])
    delta_carbon = round(carbon_t2 - carbon_t1, 1)

    imp_t1 = float(rep_t1["estimated_impervious_area_ha"])
    imp_t2 = float(rep_t2["estimated_impervious_area_ha"])
    delta_impervious = round(imp_t2 - imp_t1, 2)

    bio_t1 = float(rep_t1["biodiversity_habitat_score"])
    bio_t2 = float(rep_t2["biodiversity_habitat_score"])
    delta_bio = round(bio_t2 - bio_t1, 3)

    # Categorize ecological transition regime
    natural_classes = {"Forest", "HerbaceousVegetation", "River", "SeaLake"}
    built_classes = {"Residential", "Industrial", "Highway"}
    agri_classes = {"AnnualCrop", "PermanentCrop", "Pasture"}

    if not transition_detected:
        alert_level = "STABLE LAND-COVER REGIME"
        alert_badge = "🟢 STABLE"
        policy_action = (
            f"No primary class transition detected (`{cls_t1}` maintained). "
            f"Continue routine monitoring (JS divergence: `{js_divergence:.3f}`)."
        )
    elif cls_t1 == "Forest" and cls_t2 in (built_classes | agri_classes | {"HerbaceousVegetation"}):
        alert_level = "CRITICAL DEFORESTATION & CANOPY LOSS ALERT"
        alert_badge = "🔴 CRITICAL ALERT"
        policy_action = (
            f"High-priority canopy loss detected (`{cls_t1}` → `{cls_t2}`), releasing an estimated "
            f"`{abs(delta_carbon):,.1f} t C` across the 40.96 ha sector. Dispatch forestry verification immediately."
        )
    elif cls_t1 not in built_classes and cls_t2 in built_classes:
        alert_level = "URBAN SPRAWL & IMPERVIOUS SOIL SEALING WARNING"
        alert_badge = "🟠 URBAN SPRAWL WARNING"
        policy_action = (
            f"Conversion of `{cls_t1}` to built infrastructure (`{cls_t2}`), increasing sealed impervious surface "
            f"by `+{delta_impervious:.2f} ha`. Audit municipal zoning and stormwater runoff capacity."
        )
    elif (cls_t1 in {"River", "SeaLake"}) != (cls_t2 in {"River", "SeaLake"}):
        alert_level = "HYDROLOGICAL / SURFACE WATER EXTENT SHIFT"
        alert_badge = "🔵 HYDROLOGICAL SHIFT"
        policy_action = (
            f"Surface water boundary transition (`{cls_t1}` → `{cls_t2}`). Inspect seasonal drought drawdown, "
            f"flood-plain morphology, or riparian infrastructure encroachment."
        )
    elif delta_carbon > 20.0 or (cls_t1 in built_classes and cls_t2 in natural_classes):
        alert_level = "ECOLOGICAL RESTORATION & BIOMASS GAIN"
        alert_badge = "🟢 RESTORATION DETECTED"
        policy_action = (
            f"Positive biomass transition (`{cls_t1}` → `{cls_t2}`) adding `+{delta_carbon:,.1f} t C` sequestration potential."
        )
    else:
        alert_level = "AGRICULTURAL / VEGETATION REGIME TRANSITION"
        alert_badge = "🟡 LAND-USE SHIFT"
        policy_action = (
            f"Land-cover transition from `{cls_t1}` to `{cls_t2}` (Net carbon delta: `{delta_carbon:+,.1f} t C`)."
        )

    # Build 2-panel comparison figure: Pixel Spectral Change Map + Class Probability Shift ΔP
    diff_map = np.linalg.norm(
        disp_t2.astype(np.float32) / 255.0 - disp_t1.astype(np.float32) / 255.0,
        axis=2,
    ) / np.sqrt(3.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.4), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    im = ax1.imshow(diff_map, cmap="magma", vmin=0.0, vmax=max(float(diff_map.max()), 0.35))
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized RGB Euclidean Change ||I_t2 - I_t1||", fontsize=8.5)
    ax1.set_title(
        f"Spatial Pixel-Level Change Intensity Map\nMean Spectral Shift: {diff_map.mean()*100:.1f}%",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.axis("off")

    delta_probs = (p2 - p1) * 100.0
    order = np.argsort(np.abs(delta_probs))[::-1][:6][::-1]
    top_classes = [class_names[i] for i in order]
    top_deltas = [delta_probs[i] for i in order]
    bar_cols = ["#dc2626" if d < 0 else "#16a34a" for d in top_deltas]

    ax2.barh(top_classes, top_deltas, color=bar_cols, edgecolor="#1e293b", height=0.58)
    ax2.axvline(0.0, color="#0f172a", linewidth=1.2)
    ax2.set_xlim(-105, 105)
    ax2.set_xlabel("Posterior Probability Shift ΔP = P(t2) - P(t1) (%)", fontsize=9.5)
    ax2.set_title(
        f"Class Probability Transition Profile (t1 → t2)\n"
        f"JS Divergence = {js_divergence:.3f} | Cosine Sim = {cosine_sim:.3f}",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    comp_fig_rgb = _fig_to_rgb_array(fig)

    md_summary = (
        f"### {alert_badge} — **{alert_level}**\n"
        f"**Detected Transition:** **`{cls_t1}`** (`{rep_t1['confidence']*100:.1f}%` @ $t_1$) "
        f"$\\longrightarrow$ **`{cls_t2}`** (`{rep_t2['confidence']*100:.1f}%` @ $t_2$)\n\n"
        f"| Environmental Impact Metric (40.96 ha Sector) | Baseline ($t_1$) | Current ($t_2$) | Net Change ($\\Delta$) |\n"
        f"| :--- | :---: | :---: | :---: |\n"
        f"| **Dominant Land-Use Class** | `{cls_t1}` | `{cls_t2}` | {'**Changed**' if transition_detected else 'Unchanged'} |\n"
        f"| **Estimated Carbon Stock (t C)** | `{carbon_t1:,.1f} t C` | `{carbon_t2:,.1f} t C` | **`{delta_carbon:+,.1f} t C`** |\n"
        f"| **Impervious Sealed Surface (ha)** | `{imp_t1:.2f} ha` | `{imp_t2:.2f} ha` | **`{delta_impervious:+.2f} ha`** |\n"
        f"| **Biodiversity Habitat Score** | `{bio_t1:.2f}` | `{bio_t2:.2f}` | **`{delta_bio:+.2f}`** |\n"
        f"| **Vegetation Index (NGRDI)** | `{rep_t1['spectral_indices']['ngrdi_vegetation_index']:+.3f}` | `{rep_t2['spectral_indices']['ngrdi_vegetation_index']:+.3f}` | **`{rep_t2['spectral_indices']['ngrdi_vegetation_index'] - rep_t1['spectral_indices']['ngrdi_vegetation_index']:+.3f}`** |\n"
        f"| **Posterior Distribution Shift** | — | — | `JSD = {js_divergence:.4f}` / `CosSim = {cosine_sim:.4f}` |\n\n"
        f"#### 🌍 Recommended Environmental Action\n"
        f"> **{policy_action}**"
    )

    change_report = {
        "alert_level": alert_level,
        "transition_detected": transition_detected,
        "baseline_t1": {
            "predicted_class": cls_t1,
            "confidence": rep_t1["confidence"],
            "carbon_stock_tonnes": carbon_t1,
            "impervious_area_ha": imp_t1,
        },
        "current_t2": {
            "predicted_class": cls_t2,
            "confidence": rep_t2["confidence"],
            "carbon_stock_tonnes": carbon_t2,
            "impervious_area_ha": imp_t2,
        },
        "deltas": {
            "carbon_stock_tonnes_delta": delta_carbon,
            "impervious_area_ha_delta": delta_impervious,
            "biodiversity_score_delta": delta_bio,
            "jensen_shannon_divergence": js_divergence,
            "probability_cosine_similarity": cosine_sim,
        },
        "policy_recommendation": policy_action,
    }
    return overlay_t1, overlay_t2, comp_fig_rgb, md_summary, change_report


def build_regional_mosaic_from_preset(
    preset_name: str,
    test_csv_path: str | Path,
    tile_px: int = 64,
) -> Tuple[Image.Image, List[List[str]]]:
    """Assemble a real multi-tile Sentinel-2 regional scene from held-out test set patches."""
    df = pd.read_csv(test_csv_path)

    presets: Dict[str, List[List[str]]] = {
        "1. Deforestation & Agricultural Frontier (3x3 — 368.6 ha)": [
            ["Forest", "Forest", "HerbaceousVegetation"],
            ["Forest", "AnnualCrop", "AnnualCrop"],
            ["River", "Pasture", "PermanentCrop"],
        ],
        "2. Metropolitan Urban-Industrial Expansion Corridor (3x3 — 368.6 ha)": [
            ["Residential", "Residential", "Highway"],
            ["Residential", "Industrial", "Industrial"],
            ["Pasture", "Highway", "AnnualCrop"],
        ],
        "3. Coastal Watershed & Riparian Conservation Basin (3x3 — 368.6 ha)": [
            ["SeaLake", "SeaLake", "River"],
            ["SeaLake", "River", "Forest"],
            ["HerbaceousVegetation", "Forest", "Pasture"],
        ],
        "4. Comprehensive 9-Class Regional Survey Grid (3x3 — 368.6 ha)": [
            ["Forest", "AnnualCrop", "Residential"],
            ["River", "Highway", "Industrial"],
            ["SeaLake", "Pasture", "PermanentCrop"],
        ],
    }

    grid_classes = presets.get(preset_name, list(presets.values())[0])
    rows = len(grid_classes)
    cols = len(grid_classes[0])

    mosaic = Image.new("RGB", (cols * tile_px, rows * tile_px))
    class_counters: Dict[str, int] = {}

    for r in range(rows):
        for c in range(cols):
            cls_name = grid_classes[r][c]
            sub = df[df["class_name"] == cls_name]
            idx = class_counters.get(cls_name, 0)
            class_counters[cls_name] = idx + 1
            if not sub.empty:
                fp = Path(sub.iloc[idx % len(sub)]["filepath"])
                if fp.exists():
                    with Image.open(fp) as tile_img:
                        patch = tile_img.convert("RGB").resize((tile_px, tile_px), resample=Image.BILINEAR)
                        mosaic.paste(patch, (c * tile_px, r * tile_px))

    return mosaic, grid_classes


def analyze_regional_mosaic(
    predictor: EuroSATPredictor,
    mosaic_image: Image.Image | np.ndarray,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> Tuple[np.ndarray, pd.DataFrame, str, Dict[str, Any]]:
    """Partition a large regional satellite scene into a (grid_rows x grid_cols) sliding-window grid,
    classify every sub-tile, render a semantic land-use cartography figure, and compute regional area statistics.
    """
    pil_mosaic = predictor._coerce_pil(mosaic_image)
    grid_rows = max(1, min(int(grid_rows), 6))
    grid_cols = max(1, min(int(grid_cols), 6))

    target_w = grid_cols * 128
    target_h = grid_rows * 128
    resized_mosaic = pil_mosaic.resize((target_w, target_h), resample=Image.BILINEAR)
    mosaic_np = np.asarray(resized_mosaic, dtype=np.uint8)

    pred_grid: List[List[str]] = []
    conf_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    cell_records: List[Dict[str, Any]] = []

    for r in range(grid_rows):
        row_preds: List[str] = []
        for c in range(grid_cols):
            y0, y1 = r * 128, (r + 1) * 128
            x0, x1 = c * 128, (c + 1) * 128
            sub_patch = Image.fromarray(mosaic_np[y0:y1, x0:x1])
            _, _, rep = predictor.predict_image(sub_patch, use_tta=False)
            pred_cls = rep["predicted_class"]
            conf = float(rep["confidence"])
            row_preds.append(pred_cls)
            conf_grid[r, c] = conf
            cell_records.append({
                "grid_coord": f"R{r+1}C{c+1}",
                "row": r + 1,
                "col": c + 1,
                "predicted_class": pred_cls,
                "confidence": conf,
                "area_ha": TILE_AREA_HECTARES,
                "carbon_tonnes": rep["estimated_carbon_stock_tonnes"],
                "impervious_ha": rep["estimated_impervious_area_ha"],
            })
        pred_grid.append(row_preds)

    cells_df = pd.DataFrame(cell_records)
    total_tiles = len(cells_df)
    total_area_ha = round(total_tiles * TILE_AREA_HECTARES, 2)

    # Aggregate regional census table by Land-Use Class
    census_rows = []
    for cls_name, grp in cells_df.groupby("predicted_class"):
        cnt = len(grp)
        area_ha = round(cnt * TILE_AREA_HECTARES, 2)
        share_pct = round((cnt / total_tiles) * 100.0, 2)
        mean_conf = round(float(grp["confidence"].mean()) * 100.0, 2)
        tot_carbon = round(float(grp["carbon_tonnes"].sum()), 1)
        tot_imp = round(float(grp["impervious_ha"].sum()), 2)
        census_rows.append({
            "Land-Use Class": cls_name,
            "Tiles": cnt,
            "Area (ha)": area_ha,
            "Regional Share (%)": share_pct,
            "Mean Confidence (%)": mean_conf,
            "Carbon Stock (t C)": tot_carbon,
            "Impervious Footprint (ha)": tot_imp,
        })
    census_df = pd.DataFrame(census_rows).sort_values("Tiles", ascending=False).reset_index(drop=True)

    # Render 3-panel Cartographic Figure:
    # Panel 1: Satellite Mosaic with Grid Overlay
    # Panel 2: Semantic Land-Use Thematic Map
    # Panel 3: Regional Area Composition Bar Chart
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    ax1.imshow(mosaic_np)
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0, x0 = r * 128, c * 128
            rect = mpatches.Rectangle((x0, y0), 128, 128, linewidth=1.4, edgecolor="#ffffff", facecolor="none")
            ax1.add_patch(rect)
            ax1.text(
                x0 + 8,
                y0 + 20,
                f"R{r+1}C{c+1}",
                color="#ffffff",
                fontsize=8,
                fontweight="bold",
                bbox={"facecolor": "#0f172a", "alpha": 0.65, "pad": 2, "edgecolor": "none"},
            )
    ax1.set_title(
        f"Composite Sentinel-2 Regional Mosaic\nGrid: {grid_rows}×{grid_cols} ({total_tiles} Sectors = {total_area_ha:.1f} ha)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.axis("off")

    # Panel 2: Semantic Land-Use Cartography Map
    semantic_rgb = np.zeros_like(mosaic_np)
    for r in range(grid_rows):
        for c in range(grid_cols):
            cls_name = pred_grid[r][c]
            hex_c = LULC_COLOR_PALETTE.get(cls_name, "#3b82f6").lstrip("#")
            rgb_tuple = tuple(int(hex_c[i : i + 2], 16) for i in (0, 2, 4))
            y0, y1 = r * 128, (r + 1) * 128
            x0, x1 = c * 128, (c + 1) * 128
            semantic_rgb[y0:y1, x0:x1] = rgb_tuple

    blended_map = np.clip(0.32 * mosaic_np.astype(np.float32) + 0.68 * semantic_rgb.astype(np.float32), 0, 255).astype(np.uint8)
    ax2.imshow(blended_map)
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0, x0 = r * 128, c * 128
            cls_name = pred_grid[r][c]
            conf_pct = conf_grid[r, c] * 100.0
            rect = mpatches.Rectangle((x0, y0), 128, 128, linewidth=1.5, edgecolor="#0f172a", facecolor="none")
            ax2.add_patch(rect)
            short_lbl = cls_name[:11]
            ax2.text(
                x0 + 64,
                y0 + 64,
                f"{short_lbl}\n({conf_pct:.0f}%)",
                color="#ffffff",
                fontsize=8,
                ha="center",
                va="center",
                fontweight="bold",
                bbox={"facecolor": "#0f172a", "alpha": 0.72, "pad": 2.5, "edgecolor": "none"},
            )
    ax2.set_title(
        f"Semantic Land-Use Classification Map\nMean Tile Confidence: {conf_grid.mean()*100:.1f}%",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.axis("off")

    # Panel 3: Regional Area Breakdown
    classes_present = census_df["Land-Use Class"].tolist()[::-1]
    shares_present = census_df["Regional Share (%)"].tolist()[::-1]
    areas_present = census_df["Area (ha)"].tolist()[::-1]
    cols_present = [LULC_COLOR_PALETTE.get(c, "#2563eb") for c in classes_present]

    bars = ax3.barh(classes_present, shares_present, color=cols_present, edgecolor="#1e293b", height=0.55)
    for idx_b, bar in enumerate(bars):
        w = bar.get_width()
        ax3.text(
            min(w + 1.5, 92.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.1f}% ({areas_present[idx_b]:.1f} ha)",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )
    ax3.set_xlim(0, 115)
    ax3.set_xlabel("Share of Regional Footprint (%)", fontsize=9.5)
    ax3.set_title("Regional Land-Use Census Breakdown", fontsize=10.5, fontweight="bold")
    ax3.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    cartography_fig_rgb = _fig_to_rgb_array(fig)

    total_carbon = round(float(cells_df["carbon_tonnes"].sum()), 1)
    total_imp = round(float(cells_df["impervious_ha"].sum()), 2)
    dominant_cls = census_df.iloc[0]["Land-Use Class"]
    dominant_share = census_df.iloc[0]["Regional Share (%)"]

    md_summary = (
        f"### 🗺️ Regional Geospatial Census Complete — `{grid_rows}×{grid_cols}` Sector Grid (`{total_area_ha:,.2f} ha`)\n\n"
        f"- **Dominant Land-Cover Regime:** **`{dominant_cls}`** (`{dominant_share:.1f}%` of total surveyed area)\n"
        f"- **Mean CNN Classification Confidence:** `{conf_grid.mean()*100:.2f}%` across `{total_tiles}` sectors\n"
        f"- **Estimated Regional Carbon Pool:** **`{total_carbon:,.1f} tonnes C`**\n"
        f"- **Total Sealed Impervious Surface:** **`{total_imp:.2f} ha`** (`{(total_imp / total_area_ha)*100:.1f}%` of region)\n"
    )

    summary_dict = {
        "grid_shape": [grid_rows, grid_cols],
        "total_sectors": total_tiles,
        "total_area_hectares": total_area_ha,
        "dominant_class": dominant_cls,
        "mean_confidence": round(float(conf_grid.mean()), 4),
        "total_carbon_stock_tonnes": total_carbon,
        "total_impervious_area_ha": total_imp,
        "census": census_rows,
        "sectors": cell_records,
    }
    return cartography_fig_rgb, census_df, md_summary, summary_dict


def filter_test_set_predictions(
    predictions_csv_path: str | Path,
    class_filter: str = "All Classes",
    outcome_filter: str = "All Predictions",
    min_confidence: float = 0.0,
) -> Tuple[pd.DataFrame, str]:
    """Filter the held-out test set evaluation predictions interactively and compute subset statistics."""
    csv_path = Path(predictions_csv_path)
    if not csv_path.exists():
        return pd.DataFrame(), "⚠️ `outputs/metrics/test_predictions.csv` not found."

    df = pd.read_csv(csv_path)
    filtered = df.copy()

    if class_filter and class_filter != "All Classes":
        filtered = filtered[filtered["true_class"] == class_filter]

    if outcome_filter == "Correct Only (433)":
        filtered = filtered[filtered["correct"] == True]  # noqa: E712
    elif outcome_filter == "Misclassified / Hard Cases Only (17)":
        filtered = filtered[filtered["correct"] == False]  # noqa: E712

    if min_confidence > 0.0:
        filtered = filtered[filtered["confidence"] >= float(min_confidence)]

    total_matches = len(filtered)
    if total_matches == 0:
        return pd.DataFrame(), "No test samples match the selected filters."

    acc = float(filtered["correct"].mean()) * 100.0
    mean_conf = float(filtered["confidence"].mean()) * 100.0
    correct_cnt = int(filtered["correct"].sum())
    error_cnt = total_matches - correct_cnt

    display_df = filtered[[
        "filename",
        "true_class",
        "pred_class",
        "confidence",
        "correct",
        "filepath",
    ]].copy()
    display_df["confidence"] = (display_df["confidence"] * 100.0).round(2)
    display_df = display_df.rename(columns={
        "filename": "Patch Filename",
        "true_class": "True Class",
        "pred_class": "Predicted Class",
        "confidence": "Confidence (%)",
        "correct": "Correct?",
        "filepath": "Filepath",
    })

    summary_md = (
        f"**Filtered Subset Statistics:** `{total_matches}` patches matched | "
        f"**Correct:** `{correct_cnt}` (`{acc:.2f}%`) | "
        f"**Misclassified:** `{error_cnt}` | "
        f"**Mean Confidence:** `{mean_conf:.2f}%`"
    )
    return display_df.reset_index(drop=True), summary_md
