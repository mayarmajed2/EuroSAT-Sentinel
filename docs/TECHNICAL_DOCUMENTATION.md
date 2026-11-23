# TECHNICAL DOCUMENTATION — EuroSAT-Sentinel

This document provides an engineering reference for the modules, classes, functions, data structures, and configuration schemas in **EuroSAT-Sentinel**.

---

## 1. Technology Stack & Dependencies

| Layer | Technology / Library | Version | Purpose in EuroSAT-Sentinel |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.11+` (Verified on `3.14.5`) | Core implementation language with full type annotations (`from __future__ import annotations`). |
| **Deep Learning** | PyTorch (`torch`) | `>=2.2.0` | Tensor computation, automatic differentiation, neural network modules, and forward/backward hooks. |
| **Computer Vision** | Torchvision (`torchvision`) | `>=0.17.0` | Pretrained `ResNet18_Weights.DEFAULT` & `MobileNet_V3_Small_Weights.DEFAULT`, `datasets.EuroSAT`, and image transforms. |
| **Classical ML & Metrics** | Scikit-Learn (`scikit-learn`) | `>=1.4.0` | Stratified dataset splitting (`train_test_split`), `RandomForestClassifier`, and evaluation metrics (`f1_score`, `confusion_matrix`). |
| **Numerical & Data** | NumPy, Pandas, SciPy, Pillow | `>=1.26`, `>=2.2`, `>=10.2` | Image array manipulation, CSV manifest persistence, spectral histogram calculation, and regional census tables. |
| **Visualization** | Matplotlib | `>=3.8.0` | Headless (`Agg`) rendering of high-DPI scientific figures, Grad-CAM colormaps, contour isolines, and thematic maps. |
| **Interactive Web UI** | Gradio | `>=4.20.0` (Verified on `6.32.0`) | 5-tab interactive web dashboard with live Grad-CAM, bi-temporal simulator, mosaic scanner, and test explorer. |
| **Testing & Config** | Pytest, PyYAML | `>=8.0.0`, `>=6.0.1` | 19 automated unit/integration tests (`tests/test_pipeline.py`) and centralized YAML hyperparameter management. |

---

## 2. Configuration Schema (`configs/default_config.yaml`)

All pipeline parameters are controlled via `configs/default_config.yaml`:

* **`project`**:
  * `name`: `"EuroSAT-Sentinel"`
  * `seed`: `42` (controls Python, NumPy, PyTorch, and Scikit-Learn random states)
* **`data`**:
  * `root_dir`: `"data/raw"` (cache location for `EuroSAT` images)
  * `splits_dir`: `"data/splits"` (stores `train.csv`, `val.csv`, `test.csv`)
  * `image_size`: `64` (native Sentinel-2 tile dimensions in pixels)
  * `model_input_size`: `128` (resized tensor resolution fed to CNN)
  * `num_classes`: `10`
  * `max_samples_per_class`: `300` (`3,000` stratified images total; set to `null` or `--max-samples-per-class 3000` for all `27,000` images)
  * `train_ratio`: `0.70`, `val_ratio`: `0.15`, `test_ratio`: `0.15`
* **`model`**:
  * `architecture`: `"resnet18"` (supports `"resnet18"`, `"mobilenet_v3_small"`, `"custom_cnn"`)
  * `pretrained`: `true`
  * `dropout_rate`: `0.3`
  * `freeze_backbone_epochs`: `2`
* **`training`**:
  * `batch_size`: `32`, `epochs`: `8`, `learning_rate`: `0.0005`, `weight_decay`: `0.0001`
  * `optimizer`: `"adamw"`, `scheduler`: `"cosine"`, `label_smoothing`: `0.05`

---

## 3. Core Classes & Function Reference

### 3.1 `src/data/dataset.py`
* **`create_stratified_splits(config, force_recreate=False, allow_synthetic_fallback=False)`**
  * Scans the local EuroSAT directory (downloading via `torchvision.datasets.EuroSAT` if absent).
  * Performs two-step stratified `train_test_split` on integer class labels to produce exact `70% / 15% / 15%` partitions.
  * Executes strict set-disjointness assertions (`assert train_files.isdisjoint(val_files)`, etc.) to guarantee **zero data leakage**.
  * Persists `data/splits/train.csv`, `val.csv`, and `test.csv`.
* **`EuroSATSubsetDataset(records, transform)`**
  * PyTorch `Dataset` loading RGB images from manifest records and applying split-specific `torchvision.transforms`.

### 3.2 `src/models/classifier.py`
* **`EuroSATClassifier(architecture="resnet18", num_classes=10, pretrained=True, dropout_rate=0.3)`**
  * Unified `nn.Module` wrapper around `torchvision.models.resnet18`, `mobilenet_v3_small`, and `CustomEuroSATCNN`.
  * **`set_backbone_trainable(trainable: bool)`**: Freezes or unfreezes feature extractor layers while keeping the classification head (`fc` or `classifier`) trainable during Stage-1 warmup.
  * **`get_gradcam_target_layer() -> nn.Module`**: Returns `self.backbone.layer4[-1].conv2` for `resnet18`, `self.backbone.features[-1][0]` for `mobilenet_v3_small`, and `self.backbone.features[-3]` for `custom_cnn`.

### 3.3 `src/models/classical_baseline.py`
* **`extract_handcrafted_features(filepath, bins=16) -> np.ndarray`**
  * Computes a 60-dimensional feature vector per satellite patch:
    * `48` features: 16-bin $L_1$-normalized intensity histograms across Red, Green, and Blue channels.
    * `12` features: Per-channel mean $\mu_c$, standard deviation $\sigma_c$, horizontal finite-difference energy $\mathbb{E}[|\nabla_x I_c|]$, and vertical finite-difference energy $\mathbb{E}[|\nabla_y I_c|]$.
* **`run_classical_baseline(config, splits) -> Dict[str, Any]`**
  * Trains `RandomForestClassifier(n_estimators=200, max_depth=18, random_state=42)` and exports metrics to `outputs/metrics/classical_baseline_metrics.json`.

### 3.4 `src/training/trainer.py`
* **`Trainer(model, config, device)`**
  * Configures `nn.CrossEntropyLoss(label_smoothing=0.05)`, `AdamW` / `Adam` / `SGD`, and `CosineAnnealingLR`.
  * **`fit(train_loader, val_loader)`**:
    * Runs Stage-1 head warmup for `freeze_backbone_epochs` (`2` epochs), then unfreezes the full backbone for Stage-2 fine-tuning.
    * Applies global gradient norm clipping (`torch.nn.utils.clip_grad_norm_(..., max_norm=2.0)`).
    * Saves best checkpoint to `outputs/checkpoints/best_eurosat_model.pth` (protected against accuracy regression) and writes `training_history.csv` and `training_summary.json`.

### 3.5 `src/inference/gradcam.py`
* **`GradCAM(model, target_layer)`**
  * Attaches `register_forward_hook` and `register_full_backward_hook` to `target_layer`.
  * **`generate(input_tensor, target_class=None)`**:
    * Computes forward logits and softmax probabilities.
    * Backpropagates scalar logit $y^c$ (where $c$ is either the top-1 predicted class or a user-specified counterfactual class index).
    * Global-average-pools spatial gradients $\frac{\partial y^c}{\partial A_{i,j}^k}$ into channel weights $\alpha_k^c$, computes $\text{ReLU}(\sum_k \alpha_k^c A^k)$, upsamples bilinearly to input spatial dimensions, and min-max normalizes to $[0, 1]$.
* **`overlay_heatmap(rgb_image, heatmap, alpha=0.45, colormap_name="jet")`**
  * Blends the normalized $[0, 1]$ heatmap onto an RGB image using `jet`, `inferno`, `viridis`, `plasma`, or `magma` colormaps.

### 3.6 `src/inference/predictor.py`
* **`EuroSATPredictor(checkpoint_path, config_path, device)`**
  * High-level inference engine supporting:
    * **Dihedral Test-Time Augmentation (`_predict_tta`)**: Averages softmax predictions across $0^\circ, 90^\circ, 180^\circ, 270^\circ$ rotations and computes rotational consensus agreement.
    * **Counterfactual Grad-CAM (`target_class_name`)**: Allows querying any of the 10 land-use classes to visualize what spatial evidence exists for alternative hypotheses.
    * **Optical Remote-Sensing Spectral Indices (`compute_spectral_indices`)**: Computes Normalized Green-Red Difference Index ($\text{NGRDI}$), Excess Green Index ($\text{ExG}$), Blue-Red Water Proxy Index, and spatial texture energy.
    * **Predictive Uncertainty Quantification (`compute_uncertainty_metrics`)**: Computes Shannon entropy $H(\mathbf{p})$ in bits, normalized uncertainty in $[0, 1]$, top-2 probability margin, and operational reliability tier.

### 3.7 `src/inference/geospatial_analytics.py`
* **`create_patch_diagnostic_figure(rgb_uint8, heatmap, report)`**: Renders the 3-panel scientific diagnostic dashboard (RGB reflectance histogram, Grad-CAM activation contour map, and top-5 posterior/entropy chart).
* **`analyze_bitemporal_change(predictor, image_t1, image_t2, ...)`**: Executes multi-temporal land-use transition detection ($t_1 \rightarrow t_2$), computing carbon stock delta ($\Delta \text{t C}$), impervious surface delta ($\Delta \text{ha}$), Jensen-Shannon divergence, pixel-level Euclidean spectral difference maps, and automated policy alerts.
* **`analyze_regional_mosaic(predictor, mosaic_image, grid_rows, grid_cols)`**: Partitions a multi-tile regional scene into $G_r \times G_c$ sectors, classifies each sector, renders a semantic land-use cartography map, and compiles a regional hectare and carbon stock census DataFrame.
* **`filter_test_set_predictions(predictions_csv_path, class_filter, outcome_filter, min_confidence)`**: Powers interactive filtering and live inspection of the 450 held-out test set predictions.
