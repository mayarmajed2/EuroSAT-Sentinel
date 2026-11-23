# TESTING AND EMPIRICAL RESULTS — EuroSAT-Sentinel

All experimental benchmarks and automated test results documented below were **actually executed and verified** in this repository (`seed = 42`). Raw output logs are persisted in [`outputs/metrics/`](../outputs/metrics/).

---

## 1. Automated Unit & Integration Test Suite (`19 / 19 Passing`)

The test suite in [`tests/test_pipeline.py`](../tests/test_pipeline.py) verifies data pipeline integrity, zero-leakage stratified splitting, neural network forward/freeze mechanics across all 3 architectures, Grad-CAM attribution, spectral indices, uncertainty metrics, counterfactual TTA, bi-temporal change detection, regional mosaic cartography, and test-set filtering.

### Execution Command & Verified Output
```bash
pytest tests/ -v -p no:cacheprovider
```

| # | Test Function Name | Module Under Test | Verified Assertion & Purpose | Result |
| :---: | :--- | :--- | :--- | :---: |
| `01` | `test_config_integrity` | `src/utils.py` | Verifies YAML keys, 10 class names, and split ratios summing to `1.0`. | **PASSED** |
| `02` | `test_stratified_splits_zero_leakage` | `src/data/dataset.py` | Verifies exact `70/15/15` stratified counts and strict file set disjointness ($\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \varnothing$). | **PASSED** |
| `03` | `test_dataset_and_transforms` | `src/data/dataset.py` | Verifies normalized tensor output shape `(3, 64, 64)` and label bounds `0 <= y < 10`. | **PASSED** |
| `04` | `test_model_architectures_forward_and_freeze[resnet18]` | `src/models/classifier.py` | Verifies forward pass `(4, 10)` and head-only vs. full-backbone trainable parameter counts for `resnet18`. | **PASSED** |
| `05` | `test_model_architectures_forward_and_freeze[mobilenet_v3_small]` | `src/models/classifier.py` | Verifies forward pass `(4, 10)` and backbone freezing for `mobilenet_v3_small`. | **PASSED** |
| `06` | `test_model_architectures_forward_and_freeze[custom_cnn]` | `src/models/classifier.py` | Verifies forward pass `(4, 10)` and backbone freezing for `CustomEuroSATCNN`. | **PASSED** |
| `07` | `test_gradcam_explainability` | `src/inference/gradcam.py` | Verifies forward/backward hook capture, heatmap range `[0, 1]`, and RGB overlay blending. | **PASSED** |
| `08` | `test_classical_feature_extractor` | `src/models/classical_baseline.py` | Verifies 60D handcrafted feature vector shape `(60,)` and numerical finiteness. | **PASSED** |
| `09` | `test_gradcam_colormaps_and_alpha_clipping[jet]` | `src/inference/gradcam.py` | Verifies `jet` colormap overlay and safe alpha clipping. | **PASSED** |
| `10` | `test_gradcam_colormaps_and_alpha_clipping[inferno]` | `src/inference/gradcam.py` | Verifies `inferno` colormap overlay and resizing. | **PASSED** |
| `11` | `test_gradcam_colormaps_and_alpha_clipping[viridis]` | `src/inference/gradcam.py` | Verifies `viridis` colormap overlay and resizing. | **PASSED** |
| `12` | `test_gradcam_colormaps_and_alpha_clipping[plasma]` | `src/inference/gradcam.py` | Verifies `plasma` colormap overlay and resizing. | **PASSED** |
| `13` | `test_gradcam_colormaps_and_alpha_clipping[invalid_fallback]` | `src/inference/gradcam.py` | Verifies graceful fallback to `jet` on unknown colormap name. | **PASSED** |
| `14` | `test_spectral_indices_and_uncertainty_metrics` | `src/inference/predictor.py` | Verifies `NGRDI`, `ExG`, Shannon entropy $H(\mathbf{p})$, top-2 margin, and reliability tiers. | **PASSED** |
| `15` | `test_predictor_counterfactual_and_tta` | `src/inference/predictor.py` | Verifies $4\times$ Dihedral TTA, counterfactual target attribution, and strict JSON serializability. | **PASSED** |
| `16` | `test_patch_diagnostic_figure_and_markdown` | `src/inference/geospatial_analytics.py` | Verifies 3-panel spectral/Grad-CAM/entropy figure rendering and Markdown formatting. | **PASSED** |
| `17` | `test_bitemporal_change_detection` | `src/inference/geospatial_analytics.py` | Verifies $t_1 \rightarrow t_2$ change detection, Jensen-Shannon divergence, carbon delta, and plot generation. | **PASSED** |
| `18` | `test_regional_mosaic_cartography_scanner` | `src/inference/geospatial_analytics.py` | Verifies sliding-window regional grid classification, hectare census ($163.84\text{ ha}$), and thematic map. | **PASSED** |
| `19` | `test_filter_test_set_predictions` | `src/inference/geospatial_analytics.py` | Verifies interactive test-set filtering by ground-truth class, outcome, and confidence threshold. | **PASSED** |

---

## 2. Empirical Model Evaluation Results

Executed via `python scripts/run_phase1_pipeline.py --config configs/default_config.yaml` on `3,000` stratified Sentinel-2 images (`2,100` Train / `450` Validation / `450` Held-Out Test, `seed = 42`).

### 2.1 Classical ML vs. Deep Learning Benchmark ($N = 450$ Held-Out Test Set)

| Model Paradigm | Feature / Backbone Architecture | Val Accuracy | Test Top-1 Accuracy | Test Top-3 Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical ML Baseline** | Random Forest (`200` trees, `60D` Color Hist + Spatial Gradients) | `76.44%` | `76.67%` | — | `0.7668` | `0.7667` | `0.7633` |
| **Deep Learning (Ours)** | **ResNet-18** (`11.18M` params, Two-Stage Transfer Learning) | **`95.56%`** | **`96.22%`** | **`100.00%`** | **`0.9629`** | **`0.9622`** | **`0.9622`** |
| **Net Improvement** | **Hierarchical Spatial CNN vs. Handcrafted Features** | **`+19.12%`** | **`+19.55%`** | — | **`+0.1961`** | **`+0.1955`** | **`+0.1989`** |

### 2.2 Per-Class Held-Out Test Set Performance (`outputs/metrics/test_evaluation_metrics.json`)

| Land-Use Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `Forest` | `1.0000` | `1.0000` | **`1.0000`** | `45` |
| `Pasture` | `0.9574` | `1.0000` | **`0.9783`** | `45` |
| `AnnualCrop` | `0.9778` | `0.9778` | **`0.9778`** | `45` |
| `HerbaceousVegetation` | `0.9565` | `0.9778` | **`0.9670`** | `45` |
| `SeaLake` | `1.0000` | `0.9333` | **`0.9655`** | `45` |
| `Highway` | `0.9362` | `0.9778` | **`0.9565`** | `45` |
| `Residential` | `0.9556` | `0.9556` | **`0.9556`** | `45` |
| `Industrial` | `0.9545` | `0.9333` | **`0.9438`** | `45` |
| `PermanentCrop` | `0.9762` | `0.9111` | **`0.9425`** | `45` |
| `River` | `0.9149` | `0.9556` | **`0.9348`** | `45` |
| **Macro Average** | **`0.9629`** | **`0.9622`** | **`0.9622`** | **`450`** |

### 2.3 Epoch-by-Epoch Two-Stage Training Progression (`outputs/metrics/training_history.csv`)

| Epoch | Stage | Backbone State | Learning Rate | Train Loss | Train Acc | Val Loss | Val Acc | Epoch Time (CPU) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `01` | `head_warmup` | Frozen | `5.00e-04` | `2.1531` | `25.24%` | `1.8789` | `45.56%` | `32.2s` |
| `02` | `head_warmup` | Frozen | `4.81e-04` | `1.6045` | `50.67%` | `1.6397` | `52.44%` | `30.7s` |
| `03` | `full_finetune` | Unfrozen | `4.28e-04` | `0.9877` | `74.57%` | `0.8664` | `80.22%` | `72.1s` |
| `04` | `full_finetune` | Unfrozen | `3.47e-04` | `0.7375` | `84.29%` | `0.6136` | `91.11%` | `68.5s` |
| `05` | `full_finetune` | Unfrozen | `2.53e-04` | `0.6183` | `88.90%` | `0.6131` | `88.22%` | `70.7s` |
| **`06`** | **`full_finetune`** | **Unfrozen** | **`1.58e-04`** | **`0.5750`** | **`89.33%`** | **`0.4295`** | **`95.56%`** | **`66.1s` (Best)** |
| `07` | `full_finetune` | Unfrozen | `7.75e-05` | `0.4832` | `93.67%` | `0.4390` | `94.67%` | `68.4s` |
| `08` | `full_finetune` | Unfrozen | `2.38e-05` | `0.4435` | `95.00%` | `0.4122` | `95.11%` | `68.5s` |

---

## 3. Known Limitations & Honest Technical Assessment

1. **3-Band Optical RGB vs. 13-Band Multispectral Sentinel-2:**
   The current benchmark utilizes the 3 visible optical bands (`B04` Red, `B03` Green, `B02` Blue) to leverage ImageNet pretrained representations and standard RGB display compatibility. While optical spectral proxies (`NGRDI` and `ExG`) are computed, true Near-Infrared (`B08` NIR) and Short-Wave Infrared (`B11/B12` SWIR) bands are not yet fused.
2. **Single-Label Patch Assumption on Mixed-Boundary Tiles:**
   Of the `17` top-1 misclassifications out of `450` held-out test images (`3.78%` error rate), visual inspection in Tab 4 confirms that most are mixed-boundary tiles (e.g., an industrial warehouse bordering a residential street, or a river crossed by a highway bridge). Notice that `100.00%` (`450 / 450`) of test patches include the ground-truth label in their top-3 predictions, and our Shannon entropy metric flags mixed tiles as higher uncertainty.
3. **European Biome Prior:**
   EuroSAT is sampled across 34 European countries. Direct zero-shot deployment on desert, tropical rainforest, or polar biomes without domain fine-tuning may experience domain shift.
