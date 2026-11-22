# Phase 2 & Phase 3 Technical Roadmap

Our Phase 1 codebase (`src/`, `configs/`, `app/`, `scripts/`) was explicitly engineered with modular interfaces so that Phase 2 and Phase 3 extensions plug directly into the existing architecture without refactoring.

---

## Phase 2: Model Improvement, Multispectral Fusion & Optimization

### 1. Multi-Architecture & Vision Transformer Benchmark
* **Objective:** Systematically compare inductive biases between pure Convolutional Networks, Modernized ConvNeXt blocks, and Self-Attention Vision Transformers on overhead satellite patches.
* **Planned Models:**
  * `ResNet-18` & `ResNet-50` (Phase 1 baselines)
  * `MobileNetV3-Small` (already implemented in `src/models/classifier.py` for edge latency benchmarking)
  * `EfficientNet-B0` & `ConvNeXt-Tiny`
  * `Swin Transformer (Swin-Tiny)` / `ViT-Tiny` via `timm`
* **Evaluation Axes:** Top-1 Accuracy, Macro F1-Score, Expected Calibration Error (ECE), parameter count, FLOPs, and CPU/GPU inference latency (ms/patch).

### 2. Full 13-Band Sentinel-2 Multispectral Fusion
* **Motivation:** Phase 1 uses the 3 optical RGB bands (B04 Red, B03 Green, B02 Blue). However, Sentinel-2 captures 13 spectral bands from $443\text{ nm}$ to $2190\text{ nm}$.
* **Technical Upgrade:**
  * Adapt the first convolutional layer `conv1` from 3 input channels to 13 input channels (initializing extra channels with the mean of pretrained RGB weights) or add domain-specific spectral indices:
    * **NDVI (Normalized Difference Vegetation Index):** $\frac{\text{NIR (B08)} - \text{Red (B04)}}{\text{NIR (B08)} + \text{Red (B04)}}$ to separate `AnnualCrop`, `PermanentCrop`, `Pasture`, and `HerbaceousVegetation`.
    * **NDWI (Normalized Difference Water Index):** $\frac{\text{Green (B03)} - \text{NIR (B08)}}{\text{Green (B03)} + \text{NIR (B08)}}$ to sharpen `River` vs. shadowed terrain boundaries.

### 3. Advanced Regularization & Hard-Example Mining
* Implement **MixUp** ($\tilde{x} = \lambda x_i + (1-\lambda)x_j$) and **CutMix** patch compositing to improve calibration on multi-class boundary tiles.
* Add **Test-Time Augmentation (TTA)** averaging predictions across the 4 dihedral $90^\circ$ rotations.
* Train on the **full 27,000-image EuroSAT dataset** (`max_samples_per_class: null`) with 5-fold cross-validation.

---

## Phase 3: Advanced Geospatial Features & Production Deployment

### 1. Large-Scene Sliding-Window Land-Use Mapping
* **Feature:** Allow users to upload a high-resolution $2048 \times 2048$ satellite region (e.g., an entire city metropolitan area or river basin).
* **Implementation:** Partition the large GeoTIFF/PNG into overlapping $64 \times 64$ stride-32 windows, run batched GPU inference, and stitch the predicted class probabilities into a color-coded **Semantic Land-Use Segmentation Map** with area percentage statistics (e.g., *38% Forest, 24% Residential, 18% AnnualCrop*).

### 2. Bi-Temporal Environmental Change Detection
* **Feature:** Accept two satellite captures of the same coordinate at timestamps $t_1$ and $t_2$ (e.g., 2020 vs. 2026).
* **Implementation:** Compute class transition matrices and feature-embedding cosine distances to automatically flag **Deforestation Alerts** (`Forest` $\rightarrow$ `AnnualCrop`/`HerbaceousVegetation`) and **Urban Sprawl Alerts** (`Pasture`/`AnnualCrop` $\rightarrow$ `Residential`/`Industrial`).

### 3. Model Export, Quantization & Containerized Deployment
* Export the trained PyTorch checkpoint to **ONNX** and apply **INT8 Dynamic Quantization** for $<5\text{ ms}$ CPU inference.
* Package the Gradio/FastAPI application inside a reproducible `Dockerfile` for one-click deployment on Hugging Face Spaces.
