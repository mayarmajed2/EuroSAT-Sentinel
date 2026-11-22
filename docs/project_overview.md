# Project Overview & Idea Selection Report

**Course / Level:** University Deep Learning & Computer Vision — Phase 1  
**Project Title:** **EuroSAT-Sentinel**: Explainable Deep Learning for Satellite Land-Use & Environmental Monitoring  
**Primary Focus:** Deep Learning (Convolutional Neural Networks, Transfer Learning, Grad-CAM Explainability) & Computer Vision (Remote Sensing & Overhead Imagery Analysis)

---

## 1. Brainstorming & Comparative Analysis of 3 Innovative Project Ideas

Before selecting our project, we evaluated three realistic, visually engaging, and socially impactful Deep Learning & Computer Vision candidates suitable for university execution and live faculty demonstration.

### Idea 1: **EuroSAT-Sentinel** — Explainable Satellite Land-Use & Environmental Monitoring System *(SELECTED)*
* **Real-World Problem Solved:** Rapid urbanization, deforestation, agricultural expansion, and climate-driven hydrological changes require automated, scalable monitoring of Earth's surface. Manual GIS annotation of terabytes of Copernicus Sentinel-2 satellite imagery is slow and expensive.
* **Deep Learning & Computer Vision Core:** Fine-tunes deep residual Convolutional Neural Networks (**ResNet-18** / **MobileNetV3**) on overhead Sentinel-2 satellite imagery across 10 land-use/land-cover (LULC) classes, augmented with **Grad-CAM (Gradient-weighted Class Activation Mapping)** heatmaps to explain which spatial landforms (river meanders, highway corridors, forest canopies, industrial roofs) drive the network's predictions.
* **Interactive & Fun Component:** An interactive **Gradio Web Dashboard** where users upload or click any satellite tile to inspect real-time land-cover predictions, top-5 softmax confidence bars, interactive **Grad-CAM spatial attention overlays** with adjustable opacity, and automated **Ecological & Environmental Advisory Reports**.
* **Dataset:** **EuroSAT RGB** (Helber et al., IEEE JSTARS 2019) — 27,000 labeled $64 \times 64$ Sentinel-2 satellite patches across 10 balanced classes, open-access, and built directly into `torchvision.datasets.EuroSAT`.

### Idea 2: **SignBridge-Vision** — Real-Time Sign Language Alphabet & Gesture Recognition Assistant
* **Real-World Problem Solved:** Communication barriers between deaf/hard-of-hearing individuals and non-signers in classrooms, clinics, and public service desks.
* **Deep Learning & Computer Vision Core:** Multi-class CNN / MobileNet classification on cropped hand-gesture images (26 ASL alphabet letters + space/delete).
* **Interactive & Fun Component:** Webcam snapshot upload spelling out letter predictions.
* **Dataset:** ASL Alphabet Dataset (Kaggle, ~87,000 images).
* **Key Drawback:** Highly sensitive to uniform studio backgrounds in the Kaggle ASL dataset (models often overfit to background lighting rather than hand articulation unless complex MediaPipe hand-keypoint cropping is added first), and real-world ASL relies on continuous dynamic video morphology rather than static finger-spelling.

### Idea 3: **AgroGuard-Leaf** — Smart Crop Leaf Disease Diagnosis & Severity Estimator
* **Real-World Problem Solved:** Smallholder farmers lose 20–40% of global agricultural yield to preventable fungal, bacterial, and viral foliar diseases.
* **Deep Learning & Computer Vision Core:** Transfer learning with EfficientNet/ResNet on leaf photographs to classify healthy vs. diseased foliage.
* **Interactive & Fun Component:** Upload a smartphone leaf photo to receive disease diagnosis and treatment tips.
* **Dataset:** PlantVillage Dataset (~54,000 images across 38 classes).
* **Key Drawback:** PlantVillage images are captured on uniform grey/black paper laboratory backgrounds (`>99%` trivial accuracy on lab test splits, but severe accuracy collapse on in-field photos) and is heavily over-used in standard introductory tutorials, offering lower novelty for a university committee.

---

### Quantitative & Qualitative Comparison Matrix

| Evaluation Criterion | Idea 1: **EuroSAT-Sentinel** (Selected) | Idea 2: **SignBridge-Vision** | Idea 3: **AgroGuard-Leaf** |
| :--- | :--- | :--- | :--- |
| **1. Academic Novelty & Rigor** | **High (9.5/10)** — Combines Earth Observation CV, spatial vs. spectral ablation against Classical ML, and Grad-CAM XAI. | **Medium (7.5/10)** — Static alphabet classification is common; lacks temporal grammar in Phase 1. | **Medium (7.0/10)** — PlantVillage is the most frequently submitted standard project in university CV courses. |
| **2. Real-World Usefulness** | **Very High (9.5/10)** — Directly applicable to deforestation tracking, urban sprawl, and UN SDG 13/15 monitoring. | **High (8.5/10)** — Useful assistive tool, though static alphabet covers only finger-spelling. | **High (9.0/10)** — High agricultural value, though lab-to-field domain gap limits immediate deployment. |
| **3. Fun & Visual Demo Factor** | **Very High (9.5/10)** — Satellite tiles + interactive Grad-CAM heatmaps highlighting rivers, highways, and buildings are visually striking. | **High (8.5/10)** — Interactive webcam/photo hand letters, but sensitive to room lighting during live grading. | **Medium-High (8.0/10)** — Lesion heatmaps are clear, but professors rarely carry diseased plant leaves to class. |
| **4. Dataset Accessibility & Reliability** | **Excellent (10/10)** — Native `torchvision.datasets.EuroSAT` (~90 MB), open license, real in-the-wild Sentinel-2 imagery across 34 countries. | **Good (7.5/10)** — Requires Kaggle API token download (~1.1 GB) and suffers from video-frame correlation leakage. | **Good (8.0/10)** — Requires ~800 MB download; suffers from severe class imbalance across 38 classes. |
| **5. Feasibility on Student Hardware** | **Excellent (10/10)** — Native $64 \times 64$ (upsampled to $128 \times 128$) patches train rapidly on a single laptop GPU/CPU or Colab. | **Good (8.0/10)** — 29 classes and 87k images require longer training times. | **Good (8.0/10)** — 38 classes and $256 \times 256$ images require heavier compute. |
| **6. Phase 2 & Phase 3 Expansion Potential** | **Exceptional (10/10)** — Phase 2: 13-band multispectral fusion & Vision Transformers (ViT/Swin); Phase 3: Large-tile sliding-window map segmentation & temporal change alerts. | **High (8.5/10)** — Phase 2: MediaPipe landmarks; Phase 3: LSTM/Transformer video sequence translation. | **High (8.0/10)** — Phase 2: Background removal & field domain adaptation; Phase 3: Mobile TFLite app. |

### Selection Reasoning
**EuroSAT-Sentinel (Idea 1)** is selected because it offers the strongest balance of **real-world scientific dataset authenticity** (genuine Copernicus Sentinel-2 satellite captures rather than lab-staged photos), **clear theoretical justification for Deep Learning over Classical ML** (spatial land-use morphology vs. overlapping RGB spectral histograms), **faultless reproducibility** via PyTorch/Torchvision without external API barriers, and **outstanding visual presentation impact** through interactive Grad-CAM spatial attention mapping.

---

## 2. Project Overview & Problem Formulation

### 2.1 Name and Short Description
* **Name:** **EuroSAT-Sentinel**
* **Short Description:** An explainable Deep Learning and Computer Vision framework that classifies Sentinel-2 satellite imagery into 10 major land-use and land-cover (LULC) categories, benchmarks convolutional transfer learning against classical machine learning feature engineering, and generates interactive Grad-CAM spatial attention explanations alongside actionable ecological intelligence.

### 2.2 Problem Statement
Given an overhead optical RGB satellite patch $\mathbf{X} \in \mathbb{R}^{H \times W \times 3}$ captured by the Copernicus Sentinel-2A/2B multispectral imager at $10\text{ m/pixel}$ ground sampling distance (GSD), the objective is to learn a parameterized non-linear mapping $f_{\theta}: \mathbb{R}^{H \times W \times 3} \rightarrow \Delta^{C-1}$ (where $C=10$ land-cover classes and $\Delta^{C-1}$ is the probability simplex) that maximizes out-of-sample generalization while providing human-interpretable spatial attribution maps $\mathbf{M}_{\text{Grad-CAM}} \in [0, 1]^{H \times W}$ verifying that predictions rely on physical landforms rather than spurious sensor artifacts.

### 2.3 Target Users and Real-World Use Cases
1. **Environmental & Forestry Agencies:** Automated tracking of forest canopy loss (`Forest` $\rightarrow$ `HerbaceousVegetation` / `AnnualCrop`).
2. **Urban Planners & Municipalities:** Quantifying impervious surface expansion (`Residential`, `Industrial`, `Highway`) into agricultural greenbelts.
3. **Agricultural & Water Resource Managers:** Monitoring seasonal vs. perennial cropland (`AnnualCrop` vs. `PermanentCrop`, `Pasture`) and surface water reserves (`River`, `SeaLake`).
4. **Academic Remote Sensing Researchers:** Benchmarking lightweight CNN transfer learning and XAI methods on standardized Earth Observation benchmarks.

### 2.4 Main Objectives (Phase 1)
1. Build a **leakage-free, deterministic data pipeline** for EuroSAT with stratified Train (70%), Validation (15%), and Test (15%) splits and rotationally invariant geometric augmentations.
2. Perform comprehensive **Exploratory Data Analysis (EDA)** examining class balance, image geometry, and per-class RGB spectral signatures.
3. Implement a **Classical Machine Learning Baseline** (60-dimensional RGB Color Histogram + Spatial Gradient Statistics $\rightarrow$ Random Forest) to establish a rigorous scientific reference point.
4. Design and train a **Deep Learning Transfer Learning Classifier** (`ResNet-18` with two-stage head warmup and full fine-tuning, Cosine Annealing LR schedule, and Label Smoothing Cross-Entropy).
5. Conduct thorough **Quantitative & Qualitative Evaluation** (Top-1/Top-3 Accuracy, Macro/Per-Class Precision, Recall, F1-Score, Confusion Matrix, and Correct vs. Failure Case diagnosis).
6. Deliver **Grad-CAM Visual Explainability** and an interactive **Gradio Web Application** for live demonstration.

### 2.5 Expected Inputs and Outputs
* **Inputs:** RGB satellite patch ($64 \times 64$ native resolution, resized to $128 \times 128 \times 3$ and normalized to ImageNet statistics).
* **Outputs:**
  1. Predicted land-use class label $\hat{y} \in \{0, \dots, 9\}$ and full 10-class softmax probability distribution.
  2. $256 \times 256$ Grad-CAM spatial attention heatmap overlay highlighting discriminative sub-regions.
  3. Structured JSON/Markdown Ecological & Environmental Advisory report.

### 2.6 Why Deep Learning is Necessary for This Problem
In overhead remote sensing, **multiple distinct land-use classes share nearly identical global RGB color distributions**:
* `River` and `SeaLake` both consist of water pixels with low red and high blue/green reflectance, differing primarily in **spatial geometry** (a meandering linear channel with riparian banks vs. a homogeneous open water expanse).
* `AnnualCrop`, `PermanentCrop`, `Pasture`, and `HerbaceousVegetation` all exhibit chlorophyll-dominated green/brown spectral profiles, differing in **high-frequency spatial texture** (parallel plowing furrows, regular orchard grids, smooth grazing fields, or irregular scrub).
* `Highway`, `Industrial`, and `Residential` all contain grey concrete/asphalt spectra, differing in **multi-scale structural topology** (continuous linear road corridors vs. large warehouse footprints vs. dense neighborhood roof-and-street lattices).

Classical machine learning models that rely on global color histograms or first-order pixel statistics discard 2D spatial topology. **Convolutional Neural Networks (CNNs)** learn hierarchical, translation-equivariant spatial filters—progressing from oriented edges and textural frequencies in early layers to complex geometric motifs (road intersections, river bends, building blocks) in deeper residual blocks—making Deep Learning indispensable for accurate land-use classification.

### 2.7 Success Criteria (Phase 1)
* Zero data leakage across Train, Validation, and Test splits (verified via automated `pytest` assertions).
* Deep Learning model achieves **$\ge 90\%$ Top-1 Test Accuracy** and outperforms the Classical ML Random Forest baseline by at least **$+15\%$ absolute accuracy**.
* Grad-CAM heatmaps localize semantically meaningful regions on test patches.
* Complete test suite (`pytest`) passes 100% and interactive Gradio application runs smoothly.

### 2.8 Potential Challenges and Limitations
* **Low Native Spatial Resolution ($64 \times 64$ pixels at 10m GSD):** Fine architectural details are sub-pixel; upsampling to $128 \times 128$ allows pretrained ResNet stride hierarchies to operate without collapsing feature maps to $1 \times 1$ too early.
* **RGB vs. Multispectral Limitation in Phase 1:** Phase 1 uses the 3 optical RGB bands (B04, B03, B02) for compatibility with ImageNet pretrained weights and standard image viewers. Subtle vegetation stress that radiates strongly in Near-Infrared (NIR, B08) or Short-Wave Infrared (SWIR) is deferred to Phase 2.
* **Boundary Ambiguity:** Mixed tiles containing both a highway and surrounding forest carry a single scene-level label in EuroSAT, which can induce top-1 confusion while remaining accurate in top-3 predictions.

---

## 3. Three-Phase Project Structure & Phase 1 Scope

| Phase | Title | Core Scope & Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Foundation & Initial Implementation** | End-to-end reproducible PyTorch data & training pipeline, EDA, Classical ML comparison, ResNet-18 transfer learning baseline, test evaluation, Grad-CAM explainability, unit tests, and Gradio demo app. | **Implemented & Verified in This Repository** |
| **Phase 2** | **Model Improvement & Optimization** | Multi-architecture benchmark (EfficientNet-B0, MobileNetV3, ConvNeXt-Tiny, Vision Transformer / Swin-T), 13-band Sentinel-2 multispectral fusion (NDVI/NDWI bands), MixUp/CutMix augmentation, hyperparameter sweep, and focal loss for hard classes. | Planned (Architecture Modularized to Support) |
| **Phase 3** | **Advanced Features & Deployment** | Large-scene sliding-window geospatial mosaic segmentation, bi-temporal change detection alerts (deforestation / urban encroachment), ONNX/TensorRT quantization, and Dockerized FastAPI + Streamlit/Gradio cloud deployment. | Planned (Roadmap Documented) |
