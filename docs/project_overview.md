# PROJECT OVERVIEW & IDEA SELECTION REPORT — EuroSAT-Sentinel

**Course / Level:** University Deep Learning & Computer Vision  
**Project Title:** **EuroSAT-Sentinel**: Explainable Deep Learning for Satellite Land-Use & Environmental Monitoring  
**Primary Focus:** Deep Learning (Convolutional Neural Networks, Transfer Learning, Counterfactual Grad-CAM Explainability, Uncertainty Quantification) & Computer Vision (Remote Sensing, Bi-Temporal Change Detection & Regional Cartography)  
**Repository:** [https://github.com/mayarmajed2/EuroSAT-Sentinel](https://github.com/mayarmajed2/EuroSAT-Sentinel)

---

## 1. Brainstorming & Comparative Analysis of 3 Innovative Project Ideas

Before selecting our project, we evaluated three realistic, visually engaging, and socially impactful Deep Learning & Computer Vision candidates suitable for university execution and live faculty demonstration.

### Idea 1: **EuroSAT-Sentinel** — Explainable Satellite Land-Use & Environmental Monitoring System *(SELECTED)*
* **Real-World Problem Solved:** Rapid urbanization, deforestation, agricultural expansion, and climate-driven hydrological changes require automated, scalable monitoring of Earth's surface. Manual GIS annotation of terabytes of Copernicus Sentinel-2 satellite imagery is slow and expensive.
* **Deep Learning & Computer Vision Core:** Fine-tunes deep residual Convolutional Neural Networks (**ResNet-18** / **MobileNetV3**) on overhead Sentinel-2 satellite imagery across 10 land-use/land-cover (LULC) classes, augmented with **Grad-CAM (Gradient-weighted Class Activation Mapping)** heatmaps, **Counterfactual Class Attribution**, **$4\times$ Dihedral Test-Time Augmentation (TTA)**, and **Shannon Predictive Entropy** to explain which spatial landforms (river meanders, highway corridors, forest canopies, industrial roofs) drive the network's predictions.
* **Interactive & Fun Component:** A 5-Tab **Gradio Geospatial AI Platform** where users inspect single satellite patches with counterfactual Grad-CAM and spectral histograms, simulate **Bi-Temporal Deforestation & Urban Sprawl ($t_1 \rightarrow t_2$)**, scan **Multi-Tile Regional Mosaics ($368.6\text{ ha}$)** into semantic thematic maps, and diagnose all 450 held-out test predictions interactively.
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
| **3. Fun & Visual Demo Factor** | **Very High (9.8/10)** — Satellite tiles + interactive Grad-CAM, bi-temporal deforestation alerts, and regional cartography maps. | **High (8.5/10)** — Interactive webcam/photo hand letters, but sensitive to room lighting during live grading. | **Medium-High (8.0/10)** — Lesion heatmaps are clear, but professors rarely carry diseased plant leaves to class. |
| **4. Dataset Accessibility & Reliability** | **Excellent (10/10)** — Native `torchvision.datasets.EuroSAT` (~90 MB), open license, real in-the-wild Sentinel-2 imagery across 34 countries. | **Good (7.5/10)** — Requires Kaggle API token download (~1.1 GB) and suffers from video-frame correlation leakage. | **Good (8.0/10)** — Requires ~800 MB download; suffers from severe class imbalance across 38 classes. |
| **5. Feasibility on Student Hardware** | **Excellent (10/10)** — Native $64 \times 64$ (upsampled to $128 \times 128$) patches train rapidly on a single laptop GPU/CPU or Colab. | **Good (8.0/10)** — 29 classes and 87k images require longer training times. | **Good (8.0/10)** — 38 classes and $256 \times 256$ images require heavier compute. |
| **6. Phase 2 & Phase 3 Expansion Potential** | **Exceptional (10/10)** — Multispectral fusion, Vision Transformers, sliding-window mosaic cartography, and bi-temporal change alerts. | **High (8.5/10)** — MediaPipe landmarks and LSTM/Transformer video sequence translation. | **High (8.0/10)** — Background removal, field domain adaptation, and mobile TFLite app. |

---

## 2. Problem Formulation & Project Motivation

### 2.1 Formal Problem Statement
Given an overhead optical RGB satellite patch $\mathbf{X} \in \mathbb{R}^{H \times W \times 3}$ captured by the Copernicus Sentinel-2A/2B multispectral imager at $10\text{ m/pixel}$ ground sampling distance (GSD, covering $640\text{ m} \times 640\text{ m} = 40.96\text{ hectares}$), the objective is to learn a parameterized non-linear mapping $f_{\theta}: \mathbb{R}^{H \times W \times 3} \rightarrow \Delta^{C-1}$ (where $C=10$ land-cover classes and $\Delta^{C-1}$ is the probability simplex) that maximizes out-of-sample generalization while providing:
1. Human-interpretable spatial attribution maps $\mathbf{M}_{\text{Grad-CAM}}^c \in [0, 1]^{H \times W}$ verifying that predictions rely on physical landforms rather than spurious sensor artifacts.
2. Calibrated predictive uncertainty metrics (Shannon entropy $H(\mathbf{p})$ and $4\times$ dihedral rotational consensus).
3. Multi-temporal ($t_1 \rightarrow t_2$) ecological transition metrics ($\Delta \text{Carbon Stock}$, $\Delta \text{Impervious Surface}$, and Jensen-Shannon divergence) and sliding-window regional cartography.

### 2.2 Why Deep Learning is Scientifically Necessary
In overhead remote sensing, **multiple distinct land-use classes share nearly identical global RGB color distributions**:
* `River` and `SeaLake` both consist of dark water pixels with high blue/green reflectance, differing primarily in **2D spatial morphology** (a meandering linear channel with riparian banks vs. a homogeneous open water expanse).
* `AnnualCrop`, `PermanentCrop`, `Pasture`, `HerbaceousVegetation`, and `Forest` all exhibit chlorophyll-dominated green/brown spectral profiles, differing in **high-frequency spatial texture** (parallel plowing furrows, regular orchard grids, smooth grazing fields, irregular scrub, or dense forest crowns).
* `Highway`, `Industrial`, and `Residential` all contain grey concrete/asphalt spectra, differing in **multi-scale structural topology** (continuous linear road corridors vs. large warehouse footprints vs. dense neighborhood roof-and-street lattices).

Classical machine learning models that rely on global color histograms or first-order pixel statistics discard 2D spatial topology. **Convolutional Neural Networks (CNNs)** learn hierarchical, translation-equivariant spatial filters—progressing from oriented edges and textural frequencies in early layers to complex geometric motifs (road intersections, river bends, building blocks) in deeper residual blocks.

Empirically, on identical leakage-free stratified splits, our fine-tuned **ResNet-18 CNN (`96.22%` Test Top-1 Accuracy, `100.00%` Top-3 Accuracy)** outperforms our **60-feature Classical ML Random Forest baseline (`76.67%`)** by **`+19.55%` absolute accuracy**, cutting the classification error rate by **83.8%**.

---

## 3. Target Users & Operational Use Cases

| Target User Group | Real-World Use Case | Platform Capability Used |
| :--- | :--- | :--- |
| **Environmental & Forestry Agencies** | Automated detection of deforestation and canopy fragmentation (`Forest` $\rightarrow$ `AnnualCrop` / `HerbaceousVegetation`). | Tab 2: Bi-Temporal Change Simulator & Carbon Stock Loss ($\Delta \text{t C}$) Alerting. |
| **Urban Planners & Municipalities** | Quantifying impervious surface expansion (`Residential`, `Industrial`, `Highway`) into agricultural greenbelts. | Tab 3: Regional Mosaic Cartography & Impervious Footprint ($\text{ha}$) Census. |
| **Agricultural & Water Resource Managers** | Monitoring seasonal vs. perennial cropland (`AnnualCrop` vs. `PermanentCrop`, `Pasture`) and surface water reserves (`River`, `SeaLake`). | Tab 1: Single-Patch XAI Inspector & Optical Spectral Indices (`NGRDI`, `ExG`). |
| **Academic Remote Sensing Researchers** | Benchmarking CNN transfer learning, counterfactual XAI, and rotational invariance on Earth Observation data. | Tab 4 & Tab 5: Interactive Test-Set Failure Explorer & Mathematical Methodology Reference. |

---

## 4. Main Objectives & Verified Outcomes

1. **Leakage-Free Stratified Data Pipeline (`src/data/dataset.py`):** Deterministic Train (`70%`, `2,100` images), Validation (`15%`, `450` images), and Held-Out Test (`15%`, `450` images) splits verified by automated set-disjointness assertions.
2. **Comprehensive Exploratory Data Analysis (`src/data/eda.py`):** Automated class distribution, patch sample grids, augmentation showcases, and RGB spectral reflectance curves.
3. **Classical ML Baseline vs. Deep Learning Benchmark (`src/models/`):** `RandomForestClassifier` (`76.67%` Test Acc) vs. Two-Stage Transfer Learning `ResNet-18` (`96.22%` Test Top-1 Acc, `100.00%` Top-3 Acc, `0.9622` Macro F1).
4. **Counterfactual Grad-CAM, Dihedral TTA & Shannon Entropy (`src/inference/predictor.py`):** Visual attribution for both top-1 and counterfactual classes, $4\times$ rotational ensemble ($0^\circ, 90^\circ, 180^\circ, 270^\circ$), and predictive entropy reliability tiers.
5. **Bi-Temporal Change Detection & Regional Mosaic Cartography (`src/inference/geospatial_analytics.py`):** Multi-temporal land-use transition alerts ($t_1 \rightarrow t_2$) and sliding-window regional thematic maps with hectare and carbon stock census tables.
6. **19 Automated Unit & Integration Tests (`tests/test_pipeline.py`):** `19 / 19` passing tests verifying every pipeline module.
