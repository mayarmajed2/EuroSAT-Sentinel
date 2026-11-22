# Dataset Documentation & Exploratory Data Analysis Methodology

## 1. Dataset Provenance, Licensing, and Academic Justification

* **Dataset Name:** **EuroSAT (RGB Benchmark)**
* **Primary Academic Citation:**
  > Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS), 12(7), 2217–2226. [https://doi.org/10.1109/JSTARS.2019.2918242](https://doi.org/10.1109/JSTARS.2019.2918242)
* **Source Satellite & Sensor:** European Space Agency (ESA) Copernicus **Sentinel-2A / 2B** satellite constellation, Multispectral Instrument (MSI), georeferenced across 34 European countries.
* **Official Repository & Access:** Hosted by DFKI / Zenodo ([https://zenodo.org/records/7711810](https://zenodo.org/records/7711810)), Hugging Face (`blanchon/EuroSAT_RGB`), and natively integrated in PyTorch `torchvision.datasets.EuroSAT`.
* **License & Usage Permissions:** Open-access under the **MIT / Creative Commons Attribution 4.0 International (CC BY 4.0)** and Copernicus Open Access data policy. Free for academic research, university coursework, and open-source redistribution.

### Why EuroSAT is Ideal for Phase 1
1. **Real-World Sensor Data:** Contains genuine spaceborne Earth Observation captures with realistic atmospheric, seasonal, and geographic diversity across 34 countries.
2. **Computationally Accessible:** Native $64 \times 64$ RGB image patches allow complete dataset download (~90 MB) and fast multi-epoch deep neural network training on a personal laptop or Google Colab.
3. **Rich Spatial-Spectral Taxonomy:** Covers 10 distinct anthropogenic, agricultural, vegetative, and hydrological classes that highlight the contrast between spatial CNN feature learning and pixel-level spectral heuristics.

---

## 2. Dataset Taxonomy and Class Specifications

The full EuroSAT dataset comprises **27,000 labeled image patches** ($64 \times 64$ pixels at $10\text{ m/pixel}$ resolution, covering $640\text{ m} \times 640\text{ m}$ ground area per patch) organized into 10 classes:

| Class Index | Class Name | Full Dataset Count | Phase 1 Stratified Subset | Functional Category | Visual & Spatial Characteristics |
| :---: | :--- | :---: | :---: | :--- | :--- |
| `0` | `AnnualCrop` | 3,000 | 300 | Agriculture | Geometric field parcels, plowing lines, seasonal green/gold hues |
| `1` | `Forest` | 3,000 | 300 | Natural Ecosystem | Dark green, high-frequency mottled tree-crown canopy texture |
| `2` | `HerbaceousVegetation` | 3,000 | 300 | Natural Ecosystem | Irregular scrubland, heath, and natural meadow textures |
| `3` | `Highway` | 2,500 | 300 | Infrastructure | Linear multi-lane road corridors cutting across varied terrain |
| `4` | `Industrial` | 2,500 | 300 | Urban / Built | Large high-albedo rectangular commercial/factory roof structures |
| `5` | `Pasture` | 2,000 | 300 | Agriculture | Homogeneous, smooth green livestock grazing parcels |
| `6` | `PermanentCrop` | 2,500 | 300 | Agriculture | Structured perennial rows (vineyards, fruit orchards, olive groves) |
| `7` | `Residential` | 3,000 | 300 | Urban / Built | High-density small rooftops, street grids, and suburban gardens |
| `8` | `River` | 2,500 | 300 | Hydrology | Meandering or straight water channels bordered by riparian banks |
| `9` | `SeaLake` | 3,000 | 300 | Hydrology | Uniform dark blue/teal open water expanses |
| **Total** | **10 Classes** | **27,000** | **3,000** | — | $64 \times 64 \times 3$ RGB JPEG patches |

*Note on Phase 1 Sampling Configuration:* To guarantee rapid, deterministic experimentation while maintaining statistical rigor, `configs/default_config.yaml` defaults to a balanced stratified subset of **300 images per class (3,000 total images: 2,100 Train / 450 Validation / 450 Test)** with `imbalance_ratio = 1.0`. Setting `max_samples_per_class: null` in `configs/default_config.yaml` (or passing `--max-samples-per-class 3000` on CLI) trains on the full 27,000-image corpus without changing a single line of code.

---

## 3. Leakage-Free Stratified Splitting Methodology

A common flaw in student computer vision projects is performing random batch splitting after data augmentation or allowing overlapping file references between training and evaluation sets. Our pipeline (`src/data/dataset.py`) enforces strict leakage prevention:

1. **Pre-Split Deduplication &Deterministic Indexing:** Every physical image file path is cataloged once with a fixed random seed (`seed = 42`).
2. **Two-Stage Stratified Partitioning:**
   * **Training Split (70%):** 2,100 images (210 per class).
   * **Validation Split (15%):** 450 images (45 per class) — used exclusively for hyperparameter monitoring, learning rate scheduling, and best-model checkpoint selection.
   * **Held-Out Test Split (15%):** 450 images (45 per class) — locked until final evaluation.
3. **Programmatic Disjointness Assertion:** `create_stratified_splits()` executes set-intersection assertions (`assert train_files.isdisjoint(val_files)`, `assert train_files.isdisjoint(test_files)`, `assert val_files.isdisjoint(test_files)`) and writes immutable split manifests to `data/splits/train.csv`, `data/splits/val.csv`, and `data/splits/test.csv`.

---

## 4. Preprocessing & Domain-Specific Data Augmentation

Unlike upright natural photography (where upside-down cars or trees are physically implausible), **nadir overhead satellite imagery is rotationally and reflectionally invariant**—North is an arbitrary cartographic convention.

* **Training Augmentations (`get_transforms` in `src/data/dataset.py`):**
  1. `Resize((128, 128))`: Upsamples $64 \times 64$ patches to $128 \times 128$ so that ResNet-18's five $2\times$ downsampling stages produce a $4 \times 4$ spatial feature map at `layer4` (essential for high-resolution Grad-CAM localization).
  2. `RandomHorizontalFlip(p=0.5)` and `RandomVerticalFlip(p=0.5)`: Exploit overhead dihedral symmetry.
  3. `RandomRotation(degrees=90)`: Simulates arbitrary satellite orbital azimuth orientations.
  4. `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)`: Simulates solar zenith angle variations, seasonal illumination differences, and light atmospheric haze.
  5. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`: Aligns input channel distributions with pretrained ImageNet convolutional filters.
* **Validation & Test Transforms:** Strictly deterministic `Resize((128, 128))` $\rightarrow$ `ToTensor()` $\rightarrow$ `Normalize(...)` with zero stochastic augmentation.

---

## 5. Exploratory Data Analysis Artifacts

Running `python scripts/run_phase1_pipeline.py` automatically generates and stores the following diagnostic EDA figures and metrics:
* `outputs/metrics/eda_summary.json`: Machine-readable verification of image dimensions, RGB channel means/standard deviations, class counts, and zero-leakage confirmation.
* `outputs/figures/eda_class_distribution.png`: Grouped bar chart verifying identical class proportions across Train (70%), Validation (15%), and Test (15%).
* `outputs/figures/eda_sample_grid.png`: Visual grid of representative Sentinel-2 patches across all 10 classes.
* `outputs/figures/eda_augmentation_showcase.png`: Visual demonstration of geometric and photometric augmentations on an original satellite tile.
* `outputs/figures/eda_spectral_signatures.png`: Per-class mean Red, Green, and Blue reflectance bars demonstrating spectral overlap across classes (e.g., `River` vs. `SeaLake` and `Highway` vs. `Residential`), visually proving why spatial convolutional layers are required.
