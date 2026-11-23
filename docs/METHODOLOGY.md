# ACADEMIC & TECHNICAL METHODOLOGY — EuroSAT-Sentinel

This document presents the theoretical foundations, mathematical formulations, experimental design, and algorithmic methodology implemented in **EuroSAT-Sentinel**.

---

## 1. Problem Formulation

Let $\mathcal{D} = \{(\mathbf{X}_i, y_i)\}_{i=1}^{N}$ denote the Copernicus Sentinel-2 optical satellite benchmark dataset (**EuroSAT RGB**, Helber et al., 2019), where each patch $\mathbf{X}_i \in [0, 255]^{64 \times 64 \times 3}$ is captured at a nadir ground sampling distance (GSD) of $10\text{ m/pixel}$ (covering $640\text{ m} \times 640\text{ m} = 40.96\text{ hectares}$) and $y_i \in \{0, 1, \dots, C-1\}$ represents one of $C = 10$ mutually exclusive land-use/land-cover (LULC) categories.

Our goal is threefold:
1. **Discriminative Classification:** Learn a deep convolutional parameterization $f_\theta: \mathbb{R}^{128 \times 128 \times 3} \rightarrow \mathbb{R}^C$ yielding calibrated class posterior probabilities $\mathbf{p} = \text{Softmax}(f_\theta(\tilde{\mathbf{X}})) \in \Delta^{C-1}$.
2. **Spatial Visual Attribution (XAI):** Compute class-specific spatial attribution maps $\mathbf{M}_{\text{Grad-CAM}}^c \in [0, 1]^{128 \times 128}$ that localize the physical landforms responsible for predicting class $c$ (or any counterfactual query class $c' \neq \hat{y}$).
3. **Uncertainty & Multi-Temporal Geospatial Quantification:** Quantify predictive epistemic/aleatoric uncertainty via Shannon entropy $H(\mathbf{p})$ and rotational dihedral consensus, and measure bi-temporal ecological transitions ($\Delta \text{Carbon}$, $\Delta \text{Impervious Area}$, and Jensen-Shannon divergence).

---

## 2. Leakage-Free Stratified Splitting & Overhead Geometric Invariance

### 2.1 Deterministic Two-Stage Stratified Partitioning
To prevent class-ratio skew and data leakage (`src/data/dataset.py`), the dataset $\mathcal{D}$ is partitioned deterministically (`seed = 42`) into three mutually disjoint subsets $\mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}} = \mathcal{D}$ with proportions $(0.70, 0.15, 0.15)$ such that for every class $c \in \{0, \dots, 9\}$:

$$\frac{|\{(\mathbf{X}, y) \in \mathcal{D}_{\text{split}} : y = c\}|}{|\mathcal{D}_{\text{split}}|} = \frac{1}{C} = 0.10$$

Programmatic set-disjointness assertions verify strict zero overlap:

$$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \varnothing, \quad \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \varnothing, \quad \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \varnothing$$

### 2.2 Dihedral Symmetry in Nadir Satellite Imagery
Unlike ground-level natural photography (where gravity defines a canonical vertical orientation), nadir overhead satellite imagery exhibits **dihedral group symmetry $D_4$** (invariance to $90^\circ$ rotations and horizontal/vertical reflections). Our training augmentation pipeline (`get_transforms`) enforces this physical prior via:
* Random Horizontal Flip ($p = 0.5$) & Random Vertical Flip ($p = 0.5$)
* Continuous Random Rotation ($\theta \sim \mathcal{U}(-90^\circ, +90^\circ)$)
* Mild Radiometric Color Jitter (brightness, contrast, saturation $\pm 15\%$, hue $\pm 5\%$) simulating atmospheric transmittance and solar zenith angle variations
* Channel-wise standardization using ImageNet statistics:
  $$\tilde{\mathbf{X}}_{:,:,c} = \frac{\mathbf{X}_{:,:,c} - \mu_c}{\sigma_c}, \quad \boldsymbol{\mu} = [0.485, 0.456, 0.406], \quad \boldsymbol{\sigma} = [0.229, 0.224, 0.225]$$

---

## 3. Classical Machine Learning Baseline Formulation

To establish a rigorous scientific baseline demonstrating why spatial convolutional filters are necessary (`src/models/classical_baseline.py`), we construct a 60-dimensional handcrafted feature vector $\boldsymbol{\phi}(\mathbf{X}) \in \mathbb{R}^{60}$ for each patch:
1. **48-D Normalized RGB Color Histograms:** For each channel $c \in \{R, G, B\}$, we compute a 16-bin empirical density histogram $\mathbf{h}_c \in \mathbb{R}^{16}$ over $[0, 1]$ normalized such that $\|\mathbf{h}_c\|_1 = 1$.
2. **12-D Statistical & Directional Spatial Gradient Moments:** For each channel $c$, we compute the mean $\mu_c$, standard deviation $\sigma_c$, and first-order Sobel-like horizontal and vertical gradient energies:
   $$E_x^{(c)} = \frac{1}{H(W-1)} \sum_{i=1}^{H} \sum_{j=1}^{W-1} |I_{i, j+1}^{(c)} - I_{i,j}^{(c)}|, \quad E_y^{(c)} = \frac{1}{(H-1)W} \sum_{i=1}^{H-1} \sum_{j=1}^{W} |I_{i+1, j}^{(c)} - I_{i,j}^{(c)}|$$

These 60 features are classified using an ensemble of $T = 200$ CART decision trees (`RandomForestClassifier`, max depth $18$, Gini impurity splitting).

---

## 4. Deep Residual Transfer Learning & Two-Stage Optimization

### 4.1 Residual Spatial Hierarchy (`ResNet-18`)
In **ResNet-18** (`src/models/classifier.py`), each residual block computes:

$$\mathbf{h}^{(\ell+1)} = \text{ReLU}\left(\mathbf{h}^{(\ell)} + \mathcal{F}\left(\mathbf{h}^{(\ell)}; \mathbf{W}^{(\ell)}\right)\right)$$

Identity skip connections allow gradients to propagate unimpeded from the classification head back to early Gabor-like edge extractors:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(L)}} \prod_{m=\ell}^{L-1} \left(\mathbf{I} + \frac{\partial \mathcal{F}(\mathbf{h}^{(m)})}{\partial \mathbf{h}^{(m)}}\right)$$

The final convolutional feature tensor $\mathbf{A}^{(4)} \in \mathbb{R}^{512 \times 4 \times 4}$ is aggregated via Global Average Pooling ($\text{GAP}$) and passed through a regularized linear classification head with dropout rate $p = 0.30$:

$$\mathbf{z} = \mathbf{W}_{\text{fc}} \left(\text{Dropout}_{0.30}\left(\frac{1}{16}\sum_{i=1}^{4}\sum_{j=1}^{4} \mathbf{A}_{:, i, j}^{(4)}\right)\right) + \mathbf{b}_{\text{fc}} \in \mathbb{R}^{10}$$

### 4.2 Two-Stage Warmup & Full Fine-Tuning Protocol
1. **Stage 1 — Classification Head Warmup (Epochs 1–2):**
   All pretrained convolutional parameters $\theta_{\text{backbone}}$ are frozen ($\nabla_{\theta_{\text{backbone}}} \mathcal{L} = \mathbf{0}$). Only $\{\mathbf{W}_{\text{fc}}, \mathbf{b}_{\text{fc}}\}$ are updated, preventing large initial gradients from disrupting pretrained filters.
2. **Stage 2 — End-to-End Fine-Tuning (Epochs 3–8):**
   All $11,181,642$ parameters are unfrozen and optimized jointly using **AdamW** ($\eta_0 = 5 \times 10^{-4}$, decoupled weight decay $\lambda = 10^{-4}$, global gradient $L_2$-norm clipping at $2.0$) under a **Cosine Annealing** learning rate schedule:
   $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{t \pi}{T_{\max}}\right)\right), \quad \eta_{\min} = 5 \times 10^{-6}$$

### 4.3 Label-Smoothed Cross-Entropy Loss
Because overhead satellite tiles occasionally straddle transition zones (e.g., a highway bordering a forest), hard one-hot targets encourage overconfident logit saturation. We optimize **Label-Smoothed Cross-Entropy** ($\epsilon = 0.05$):

$$\mathcal{L}_{\text{LS}}(\mathbf{p}, y) = -\sum_{c=0}^{C-1} \tilde{y}_c \log(p_c), \quad \text{where } \tilde{y}_c = (1 - \epsilon)\mathbb{I}[c = y] + \frac{\epsilon}{C}$$

---

## 5. Grad-CAM & Counterfactual Spatial Attribution

To explain individual predictions (`src/inference/gradcam.py`), we extract the spatial activation maps $A^k \in \mathbb{R}^{H' \times W'}$ ($k \in \{1, \dots, 512\}$) from `layer4[-1].conv2` and compute the gradient of the pre-softmax logit $y^c$ for target class $c$ with respect to $A_{i,j}^k$:

$$\alpha_k^c = \frac{1}{H' \times W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial y^c}{\partial A_{i,j}^k}$$

The class-discriminative localization map is obtained via a ReLU-gated linear combination upsampled bilinearly to the full image resolution:

$$\mathbf{M}_{\text{Grad-CAM}}^c = \text{Normalize}_{[0,1]}\left(\text{Upsample}\left(\text{ReLU}\left(\sum_{k=1}^{512} \alpha_k^c A^k\right)\right)\right)$$

* **Standard Attribution Mode:** Sets $c = \hat{y} = \arg\max_{c'} p_{c'}$ to highlight landforms supporting the model's top-1 decision.
* **Counterfactual Attribution Mode:** Sets $c = c_{\text{query}} \neq \hat{y}$ to visualize what spatial structures in the tile resemble an alternative user-queried class.

---

## 6. Dihedral Test-Time Augmentation (TTA) & Predictive Uncertainty

### 6.1 Dihedral $D_4$ Rotational Ensemble
At inference time (`src/inference/predictor.py`), enabling Dihedral TTA evaluates the input tile across the 4 cyclic rotations $\mathcal{R} = \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$:

$$\bar{\mathbf{p}}(\mathbf{X}) = \frac{1}{4} \sum_{r \in \mathcal{R}} \text{Softmax}\left(f_\theta\left(\text{Rot}_r(\mathbf{X})\right)\right)$$

We also compute the **Rotational Agreement Ratio**:

$$\text{Agreement}(\mathbf{X}) = \frac{1}{4} \sum_{r \in \mathcal{R}} \mathbb{I}\left[\arg\max_c f_\theta(\text{Rot}_r(\mathbf{X}))_c = \arg\max_c \bar{p}_c\right]$$

### 6.2 Shannon Predictive Entropy & Top-2 Margin
To detect ambiguous boundary patches or out-of-distribution inputs, we compute the **Shannon Entropy** (in bits) of the posterior distribution $\mathbf{p}$:

$$H(\mathbf{p}) = -\sum_{c=0}^{C-1} p_c \log_2(p_c) \in \left[0, \log_2(10) \approx 3.3219\text{ bits}\right]$$

alongside the normalized uncertainty $U(\mathbf{p}) = \frac{H(\mathbf{p})}{\log_2(C)} \in [0, 1]$ and Top-2 probability margin $\Delta p = p_{(1)} - p_{(2)}$.

---

## 7. Optical Spectral Indices & Bi-Temporal Divergence

### 7.1 Optical Vegetation & Water Proxy Indices
From normalized RGB channels $R, G, B \in [0, 1]$, we compute:
* **Normalized Green-Red Difference Index (NGRDI):**
  $$\text{NGRDI} = \mathbb{E}\left[\frac{G - R}{G + R + \epsilon}\right] \in [-1, +1]$$
* **Excess Green Vegetation Index (ExG):**
  $$\text{ExG} = \mathbb{E}[2G - R - B]$$

### 7.2 Bi-Temporal Jensen-Shannon Divergence
Given posterior distributions $\mathbf{p}_{t_1}$ and $\mathbf{p}_{t_2}$ over a sector at times $t_1$ and $t_2$ (`src/inference/geospatial_analytics.py`), we quantify distributional regime shift via the bounded, symmetric **Jensen-Shannon Divergence** (base 2, in $[0, 1]$):

$$D_{\text{JS}}(\mathbf{p}_{t_1} \parallel \mathbf{p}_{t_2}) = \frac{1}{2} D_{\text{KL}}(\mathbf{p}_{t_1} \parallel \mathbf{m}) + \frac{1}{2} D_{\text{KL}}(\mathbf{p}_{t_2} \parallel \mathbf{m}), \quad \mathbf{m} = \frac{1}{2}(\mathbf{p}_{t_1} + \mathbf{p}_{t_2})$$
