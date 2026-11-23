# PROFESSOR LIVE DEMONSTRATION GUIDE (5–10 Minute Presentation)

**Project:** **EuroSAT-Sentinel** — Explainable Deep Learning for Satellite Land-Use & Environmental Monitoring  
**Launch Command:** `python app/gradio_app.py` (or press **`F5`** in VS Code)

---

## Pre-Presentation Checklist (30 Seconds Before Starting)

1. Open the project in **VS Code** and press **`F5`** (`1. Launch Gradio Interactive App (Browser Demo)`), or run:
   ```bash
   python app/gradio_app.py
   ```
2. Keep the browser open to the **EuroSAT-Sentinel Geospatial Deep Learning Platform** (`http://127.0.0.1:7860`).
3. Keep a VS Code terminal open showing `pytest tests/ -v -p no:cacheprovider` (`19 passed`).

---

## Recommended 7-Minute Demonstration Sequence

### Minute 0:00 – 1:30 | 1. Scientific Motivation & Classical ML vs. Deep Learning Proof
* **What to Show:** Point to the **Executive KPI Header** at the top of the Gradio app and **Tab 5 (`Professor Live Demo Mode & Academic Methodology`)**.
* **Key Talking Points:**
  * *"We built EuroSAT-Sentinel to automate Land-Use and Land-Cover (LULC) monitoring on Copernicus Sentinel-2 optical satellite imagery ($64 \times 64$ tiles at $10\text{ m/pixel}$, where each patch covers $40.96\text{ hectares}$)."*
  * *"Why is Deep Learning scientifically necessary here? In overhead imagery, distinct land-use classes share nearly identical global RGB color histograms: `River` and `SeaLake` are both dark water, while `Highway`, `Industrial`, and `Residential` are all grey asphalt/concrete. They differ in **2D spatial morphology**."*
  * *"To prove this empirically on identical leakage-free stratified splits (`2,100` Train / `450` Val / `450` Held-Out Test), we first built a **60-feature Classical ML Random Forest** using RGB color histograms and directional spatial gradients. It plateaus at **`76.67%` accuracy**. Our two-stage fine-tuned **ResNet-18 CNN** reaches **`96.22%` Top-1 Accuracy** and **`100.00%` Top-3 Accuracy**—a **`+19.55%` gain** that cuts classification errors by **83.8%**."*

---

### Minute 1:30 – 3:15 | 2. Single-Patch XAI, Counterfactual Grad-CAM & Spectral-Uncertainty Diagnostics (Tab 1)
* **What to Do:**
  1. Switch to **Tab 1 (`🛰️ 1. Single-Patch XAI & Counterfactual Inspector`)**.
  2. Click the **`River`** or **`Highway`** example tile in the bottom examples table.
  3. Point out the **Grad-CAM Spatial Attention Overlay** and the **Tri-Panel Scientific Diagnostic Plot**.
  4. Change the **Grad-CAM Attribution Target** dropdown from `Auto (Top-1 Predicted Class)` to a counterfactual class (e.g., `Forest` or `Residential`) and click **`🔬 Analyze Satellite Patch`**.
* **Key Talking Points:**
  * *"We don't treat the CNN as a black box. By attaching forward and backward hooks to `layer4[-1].conv2` on $128 \times 128$ upsampled inputs, our Grad-CAM engine localizes the exact river bend or highway corridor."*
  * *"We also implemented **Counterfactual Visual Attribution**: we can query the network for any alternative class to see which background pixels weakly activate for that hypothesis."*
  * *"In the Tri-Panel Diagnostic Plot below, the system computes live optical remote-sensing indices (**NGRDI** and **Excess Green ExG**), Grad-CAM activation contour isolines, **$4\times$ Dihedral Rotational Test-Time Augmentation ($0^\circ, 90^\circ, 180^\circ, 270^\circ$)** consensus, and **Shannon Predictive Entropy** ($H(\mathbf{p})$ in bits) to quantify prediction reliability."*

---

### Minute 3:15 – 4:45 | 3. Bi-Temporal Deforestation & Urban Sprawl Change Simulator (Tab 2)
* **What to Do:**
  1. Switch to **Tab 2 (`🌍 2. Bi-Temporal Change & Deforestation Simulator (t1 → t2)`)**.
  2. Select **`1. Deforestation Alert: Dense Forest (t1) → Annual Cropland (t2)`** and click **`📂 Load Scenario Preset Patches`**.
  3. Then select **`2. Urban Sprawl Warning: Managed Pasture (t1) → Residential Settlement (t2)`** and click **`📂 Load Scenario Preset Patches`**.
* **Key Talking Points:**
  * *"In real environmental monitoring, agencies compare satellite passes over time ($t_1 \rightarrow t_2$)."*
  * *"Here, comparing a baseline `Forest` acquisition ($t_1$) against an `AnnualCrop` conversion ($t_2$), the system automatically triggers a **🔴 CRITICAL DEFORESTATION ALERT**, computes the pixel-level Euclidean spectral shift map, measures the posterior **Jensen-Shannon Divergence**, and estimates an above-ground carbon pool loss of **`-5,447.7 tonnes C`** across the $40.96\text{ ha}$ sector."*

---

### Minute 4:45 – 6:00 | 4. Regional Mosaic Scanner & Semantic Land-Use Cartography (Tab 3)
* **What to Do:**
  1. Switch to **Tab 3 (`🗺️ 3. Regional Mosaic Scanner & Land-Use Cartography`)**.
  2. Select **`1. Deforestation & Agricultural Frontier (3x3 — 368.6 ha)`** (or **`2. Metropolitan Urban-Industrial Expansion Corridor`**) and click **`🛰️ Assemble & Scan Regional Preset Mosaic`**.
* **Key Talking Points:**
  * *"To scale beyond single tiles, Tab 3 implements a sliding-window regional mosaic scanner. It partitions a $368.64\text{ ha}$ multi-tile satellite scene into sectors (`R1C1` through `R3C3`), classifies every sector, renders a color-coded **Semantic Land-Use Thematic Map**, compiles a regional hectare and carbon stock census table, and exports a downloadable CSV report."*

---

### Minute 6:00 – 7:00 | 5. Interactive Test-Set Failure Diagnosis & Automated QA Verification (Tab 4)
* **What to Do:**
  1. Switch to **Tab 4 (`📊 4. Benchmark Explorer & Failure Case Diagnosis`)**.
  2. Keep the filter on **`Misclassified / Hard Cases Only (17)`** and **click any row** in the table.
* **Key Talking Points:**
  * *"Academic rigor requires examining where a model fails. Out of 450 held-out test images, our model made only 17 top-1 errors (`96.22%` Top-1, `100.00%` Top-3)."*
  * *"When we click any of those 17 tiles right here in Tab 4, live Grad-CAM diagnosis shows why: they are genuine mixed-boundary tiles (such as a highway bridge crossing a river, or industrial warehouses bordering residential buildings). Even on these 17 cases, the true label is 100% present in the Top-3 predictions, and our Shannon entropy metric flags them for human review."*
  * *"Finally, our repository includes **19 automated Pytest unit and integration tests** verifying zero data leakage, all three CNN backbones, Grad-CAM colormaps, TTA, and geospatial analytics."*
