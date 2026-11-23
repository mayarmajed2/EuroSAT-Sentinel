# INSTALLATION AND SETUP GUIDE — EuroSAT-Sentinel

This guide provides step-by-step instructions to install, configure, run, and troubleshoot **EuroSAT-Sentinel** on Windows, macOS, and Linux, as well as inside **VS Code**.

---

## 1. System Prerequisites

* **Operating System:** Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+).
* **Python Version:** Python `3.11+` (tested and verified on Python `3.11`, `3.12`, and `3.14.5`).
* **Disk Space:** ~350 MB (includes ~90 MB for the official EuroSAT RGB dataset and ~45 MB for the trained ResNet-18 checkpoint).
* **Hardware Compute:**
  * **Inference & Interactive Gradio App:** Runs in real time (`<40 ms` per patch) on any standard CPU or laptop GPU.
  * **Full 8-Epoch Training Pipeline:** ~20–30 seconds total on an NVIDIA CUDA GPU (`RTX 3050+`), or ~7.5 minutes on a multi-core CPU.

---

## 2. Step-by-Step Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/mayarmajed2/EuroSAT-Sentinel.git
cd EuroSAT-Sentinel
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on Windows (CMD):
.venv\Scripts\activate.bat

# Activate on Linux / macOS:
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running the Application Locally

### A. Launch the 5-Tab Interactive Gradio Web Platform
```bash
python app/gradio_app.py
```
* The server binds to `http://127.0.0.1:7860` (or automatically selects the next available port such as `http://127.0.0.1:7861` if `7860` is already in use) and opens your default web browser automatically.

### B. Run inside VS Code (`F5` Run & Debug)
The repository includes pre-configured VS Code launch profiles in [`.vscode/launch.json`](../.vscode/launch.json):
1. Open the repository root folder in **Visual Studio Code**.
2. Press `Ctrl+Shift+D` (or `Cmd+Shift+D` on macOS) to open the **Run and Debug** panel.
3. Choose any of the 4 configurations from the dropdown and press **`F5`**:
   * **`1. Launch Gradio Interactive App (Browser Demo)`**
   * **`2. Run Phase 1 End-to-End Pipeline (EDA + Train + Eval)`**
   * **`3. Run Single-Image CLI Prediction + Grad-CAM`**
   * **`4. Run Pytest Unit Test Suite`**

### C. Run the End-to-End Training & Evaluation Pipeline via CLI
```bash
python scripts/run_phase1_pipeline.py --config configs/default_config.yaml
```
* Automatically downloads the EuroSAT RGB dataset (if not already present in `data/raw/`), generates leakage-free stratified splits in `data/splits/`, runs Exploratory Data Analysis, trains the 60-feature Classical ML Random Forest baseline, trains the two-stage ResNet-18 CNN, evaluates on the held-out test set, and saves all JSON/CSV logs and diagnostic figures to `outputs/`.

### D. Run Single-Image CLI Inference with Grad-CAM
```bash
python scripts/predict_single.py --image data/raw/eurosat/2750/River/River_1015.jpg --output outputs/figures/single_inference_gradcam.png
```

### E. Execute the Automated Test Suite
```bash
pytest tests/ -v -p no:cacheprovider
```

---

## 4. Configuration Customization

All hyperparameters and paths can be modified in [`configs/default_config.yaml`](../configs/default_config.yaml) without editing source code:
* **Switch CNN Backbone Architecture:** Change `model.architecture` between `"resnet18"`, `"mobilenet_v3_small"`, or `"custom_cnn"`.
* **Scale Dataset Size:** Set `data.max_samples_per_class: 300` (default `3,000` balanced images for fast laptop reproduction) or `null` / `3000` (all `27,000` EuroSAT images).
* **Adjust Training Schedule:** Modify `training.epochs`, `training.learning_rate`, `training.label_smoothing`, or `model.freeze_backbone_epochs`.

---

## 5. Troubleshooting

| Issue / Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **`OSError: Cannot find empty port in range: 7860-7860`** | Port `7860` is already in use or reserved by Windows Hyper-V/WSL. | Resolved automatically in `app/gradio_app.py` via `server_port=None` (Gradio auto-selects `7860`, `7861`, etc.). |
| **SSL Certificate Error during EuroSAT download** | University or corporate firewall intercepting Zenodo/DFKI HTTPS certificates. | Handled automatically in `src/data/dataset.py` via fallback SSL context and offline verification. |
| **`Checkpoint not found! Initializing untrained resnet18`** | `outputs/checkpoints/best_eurosat_model.pth` has not been generated yet. | Run `python scripts/run_phase1_pipeline.py --config configs/default_config.yaml` once to generate and save the `95.56%` Val Acc checkpoint. |
| **Pytest cache permission warning on Windows** | Windows read-only `.pytest_cache` folder attributes. | Run pytest with `-p no:cacheprovider` (already configured in `.vscode/settings.json` and `.vscode/launch.json`). |
