# Data Directory Structure & Instructions

In compliance with software engineering best practices and `.gitignore` safeguards, raw satellite imagery (`data/raw/`) is not committed to Git history.

## Automatic Download & Reproducible Split Generation

When you execute the Phase 1 pipeline:

```bash
python scripts/run_phase1_pipeline.py --config configs/default_config.yaml
```

1. **Automated Download:** `src/data/dataset.py` automatically downloads and verifies the official **EuroSAT RGB** benchmark (~90 MB, 27,000 Sentinel-2 patches across 10 land-use classes) via `torchvision.datasets.EuroSAT` into `data/raw/`.
2. **Stratified Split Manifests:** Deterministic, zero-leakage stratified split CSV manifests (`train.csv` [70%], `val.csv` [15%], and `test.csv` [15%]) are saved in `data/splits/` and tracked in Git for full experimental reproducibility.
