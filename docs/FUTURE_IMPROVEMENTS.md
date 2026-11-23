# FUTURE IMPROVEMENTS & RESEARCH ROADMAP — EuroSAT-Sentinel

While **EuroSAT-Sentinel** delivers a complete, verified Deep Learning, Explainable AI, and Geospatial Analytics platform, its modular architecture is engineered to support the following future research and engineering extensions.

---

## 1. Full 13-Band Sentinel-2 Multispectral Sensor Fusion
* **What It Does:** Expands the input tensor from 3 optical RGB bands (`B04, B03, B02`) to all 13 Copernicus Sentinel-2 spectral bands ($443\text{ nm}$ coastal aerosol to $2190\text{ nm}$ SWIR), including true Near-Infrared (`B08`) and Red-Edge (`B05–B07`).
* **Why It Matters:** Our error analysis (`outputs/metrics/test_evaluation_metrics.json`) shows that 3 of the 17 test errors involve distinguishing `PermanentCrop` (orchards/vineyards) from `HerbaceousVegetation` or `AnnualCrop`. True NIR reflectance captures cellular leaf structure (true $\text{NDVI} = \frac{\text{B08}-\text{B04}}{\text{B08}+\text{B04}}$), resolving subtle vegetative phenology differences.
* **Implementation Path:** Replace `backbone.conv1 = nn.Conv2d(3, 64, ...)` with a 13-channel convolution initialized via channel-mean weight inflation from pretrained ImageNet weights.

---

## 2. Vision Transformer (Swin-T / ViT) & ConvNeXt Multi-Backbone Benchmark
* **What It Does:** Benchmarks hierarchical Vision Transformers (`Swin-Tiny`, `ViT-Tiny`) and modernized convolutional networks (`ConvNeXt-Tiny`, `EfficientNet-B0`) against our existing `ResNet-18`, `MobileNetV3-Small`, and `CustomEuroSATCNN` models in `src/models/classifier.py`.
* **Why It Matters:** Evaluates whether global self-attention mechanisms better capture long-range linear corridors (`River` vs. `Highway`) compared to local $3 \times 3$ convolutional receptive fields.
* **Measurable Target:** Improve held-out Top-1 Test Accuracy from `96.22%` to $\ge 98.5\%$ on the full 27,000-image benchmark while comparing Expected Calibration Error (ECE) and CPU/GPU latency (`ms/patch`).

---

## 3. Pixel-Level Semantic Segmentation (U-Net / SegFormer) for Mixed Boundary Tiles
* **What It Does:** Upgrades patch-level scene classification ($1$ label per $64 \times 64$ tile) to pixel-level semantic segmentation ($64 \times 64$ dense class mask) or multi-label classification.
* **Why It Matters:** Eliminates the single-label ambiguity on boundary tiles that contain both an `Industrial` facility and an adjacent `Residential` neighborhood or a `Highway` crossing a `River`.

---

## 4. GeoTIFF Coordinate Ingestion & Automated Copernicus Data Space API Sync
* **What It Does:** Connects the Bi-Temporal Change Simulator (Tab 2) and Regional Mosaic Scanner (Tab 3) directly to the **Copernicus Data Space Ecosystem (Sentinel Hub / STAC API)**, allowing users to enter a latitude/longitude bounding box and date range ($t_1, t_2$) to fetch live Sentinel-2 Level-2A GeoTIFFs automatically.
* **Why It Matters:** Enables live operational monitoring of real-world conservation reserves and municipal greenbelts with georeferenced GeoJSON/Shapefile polygon exports for QGIS and ArcGIS.

---

## 5. ONNX Runtime Export & INT8 Edge Quantization
* **What It Does:** Exports `best_eurosat_model.pth` to **ONNX** (`torch.onnx.export`) with dynamic **INT8 Quantization** (`onnxruntime.quantization`).
* **Why It Matters:** Reduces model checkpoint size from `44.8 MB` (FP32) to `~11.2 MB` (INT8) and accelerates CPU batch scanning of large regional mosaics by $3\times\text{–}4\times$ for onboard satellite or UAV edge processors.
