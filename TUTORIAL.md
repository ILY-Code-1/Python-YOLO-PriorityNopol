← [Back to README](README.md)

# 📚 Training Tutorial — Priority Vehicle Detection

End-to-end guide for preparing the dataset, training both YOLOv8 stages, running the API, and tuning performance for the Priority Vehicle Detection project.

---

## 📑 Table of Contents

1. [Prerequisites](#-prerequisites)
2. [Dataset Preparation](#-dataset-preparation)
3. [Training Stage 1 — Vehicle Detection](#-training-stage-1--vehicle-detection)
4. [Training Stage 2 — License Plate Detection](#-training-stage-2--license-plate-detection)
5. [Running the API](#-running-the-api)
6. [Troubleshooting](#-troubleshooting)
7. [Performance Tuning](#-performance-tuning)

---

## 🛠️ Prerequisites

### Hardware

- **Minimum**: Intel Core i5 (or equivalent), 8 GB RAM. CPU-only training is supported (slow but functional — expect ~3 minutes per epoch at `imgsz=832`).
- **Recommended**: 16 GB RAM and any modern CUDA-capable GPU for a ≥10× speedup on training.

### Software

- **Python 3.11+**
- **Git**

### Packages

Install everything from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages used by training, inference, and serving:

- `ultralytics` — YOLOv8 training + inference
- `easyocr` — license plate OCR
- `fastapi` + `uvicorn` — HTTP API
- `opencv-python` — image I/O and preprocessing

---

## 📂 Dataset Preparation

### Step 1 — Dataset structure (YOLOv8 format)

```
dataset/
├── images/
│   ├── train/        # *.jpg
│   └── val/
├── labels/
│   ├── train/        # *.txt — one row per object:
│   │                 #   class_id  cx  cy  w  h  (normalized 0..1)
│   └── val/
└── dataset.yaml      # paths + class names
```

Class IDs (must match `dataset.yaml`):

| ID | Name |
|:---:|---|
| 0 | ambulance |
| 1 | police |
| 2 | fire_truck |
| 3 | license_plate |

### Step 2 — Audit the dataset

```bash
python scripts/audit_dataset.py
```

Reports per-class image and bbox counts, flags suspiciously large/small boxes, and writes `dataset/audit_report.json`.

### Step 3 — Augment (license plate synthesis)

```bash
python scripts/augment_nopol.py
```

Generates copy-paste `vehicle + plate` composites plus standard albumentations augmentations into `dataset_aug/`. This is the key step for boosting plate-in-context training samples.

### Step 4 — Merge and verify

```bash
python scripts/merge_dataset.py
python scripts/filter_huge_boxes.py
python scripts/verify_dataset.py
```

`verify_dataset.py` prints class distribution, integrity checks, split ratio, and a final **GO / NOT READY** verdict.

---

## 🚗 Training Stage 1 — Vehicle Detection

Trains `vehicle_best.pt` — the priority-vehicle classifier.

### Start

```bash
python scripts/train_vehicle.py
```

### Stop

Press **Ctrl+C** in the terminal — YOLOv8 auto-saves `last.pt` before exit.

Or kill by PID:

```cmd
taskkill /PID <pid> /F
```

Find the PID first:

```cmd
tasklist | findstr python
```

### Resume

Edit `scripts/train_vehicle.py`:

```python
RESUME = True
```

Then run again:

```bash
python scripts/train_vehicle.py
```

Training will continue from `runs/detect/runs/train/vehicle_detection/weights/last.pt`.

### Monitor progress

```cmd
type runs\detect\runs\train\vehicle_detection\results.csv
```

Per-epoch loss + mAP rows append as training proceeds. Plots are written to the same folder.

### Expected result

- `mAP50` > **0.75** on the validation split
- Best weights at `runs/detect/runs/train/vehicle_detection/weights/best.pt`
- Auto-copied to `model/vehicle_best.pt` on completion

---

## 🪪 Training Stage 2 — License Plate Detection

Trains `nopol_best.pt` — the plate localizer used on Stage 1 crops.

### Start

```bash
python scripts/train_nopol.py
```

### Stop

**Ctrl+C** in the terminal, or:

```cmd
taskkill /PID <pid> /F
```

### Resume

Set `RESUME = True` in `scripts/train_nopol.py`, then re-run:

```bash
python scripts/train_nopol.py
```

### Monitor progress

```cmd
type runs\detect\runs\train\nopol_detection\results.csv
```

### Expected result

- `mAP50` > **0.65** on validation (plates are smaller objects → lower mAP than Stage 1)
- Best weights at `runs/detect/runs/train/nopol_detection/weights/best.pt`
- Auto-copied to `model/nopol_best.pt` on completion

---

## 🌐 Running the API

### Local (development)

```bash
python -m uvicorn app.main:app --reload
```

`--reload` watches `.py` files and restarts on save — perfect for iteration.

### Docker (production)

```bash
docker build -t priority-vehicle .
docker run -p 8000:8000 priority-vehicle
```

### Test against a local server

```bash
python scripts/test_detect.py \
  --image your_image.jpg \
  --url http://localhost:8000/py-yolo-nopol
```

Expected JSON response:

```json
{
  "vehicle":        "ambulance",
  "plate_number":   "B7564FDA",
  "confidence":     0.92,
  "plate_detected": true
}
```

---

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `ImportError: ultralytics` | Package not installed | `pip install ultralytics` |
| `FileNotFoundError: model/...best.pt` | Weights missing | Run the matching training script, or place a `.pt` file in `model/` |
| Low confidence on detections | Threshold too high | Lower `VEHICLE_CONF_THRESHOLD` in `.env` |
| Plate not readable | Image quality / occluded plate | Use a clearer image; ensure the plate is visible and not heavily blurred |
| Docker `numpy` build error | Python version mismatch | Use a `python:3.11-slim` base image (already configured) |
| Server fails to start | Port already in use | Change `PORT` in `.env`, or stop the conflicting process |

---

## ⚡ Performance Tuning

### Confidence threshold guide

| Variable | Start at | Lower if... | Raise if... |
|---|:---:|---|---|
| `VEHICLE_CONF_THRESHOLD` | `0.15` | Missing valid detections (silver / side-angle vehicles) | Too many false positives on non-priority vehicles |
| `NOPOL_CONF_THRESHOLD` | `0.05` | Plates not being localized | OCR returning garbage from non-plate regions |

Tune one at a time, evaluate on `example_images/`, then commit your `.env`.

### Image quality tips

- **Minimum resolution**: **480 × 480 px**. Below this, Stage 1 starts missing smaller vehicles.
- **Plate visibility**: should be clearly readable to a human — not blurred, not severely angled.
- **Lighting**: good lighting helps both stages. EasyOCR loses accuracy in heavily low-light or backlit conditions.
- **Compression**: avoid heavily re-encoded inputs (e.g. low-bitrate messaging-app re-uploads). They introduce artifacts that hurt plate OCR.

---

← [Back to README](README.md)
