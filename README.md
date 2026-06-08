# 🚨 Priority Vehicle Detection

Real-time detection of priority vehicles (ambulance, police, fire truck) and OCR of their license plates — powered by a 2-stage YOLOv8n pipeline + EasyOCR, served via FastAPI.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Demo](docs/demo.gif)

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Detection Results](#-detection-results)
3. [Quick Start](#-quick-start)
4. [API Documentation](#-api-documentation)
5. [Configuration](#-configuration)
6. [Example Images](#-example-images)
7. [Training Guide](#-training-guide)
8. [Known Limitations](#-known-limitations)
9. [Contributing & License](#-contributing--license)

---

## 🔭 Overview

This service classifies the vehicle type (ambulance / police / fire truck), localizes the license plate inside the detected vehicle crop, then reads the plate text with EasyOCR. Every request returns a single, mobile-friendly JSON object — no images, no base64, no extra bandwidth.

It is designed for **research and traffic-management pilots** where a downstream system needs an authoritative "this is a priority vehicle and here is its plate number" signal from a still image.

### 2-Stage + OCR Pipeline

```
   ┌──────────────────┐
   │  Image input     │
   │  (JPEG / PNG)    │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────────────────┐
   │  Stage 1                     │
   │  vehicle_best.pt (YOLOv8n)   │
   │  → ambulance / police /      │
   │    fire_truck  + bbox        │
   └────────┬─────────────────────┘
            │ crop vehicle (+padding)
            ▼
   ┌──────────────────────────────┐
   │  Stage 2                     │
   │  nopol_best.pt   (YOLOv8n)   │
   │  → plate bbox (in crop)      │
   └────────┬─────────────────────┘
            │ crop plate (+padding)
            ▼
   ┌──────────────────────────────┐
   │  Stage 3                     │
   │  EasyOCR                     │
   │  → plate text "B7564FDA"     │
   └────────┬─────────────────────┘
            │
            ▼
   ┌──────────────────────────────┐
   │  JSON response               │
   └──────────────────────────────┘
```

---

## 🎯 Detection Results

| Vehicle | Detection | Plate Reading |
|---|---|---|
| ambulance | ✅ Good | ✅ Good |
| police | ✅ Good | ✅ Good |
| fire_truck | ✅ Good | ⚠️ Limited (small plate region) |
| ambulance (silver / side-angle) | ⚠️ Limited | ⚠️ Limited |

Example response for a clean ambulance hit:

```json
{
  "vehicle":        "ambulance",
  "plate_number":   "B7564FDA",
  "confidence":     0.92,
  "plate_detected": true
}
```

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.11+**
- **pip**
- (Optional) **Docker** — for one-command deployment

### Installation

```bash
git clone https://github.com/<your-org>/Python-YOLO-PriorityNopol.git
cd Python-YOLO-PriorityNopol

# 1. Create + activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env template and edit if needed
cp .env.example .env
```

### Run with Docker (recommended)

```bash
docker build -t priority-vehicle .
docker run -p 8000:8000 priority-vehicle
```

### Run locally

```bash
python -m uvicorn app.main:app --reload
```

Server is ready when you see `Application startup complete.` — health check:

```bash
curl http://localhost:8000/health
```

---

## 📡 API Documentation

### Endpoint

```
POST /py-yolo-nopol/api/v1/detect
Content-Type: multipart/form-data
```

### Request

| Field | Type | Required | Description |
|---|---|:---:|---|
| `file` | file (image) | ✅ | JPEG or PNG of the vehicle |

### Curl example

```bash
curl -X POST http://localhost:8000/py-yolo-nopol/api/v1/detect \
  -F "file=@your_image.jpg"
```

### Response format

```json
{
  "vehicle":        "ambulance" | "police" | "fire_truck" | null,
  "plate_number":   "B1234XYZ" | "",
  "confidence":     0.92 | 0,
  "plate_detected": true | false
}
```

### Response examples

**Detected (vehicle + plate):**

```json
{
  "vehicle":        "ambulance",
  "plate_number":   "B7564FDA",
  "confidence":     0.9234,
  "plate_detected": true
}
```

**No priority vehicle detected:**

```json
{
  "vehicle":        null,
  "plate_number":   "",
  "confidence":     0,
  "plate_detected": false
}
```

**Vehicle detected, plate not legible:**

```json
{
  "vehicle":        "fire_truck",
  "plate_number":   "",
  "confidence":     0.8044,
  "plate_detected": true
}
```

---

## ⚙️ Configuration

All settings come from environment variables. Copy `.env.example` to `.env` and edit.

| Variable | Description | Default |
|---|---|---|
| `VEHICLE_MODEL_PATH` | Path to Stage 1 weights (`.pt`) | `model/vehicle_best.pt` |
| `NOPOL_MODEL_PATH` | Path to Stage 2 weights (`.pt`) | `model/nopol_best.pt` |
| `VEHICLE_CONF_THRESHOLD` | Stage 1 minimum confidence (0.0 – 1.0) | `0.05` |
| `NOPOL_CONF_THRESHOLD` | Stage 2 minimum confidence (0.0 – 1.0) | `0.05` |
| `VEHICLE_CROP_PADDING` | Padding (px) around vehicle bbox before Stage 2 | `15` |
| `PLATE_CROP_PADDING` | Padding (px) around plate bbox before OCR | `4` |
| `OCR_LANG` | EasyOCR language code | `en` |
| `OCR_GPU` | Use GPU for OCR (`true` / `false`) | `false` |
| `LOG_LEVEL` | Logger level (`debug` / `info` / `warning` / `error`) | `info` |
| `PORT` | HTTP port the server binds to | `8000` |

---

## 🖼️ Example Images

The `example_images/` folder contains **curated detection examples per class**, plus debug copies with bounding boxes drawn at `example_images/debug/{class}/`. Use these to verify your local install matches the reference behavior.

Naming convention: `{class_name}_{plate_number}.jpg` — e.g. `police_4212XXI.jpg`.

📖 See [TUTORIAL.md](TUTORIAL.md) for the full training guide.

---

## 🧠 Training Guide

- **Stage 1 — Vehicle Detection**: trains `vehicle_best.pt` on the labeled `dataset/` (ambulance, police, fire_truck, license_plate). YOLOv8n + augmentations for HSV, scale, fliplr, and copy-paste.
- **Stage 2 — License Plate Detection**: trains `nopol_best.pt` on cropped vehicles to localize the plate. Benefits from `scripts/augment_nopol.py` copy-paste synthesis.
- **Resume support**: both training scripts honor `RESUME = True` to continue from `last.pt`.
- **Monitoring**: progress logged to `runs/detect/runs/train/<exp>/results.csv` and visualized as `train_batch*.jpg` in the same folder.

📖 [Full Training Guide →](TUTORIAL.md)

---

## ⚠️ Known Limitations

- **Silver / side-angle ambulance** — Stage 1 has low recall for this appearance (detection confidence can drop below 0.03). Augmentation alone won't close the gap; retraining with more silver/side-angle examples is the real fix.
- **Fire truck plates** — the plate region is small relative to the truck (often < 150×50 px in source). Stage 2 may localize it correctly but EasyOCR can fail to read at that resolution. Higher input `imgsz` during training mitigates this.
- **OCR accuracy** — depends on plate visibility, lighting, motion blur, and angle. The pipeline includes a digit-token filter and a strict-format validator, but a plausibly-shaped misread can still pass — manual verification is recommended for downstream actions.

---

## 🤝 Contributing & License

Pull requests are welcome. For larger changes, please open an issue first to discuss what you'd like to change. Quick checklist:

1. Fork and create a feature branch.
2. Add tests / sample images where applicable.
3. Run `python scripts/verify_dataset.py` if you touched the dataset.
4. Open a PR with a clear description.

Licensed under the **MIT License** — see `LICENSE` for full text.

---

📖 **[View Full Training Tutorial →](TUTORIAL.md)**
