# Dokumentasi Teknis: Deteksi Otomatis Kendaraan Prioritas Menggunakan YOLOv8n untuk Pencatatan Nomor Polisi

> **Jenis Dokumen:** Panduan Implementasi Sistem Backend Penelitian
> **Stack Teknologi:** FastAPI · PyTorch · YOLOv8n (Ultralytics) · EasyOCR
> **Platform:** Python 3.10+

---

## Daftar Isi

1. [Pendahuluan](#1-pendahuluan)
2. [Arsitektur Sistem](#2-arsitektur-sistem)
3. [Cara Kerja API Backend](#3-cara-kerja-api-backend)
4. [Persiapan Environment](#4-persiapan-environment)
5. [Struktur Project](#5-struktur-project)
6. [Persiapan Dataset](#6-persiapan-dataset)
7. [Cara Mengumpulkan Dataset](#7-cara-mengumpulkan-dataset)
8. [Cara Anotasi Dataset](#8-cara-anotasi-dataset)
9. [Proses Training Model YOLOv8n](#9-proses-training-model-yolov8n)
10. [Menjalankan API Backend](#10-menjalankan-api-backend)
11. [Pengujian Endpoint /api/v1/detect](#11-pengujian-endpoint-apiv1detect)
12. [Contoh Request dan Response](#12-contoh-request-dan-response)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Pendahuluan

### 1.1 Latar Belakang

Kendaraan prioritas seperti ambulans, kendaraan kepolisian, dan mobil pemadam kebakaran memerlukan akses cepat di jalan raya untuk menjalankan tugasnya. Keterlambatan akses kendaraan prioritas dapat berdampak langsung pada keselamatan jiwa. Oleh karena itu, dibutuhkan sistem yang mampu mendeteksi kehadiran kendaraan prioritas secara otomatis dan mencatat identitasnya melalui pembacaan nomor polisi.

Dokumen ini merupakan panduan teknis lengkap untuk membangun, melatih, dan menjalankan sistem backend deteksi otomatis kendaraan prioritas menggunakan:

- **YOLOv8n** – model deteksi objek real-time berbasis deep learning
- **EasyOCR** – pustaka OCR (Optical Character Recognition) untuk membaca teks plat nomor
- **FastAPI** – framework backend Python yang ringan dan berkinerja tinggi

### 1.2 Tujuan Sistem

| No | Tujuan |
|----|--------|
| 1 | Mendeteksi kendaraan prioritas (ambulance, police, fire_truck) secara otomatis dari gambar |
| 2 | Mengidentifikasi dan membaca nomor polisi kendaraan yang terdeteksi |
| 3 | Menyediakan REST API yang dapat dikonsumsi oleh aplikasi mobile (Flutter) |
| 4 | Menghasilkan respons JSON ringan yang cocok untuk integrasi sistem |

### 1.3 Kelas Objek yang Dideteksi

| Class ID | Nama Kelas | Deskripsi |
|----------|------------|-----------|
| 0 | `ambulance` | Kendaraan ambulans |
| 1 | `police` | Kendaraan kepolisian |
| 2 | `fire_truck` | Kendaraan pemadam kebakaran |
| 3 | `license_plate` | Plat nomor kendaraan |

---

## 2. Arsitektur Sistem

### 2.1 Gambaran Umum

Sistem ini dirancang sebagai backend API yang berdiri sendiri dan dapat diintegrasikan dengan aplikasi frontend maupun mobile. Berikut gambaran arsitektur sistem secara keseluruhan:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ARSITEKTUR SISTEM                           │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐         HTTP POST          ┌────────────────────┐
  │              │   /api/v1/detect           │                    │
  │  Flutter App │ ─────────────────────────► │   FastAPI Server   │
  │  (Mobile)    │   multipart/form-data      │   (Backend API)    │
  │              │ ◄───────────────────────── │                    │
  └──────────────┘       JSON Response        └────────┬───────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │                         │
                                          │   YOLOv8n Inference     │
                                          │   (Ultralytics)         │
                                          │                         │
                                          │  Deteksi:               │
                                          │  ✓ ambulance            │
                                          │  ✓ police               │
                                          │  ✓ fire_truck           │
                                          │  ✓ license_plate        │
                                          │                         │
                                          └────────────┬────────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │                         │
                                          │   Seleksi Objek         │
                                          │   - Pilih 1 kendaraan   │
                                          │     (conf tertinggi)    │
                                          │   - Pilih plat terdekat │
                                          │   - Crop area plat      │
                                          │                         │
                                          └────────────┬────────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │                         │
                                          │   EasyOCR               │
                                          │   - Preprocess gambar   │
                                          │   - Baca teks plat      │
                                          │   - Clean & format teks │
                                          │                         │
                                          └────────────┬────────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │                         │
                                          │   JSON Response         │
                                          │   {                     │
                                          │     "vehicle": "...",   │
                                          │     "plate_number":".." │
                                          │     "confidence": 0.92  │
                                          │   }                     │
                                          │                         │
                                          └─────────────────────────┘
```

### 2.2 Lapisan Arsitektur Backend

```
app/
├── routes/        ← Lapisan Presentasi  : terima request, kirim response
├── services/      ← Lapisan Bisnis      : logika deteksi & OCR
└── utils/         ← Lapisan Utilitas    : fungsi bantu pemrosesan gambar
```

**Prinsip yang diterapkan:**
- **Separation of Concerns** – setiap layer memiliki tanggung jawab yang berbeda
- **Singleton Pattern** – model YOLO dan EasyOCR di-load sekali saat startup
- **Stateless API** – setiap request berdiri sendiri, tidak bergantung state sebelumnya

---

## 3. Cara Kerja API Backend

### 3.1 Alur Pemrosesan Request

Berikut alur lengkap dari saat gambar diterima hingga respons JSON dikembalikan:

```
[1] CLIENT
    │
    │  POST /api/v1/detect
    │  Content-Type: multipart/form-data
    │  Body: file = <gambar.jpg>
    │
    ▼
[2] VALIDASI FILE
    │  ✓ Cek tipe MIME (JPEG/PNG)
    │  ✓ Decode bytes → NumPy array BGR
    │
    ▼
[3] YOLO INFERENCE
    │  ✓ Load model/best.pt (sudah di-cache)
    │  ✓ Jalankan prediksi pada gambar
    │  ✓ Dapatkan semua bounding box + class + confidence
    │
    ▼
[4] SELEKSI KENDARAAN UTAMA
    │  ✓ Filter hanya class 0, 1, 2 (kendaraan prioritas)
    │  ✓ Pilih 1 objek dengan confidence tertinggi
    │  ✗ Jika tidak ada → return response kosong
    │
    ▼
[5] SELEKSI PLAT NOMOR
    │  ✓ Filter class 3 (license_plate)
    │  ✓ Jika >1 plat → pilih yang center-nya
    │    paling dekat dengan kendaraan utama
    │  ✓ Crop area plat dari gambar asli
    │
    ▼
[6] OCR PROCESSING
    │  ✓ Preprocess: resize → grayscale → CLAHE → Otsu
    │  ✓ EasyOCR baca teks
    │  ✓ Bersihkan: hapus karakter non-alfanumerik
    │  ✓ Format: uppercase, tanpa spasi → "B1234XYZ"
    │
    ▼
[7] JSON RESPONSE
       {
         "vehicle": "ambulance",
         "plate_number": "B1234XYZ",
         "confidence": 0.9512,
         "plate_detected": true
       }
```

### 3.2 Service Layer

| Service | File | Tanggung Jawab |
|---------|------|----------------|
| `YOLOService` | `app/services/yolo_service.py` | Inference YOLO, seleksi kendaraan & plat |
| `OCRService` | `app/services/ocr_service.py` | Preprocess, OCR, clean teks plat |

---

## 4. Persiapan Environment

### 4.1 Persyaratan Sistem

| Komponen | Minimum | Rekomendasi |
|----------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 8 GB |
| Storage | 5 GB | 10 GB |
| GPU | Tidak wajib | NVIDIA CUDA 11.8+ |
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |

### 4.2 Instalasi Python

Unduh Python dari situs resmi: https://www.python.org/downloads/

Verifikasi instalasi:

```bash
python --version
# Output: Python 3.11.x
```

### 4.3 Membuat Virtual Environment

Virtual environment diperlukan agar dependensi project tidak bertabrakan dengan instalasi Python global.

```bash
# Buat virtual environment
python -m venv venv
```

**Aktivasi virtual environment:**

```bash
# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

Tanda bahwa virtual environment aktif: nama `(venv)` akan muncul di awal baris terminal.

```
(venv) C:\Users\YusnarSetiyadi\project>
```

### 4.4 Instalasi Dependensi

```bash
pip install -r requirements.txt
```

Daftar dependensi utama yang akan terinstal:

```
fastapi>=0.111.0          ← Framework API
uvicorn[standard]>=0.29.0 ← ASGI server
ultralytics>=8.2.0        ← YOLOv8n
easyocr>=1.7.1            ← OCR plat nomor
opencv-python>=4.9.0.80   ← Pemrosesan gambar
numpy>=1.26.0             ← Operasi array
python-multipart>=0.0.9   ← Parsing file upload
torch>=2.2.0              ← Deep learning backend
torchvision>=0.17.0       ← Utilitas vision
```

> **Catatan GPU:** Jika menggunakan GPU NVIDIA, instal PyTorch versi CUDA terlebih dahulu sebelum requirements.txt:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4.5 Verifikasi Instalasi

```bash
python -c "import ultralytics; ultralytics.checks()"
python -c "import easyocr; print('EasyOCR OK')"
python -c "import fastapi; print('FastAPI OK')"
```

---

## 5. Struktur Project

```
Python-YOLO-PriorityNopol/
│
├── app/                              ← Aplikasi FastAPI
│   ├── __init__.py
│   ├── main.py                       ← Entry point: setup app, CORS, router
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   └── detect.py                 ← Endpoint POST /api/v1/detect
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── yolo_service.py           ← Load YOLO, inference, seleksi objek
│   │   └── ocr_service.py            ← Preprocess, EasyOCR, clean teks
│   │
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py            ← decode_image, crop_region, resize
│
├── model/
│   └── best.pt                       ← Model hasil training (di-generate)
│
├── dataset/
│   ├── images/
│   │   ├── train/                    ← Gambar training (.jpg / .png)
│   │   └── val/                      ← Gambar validasi (.jpg / .png)
│   ├── labels/
│   │   ├── train/                    ← Label YOLO training (.txt)
│   │   └── val/                      ← Label YOLO validasi (.txt)
│   └── dataset.yaml                  ← Konfigurasi dataset
│
├── scripts/
│   ├── train.py                      ← Script training model
│   └── predict.py                    ← Script testing inferensi standalone
│
├── runs/                             ← Hasil training (auto-generate)
│   └── train/
│       └── priority_vehicle_detection/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           └── results.png
│
├── requirements.txt
├── README.md
└── readme_tutorial.md                ← File dokumentasi ini
```

**Penjelasan file kunci:**

| File | Fungsi |
|------|--------|
| `app/main.py` | Inisialisasi FastAPI, konfigurasi CORS, daftarkan router |
| `app/routes/detect.py` | Thin layer: orkestrasi request → service → response |
| `app/services/yolo_service.py` | Semua logika YOLO: load model, inference, seleksi |
| `app/services/ocr_service.py` | Semua logika OCR: preprocess, baca, bersihkan teks |
| `app/utils/image_utils.py` | Fungsi bantu: decode bytes, crop region gambar |
| `scripts/train.py` | Training pipeline lengkap dengan auto-copy best.pt |
| `dataset/dataset.yaml` | Konfigurasi dataset: path, jumlah kelas, nama kelas |

---

## 6. Persiapan Dataset

### 6.1 Jumlah Dataset yang Disarankan

| Kelas | Minimum | Disarankan | Ideal |
|-------|---------|------------|-------|
| ambulance | 100 | 300 | 500+ |
| police | 100 | 300 | 500+ |
| fire_truck | 100 | 300 | 500+ |
| license_plate | 150 | 400 | 600+ |
| **Total** | **450** | **1.300** | **2.100+** |

> **Catatan Penelitian:** Untuk hasil mAP (mean Average Precision) yang baik di atas 70%, disarankan menggunakan minimal **500 gambar** total dengan distribusi kelas yang seimbang.

### 6.2 Pembagian Data Training dan Validasi

Gunakan rasio **80% training : 20% validasi**.

Contoh untuk 500 gambar total:
- `dataset/images/train/` → 400 gambar
- `dataset/images/val/` → 100 gambar

### 6.3 Struktur Folder Dataset

```
dataset/
├── images/
│   ├── train/
│   │   ├── ambulance_001.jpg
│   │   ├── ambulance_002.jpg
│   │   ├── police_001.jpg
│   │   └── ...
│   └── val/
│       ├── ambulance_val_001.jpg
│       ├── police_val_001.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── ambulance_001.txt    ← nama file sama dengan gambar
│   │   ├── ambulance_002.txt
│   │   ├── police_001.txt
│   │   └── ...
│   └── val/
│       ├── ambulance_val_001.txt
│       ├── police_val_001.txt
│       └── ...
└── dataset.yaml
```

> **Penting:** Nama file gambar dan file label harus **sama persis** (hanya berbeda ekstensi). YOLO secara otomatis mencari file `.txt` yang berpadanan dengan setiap file gambar.

### 6.4 Format Label YOLO

Setiap file `.txt` berisi satu atau lebih baris, satu baris per objek:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Ketentuan nilai koordinat:**
- Semua nilai berupa angka desimal dalam rentang **0.0 hingga 1.0**
- Koordinat **dinormalisasi** terhadap ukuran gambar (bukan piksel absolut)
- `x_center` dan `y_center` adalah titik tengah bounding box
- `width` dan `height` adalah lebar dan tinggi bounding box

**Contoh perhitungan:**

Untuk gambar berukuran **640 × 480 piksel**, sebuah ambulans berada di koordinat piksel `(x1=100, y1=80, x2=500, y2=380)`:

```
x_center = (100 + 500) / 2 / 640 = 300 / 640 = 0.4688
y_center = (80 + 380) / 2 / 480  = 230 / 480 = 0.4792
width    = (500 - 100) / 640      = 400 / 640 = 0.6250
height   = (380 - 80) / 480       = 300 / 480 = 0.6250
```

Label yang dituliskan di file `.txt`:
```
0 0.4688 0.4792 0.6250 0.6250
```

### 6.5 Contoh File Label Lengkap

**File:** `dataset/labels/train/ambulance_001.txt`

```
0 0.5134 0.4821 0.6230 0.5940
3 0.5020 0.7812 0.2100 0.0850
```

- Baris 1: Kelas `0` (ambulance) dengan bounding box di tengah gambar
- Baris 2: Kelas `3` (license_plate) di bagian bawah kendaraan

**File:** `dataset/labels/train/police_001.txt`

```
1 0.4800 0.5000 0.7200 0.6800
3 0.4750 0.8100 0.1800 0.0750
```

### 6.6 Isi File dataset.yaml

File ini adalah konfigurasi yang dibaca oleh YOLOv8 saat proses training.

```yaml
# dataset/dataset.yaml

# Path root dataset (relatif dari direktori project)
path: dataset

# Sub-path untuk data training dan validasi
train: images/train
val:   images/val

# Jumlah kelas
nc: 4

# Nama kelas (urutan harus sesuai dengan class_id di file label)
names:
  0: ambulance
  1: police
  2: fire_truck
  3: license_plate
```

---

## 7. Cara Mengumpulkan Dataset

### 7.1 Sumber Dataset

#### A. Google Images

Pencarian gambar di Google Images merupakan cara termudah untuk mengumpulkan dataset awal.

**Kata kunci yang disarankan:**

```
Kendaraan Ambulans:
- ambulance indonesia
- ambulance 118 indonesia
- mobil ambulans puskesmas

Kendaraan Polisi:
- mobil polisi indonesia
- patroli polisi indonesia
- patwal polisi

Kendaraan Pemadam Kebakaran:
- fire truck indonesia
- mobil pemadam kebakaran
- damkar jakarta

Plat Nomor:
- plat nomor indonesia
- nomor polisi kendaraan
- license plate indonesia
```

**Cara mengunduh secara massal:**

Gunakan ekstensi browser seperti **Image Downloader** (Chrome/Firefox) untuk mengunduh banyak gambar sekaligus dari hasil pencarian.

#### B. Ekstraksi Frame dari Video YouTube

Video YouTube merupakan sumber gambar yang kaya karena satu video dapat menghasilkan ratusan frame berbeda.

```python
# Script untuk ekstraksi frame dari video
# Simpan sebagai scripts/extract_frames.py

import cv2
import os

def extract_frames(video_path: str, output_dir: str, interval: int = 30):
    """
    Ekstrak frame dari video setiap N frame.

    Args:
        video_path : Path ke file video (.mp4)
        output_dir : Folder output untuk menyimpan frame
        interval   : Ambil 1 frame setiap N frame (default 30 = ~1 detik)
    """
    os.makedirs(output_dir, exist_ok=True)
    cap   = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"Selesai: {saved} frame disimpan ke {output_dir}")

if __name__ == "__main__":
    extract_frames(
        video_path="video_ambulance.mp4",
        output_dir="dataset/raw/ambulance",
        interval=30,
    )
```

Cara menjalankan:

```bash
python scripts/extract_frames.py
```

#### C. Dataset Publik

Beberapa platform menyediakan dataset kendaraan prioritas yang sudah berlabel:

| Platform | URL | Keterangan |
|----------|-----|------------|
| Roboflow Universe | roboflow.com/universe | Banyak dataset YOLO siap pakai |
| Kaggle | kaggle.com/datasets | Dataset beragam, termasuk kendaraan |
| Open Images Dataset | storage.googleapis.com/openimages | Dataset Google skala besar |

**Cara download dari Roboflow:**

```bash
pip install roboflow

python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_API_KEY')
project = rf.workspace('workspace-name').project('project-name')
dataset = project.version(1).download('yolov8')
"
```

### 7.2 Tips Kualitas Dataset

- **Variasi sudut pandang:** Ambil gambar dari depan, samping, dan belakang kendaraan
- **Variasi pencahayaan:** Siang hari, malam hari, kondisi hujan, kondisi mendung
- **Variasi jarak:** Dekat, sedang, dan jauh dari kamera
- **Variasi kondisi:** Kendaraan bergerak dan berhenti
- **Hindari duplikasi:** Jangan gunakan gambar yang hampir identik (frame video terlalu berdekatan)

---

## 8. Cara Anotasi Dataset

Proses anotasi adalah pemberian label pada setiap gambar dengan cara menggambar bounding box di sekitar objek yang ingin dideteksi.

### 8.1 Menggunakan LabelImg (Offline, Gratis)

LabelImg adalah aplikasi desktop ringan untuk anotasi gambar.

**Instalasi:**

```bash
pip install labelImg
```

**Menjalankan LabelImg:**

```bash
labelImg
```

**Langkah anotasi dengan LabelImg:**

1. Buka LabelImg
2. Klik **Open Dir** → pilih folder `dataset/images/train/`
3. Klik **Change Save Dir** → arahkan ke `dataset/labels/train/`
4. Pastikan format output dipilih **YOLO** (bukan PascalVOC)
5. Untuk setiap gambar:
   - Tekan `W` untuk membuat bounding box baru
   - Seret mouse untuk menggambar kotak di sekitar objek
   - Pilih nama kelas dari dropdown yang muncul
   - Ulangi untuk setiap objek dalam gambar
   - Tekan `Ctrl+S` untuk menyimpan file label `.txt`
6. Tekan `D` untuk pindah ke gambar berikutnya

**Shortcut penting LabelImg:**

| Tombol | Fungsi |
|--------|--------|
| `W` | Buat bounding box baru |
| `D` | Gambar berikutnya |
| `A` | Gambar sebelumnya |
| `Ctrl+S` | Simpan anotasi |
| `Del` | Hapus bounding box terpilih |

### 8.2 Menggunakan Roboflow (Online, Berbasis Web)

Roboflow menyediakan platform anotasi berbasis web dengan fitur kolaborasi tim dan augmentasi otomatis.

**Langkah menggunakan Roboflow:**

1. Daftar akun di [roboflow.com](https://roboflow.com)
2. Buat project baru → pilih tipe **Object Detection**
3. Upload gambar ke platform
4. Mulai anotasi:
   - Klik **Annotate** pada gambar
   - Pilih tool **Bounding Box**
   - Gambar kotak di sekitar objek
   - Ketik atau pilih nama kelas
   - Klik **Save** untuk menyimpan
5. Setelah semua gambar dianotasi:
   - Klik **Generate** untuk membuat versi dataset
   - Pilih split ratio (misal: 80% train, 20% val)
   - Download dalam format **YOLOv8**

### 8.3 Menggunakan CVAT (Computer Vision Annotation Tool)

CVAT adalah platform anotasi open-source yang dapat dijalankan secara lokal.

**Instalasi dengan Docker:**

```bash
git clone https://github.com/openvinotoolkit/cvat
cd cvat
docker compose up -d
```

Akses melalui browser: `http://localhost:8080`

### 8.4 Validasi File Label

Setelah proses anotasi selesai, validasi bahwa setiap file gambar memiliki file label yang berpadanan:

```bash
# Periksa apakah ada gambar tanpa label
python -c "
import os
from pathlib import Path

img_dir = Path('dataset/images/train')
lbl_dir = Path('dataset/labels/train')

missing = []
for img_file in img_dir.glob('*.jpg'):
    lbl_file = lbl_dir / (img_file.stem + '.txt')
    if not lbl_file.exists():
        missing.append(img_file.name)

if missing:
    print(f'PERINGATAN: {len(missing)} gambar tidak memiliki label:')
    for f in missing[:10]:
        print(f'  - {f}')
else:
    print('OK: Semua gambar memiliki file label.')
"
```

---

## 9. Proses Training Model YOLOv8n

### 9.1 Konfigurasi Training

Sebelum menjalankan training, periksa konfigurasi di `scripts/train.py`:

```python
# Konfigurasi yang dapat disesuaikan
PRETRAINED_MODEL = "yolov8n.pt"                  # Model dasar pretrained COCO
DATASET_CONFIG   = "dataset/dataset.yaml"         # Path konfigurasi dataset
PROJECT_DIR      = "runs/train"                   # Folder output hasil training
EXPERIMENT_NAME  = "priority_vehicle_detection"   # Nama eksperimen

EPOCHS      = 100   # Jumlah epoch (semakin banyak = semakin lama tapi bisa lebih akurat)
IMAGE_SIZE  = 640   # Ukuran gambar input (piksel)
BATCH_SIZE  = 16    # Turunkan ke 8 jika memori GPU tidak cukup
DEVICE      = 0     # 0 = GPU pertama; "cpu" = tanpa GPU
```

**Panduan pemilihan BATCH_SIZE:**

| VRAM GPU | Rekomendasi BATCH_SIZE |
|----------|------------------------|
| 4 GB | 8 |
| 6 GB | 16 |
| 8 GB | 32 |
| Tanpa GPU (CPU) | 4 |

### 9.2 Menjalankan Training

```bash
python scripts/train.py
```

Tampilan output saat training berjalan:

```
============================================================
  YOLOv8n Training - Deteksi Kendaraan Prioritas
============================================================

[1/3] Loading pretrained model: yolov8n.pt
[2/3] Memulai training selama 100 epoch...
      Dataset : dataset/dataset.yaml
      Imgsz   : 640
      Batch   : 16
      Device  : 0

Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100     2.41G      1.842      3.054      1.271         47        640
  2/100     2.41G      1.756      2.891      1.243         52        640
  ...
100/100     2.41G      0.821      1.124      0.954         61        640

[3/3] Menyalin model terbaik ke: model/best.pt
  ✓ Model berhasil disimpan di: model/best.pt

============================================================
  Training selesai!
  Hasil lengkap : runs/train/priority_vehicle_detection/
  Model API     : model/best.pt
============================================================
```

### 9.3 Memahami Output Training

Setelah training selesai, tersedia beberapa file hasil di `runs/train/priority_vehicle_detection/`:

```
runs/train/priority_vehicle_detection/
├── weights/
│   ├── best.pt          ← Model terbaik berdasarkan mAP validasi
│   └── last.pt          ← Checkpoint epoch terakhir
├── results.png          ← Grafik loss dan mAP
├── results.csv          ← Data numerik hasil training
├── confusion_matrix.png ← Matriks konfusi pada data validasi
├── PR_curve.png         ← Precision-Recall curve
├── F1_curve.png         ← F1-score curve
└── val_batch0_pred.jpg  ← Contoh prediksi pada data validasi
```

**Metrik yang perlu diperhatikan:**

| Metrik | Keterangan | Target |
|--------|------------|--------|
| `mAP50` | Mean AP pada IoU 0.5 | > 0.70 |
| `mAP50-95` | Mean AP pada IoU 0.5–0.95 | > 0.50 |
| `Precision` | Ketepatan prediksi positif | > 0.75 |
| `Recall` | Kelengkapan deteksi | > 0.70 |

### 9.4 Tips Meningkatkan Akurasi

Jika hasil training kurang memuaskan, beberapa langkah yang dapat dilakukan:

1. **Tambah jumlah data** – Minimal 500 gambar per kelas untuk hasil optimal
2. **Augmentasi data** – Roboflow menyediakan augmentasi otomatis (flip, rotate, brightness)
3. **Perpanjang epoch** – Ubah `EPOCHS = 150` atau `200`
4. **Perbaiki kualitas anotasi** – Pastikan bounding box rapat dan akurat
5. **Gunakan model lebih besar** – Ganti `yolov8n.pt` dengan `yolov8s.pt` jika memori mencukupi

---

## 10. Menjalankan API Backend

### 10.1 Menjalankan Server

Pastikan virtual environment sudah aktif, lalu jalankan:

```bash
uvicorn app.main:app --reload
```

Output yang muncul:

```
INFO:     Will watch for changes in these directories: ['/path/to/project']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using WatchFiles
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
[YOLOService] Loading trained model: model/best.pt
[OCRService] Inisialisasi EasyOCR, bahasa: ['en']
INFO:     Application startup complete.
```

> **Catatan:** Proses startup memerlukan beberapa detik karena model YOLO dan EasyOCR di-load ke memori.

### 10.2 Opsi Perintah Uvicorn

```bash
# Development (auto-reload saat file berubah)
uvicorn app.main:app --reload

# Production (tanpa reload, bisa diakses dari jaringan)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Port berbeda
uvicorn app.main:app --reload --port 8080

# Dengan log level
uvicorn app.main:app --reload --log-level debug
```

### 10.3 Endpoint yang Tersedia

| Method | URL | Deskripsi |
|--------|-----|-----------|
| `GET` | `/` | Health check dasar |
| `GET` | `/health` | Health check untuk monitoring |
| `POST` | `/api/v1/detect` | Deteksi kendaraan + OCR plat |
| `GET` | `/docs` | Dokumentasi Swagger UI (interaktif) |
| `GET` | `/redoc` | Dokumentasi ReDoc |

### 10.4 Akses Dokumentasi Interaktif

Buka browser dan akses Swagger UI:

```
http://127.0.0.1:8000/docs
```

Swagger UI memungkinkan pengujian endpoint langsung dari browser tanpa memerlukan tool tambahan. Klik **POST /api/v1/detect** → **Try it out** → pilih file gambar → **Execute**.

---

## 11. Pengujian Endpoint /api/v1/detect

### 11.1 Menggunakan curl (Terminal)

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detect" \
  -H "accept: application/json" \
  -F "file=@/path/to/gambar_ambulance.jpg"
```

Contoh di Windows:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detect" ^
  -H "accept: application/json" ^
  -F "file=@C:\Users\YusnarSetiyadi\Pictures\ambulance.jpg"
```

### 11.2 Menggunakan Postman

1. Buka Postman → klik **New** → **HTTP Request**
2. Ubah method menjadi **POST**
3. Masukkan URL: `http://127.0.0.1:8000/api/v1/detect`
4. Klik tab **Body**
5. Pilih **form-data**
6. Isi kolom:
   - KEY: `file`
   - VALUE: klik ikon dropdown di kolom VALUE → pilih **File**
   - Pilih file gambar dari komputer
7. Klik tombol **Send**

### 11.3 Menggunakan Python (requests)

```python
import requests
import json

def test_detect(image_path: str, server_url: str = "http://127.0.0.1:8000"):
    """Test endpoint deteksi kendaraan."""
    url = f"{server_url}/api/v1/detect"

    with open(image_path, "rb") as f:
        response = requests.post(url, files={"file": f})

    if response.status_code == 200:
        result = response.json()
        print("=== Hasil Deteksi ===")
        print(f"Kendaraan    : {result['vehicle']}")
        print(f"Nomor Polisi : {result['plate_number']}")
        print(f"Confidence   : {result['confidence']}")
        print(f"Plat Terdeteksi: {result['plate_detected']}")
        return result
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Jalankan test
test_detect("dataset/images/val/ambulance_val_001.jpg")
```

Simpan sebagai `scripts/test_api.py` lalu jalankan:

```bash
python scripts/test_api.py
```

### 11.4 Menggunakan HTTPie (Alternatif curl)

```bash
pip install httpie

http --form POST http://127.0.0.1:8000/api/v1/detect file@gambar.jpg
```

---

## 12. Contoh Request dan Response

### 12.1 Response Sukses – Kendaraan dan Plat Terdeteksi

**Request:**
```
POST /api/v1/detect
Content-Type: multipart/form-data
Body: file = ambulance_001.jpg
```

**Response (HTTP 200):**
```json
{
  "vehicle": "ambulance",
  "plate_number": "B1234XYZ",
  "confidence": 0.9512,
  "plate_detected": true
}
```

### 12.2 Response Parsial – Kendaraan Terdeteksi, Plat Tidak Terbaca

Terjadi ketika plat nomor terdeteksi oleh YOLO tetapi teks tidak berhasil dibaca oleh OCR (misalnya gambar terlalu buram atau plat tertutup).

```json
{
  "vehicle": "police",
  "plate_number": "",
  "confidence": 0.8731,
  "plate_detected": false
}
```

### 12.3 Response Parsial – Kendaraan Terdeteksi, Plat Tidak Ada

Terjadi ketika YOLO mendeteksi kendaraan prioritas tetapi tidak menemukan objek `license_plate` dalam gambar.

```json
{
  "vehicle": "fire_truck",
  "plate_number": "",
  "confidence": 0.8103,
  "plate_detected": false
}
```

### 12.4 Response Gagal – Tidak Ada Kendaraan Prioritas

Terjadi ketika tidak ada objek `ambulance`, `police`, atau `fire_truck` terdeteksi dalam gambar.

```json
{
  "vehicle": null,
  "plate_number": "",
  "confidence": 0,
  "plate_detected": false
}
```

### 12.5 Response Error – Format File Tidak Didukung

```json
{
  "detail": "Format tidak didukung. Gunakan JPEG atau PNG."
}
```

HTTP Status: `400 Bad Request`

### 12.6 Deskripsi Field Response

| Field | Tipe | Keterangan |
|-------|------|------------|
| `vehicle` | `string` / `null` | Nama kelas kendaraan utama yang terdeteksi. `null` jika tidak ada. |
| `plate_number` | `string` | Nomor polisi hasil OCR dalam format "B1234XYZ". String kosong `""` jika tidak terbaca. |
| `confidence` | `float` | Skor keyakinan deteksi kendaraan (0.0–1.0). `0` jika tidak ada kendaraan. |
| `plate_detected` | `boolean` | `true` jika teks plat berhasil dibaca oleh OCR. |

---

## 13. Troubleshooting

### 13.1 Model Tidak Ditemukan

**Gejala:**
```
[YOLOService] model/best.pt tidak ditemukan, menggunakan pretrained: yolov8n.pt
```

**Penyebab:** File `model/best.pt` belum ada karena training belum dijalankan.

**Solusi:**

```bash
# Pastikan dataset sudah disiapkan, lalu jalankan training
python scripts/train.py

# Verifikasi file model berhasil dibuat
ls model/best.pt
```

> **Catatan:** Sistem tetap berfungsi menggunakan model pretrained COCO sebagai fallback, namun akurasi untuk kelas kendaraan prioritas mungkin rendah karena model belum dilatih dengan dataset spesifik.

---

### 13.2 OCR Tidak Membaca Teks Plat

**Gejala:** `plate_number` selalu bernilai `""` meskipun plat terlihat jelas.

**Kemungkinan penyebab dan solusinya:**

| Penyebab | Solusi |
|----------|--------|
| Gambar terlalu kecil / buram | Gunakan gambar resolusi lebih tinggi |
| Plat nomor tidak terdeteksi YOLO | Periksa apakah class `license_plate` ada di model |
| Confidence OCR di bawah threshold | Turunkan threshold di `ocr_service.py`: `if conf >= 0.30` |
| Orientasi teks miring | Tambahkan preprocessing rotasi pada `_preprocess()` |

**Cara debug:**

```python
# Tambahkan sementara di ocr_service.py untuk melihat raw output OCR
ocr_results = self.reader.readtext(processed, detail=1, paragraph=False)
print("DEBUG OCR raw:", ocr_results)
```

---

### 13.3 Hasil Deteksi Selalu Kosong

**Gejala:** Response selalu `"vehicle": null`.

**Langkah diagnosis:**

```bash
# Test model langsung tanpa API
python scripts/predict.py --image path/ke/gambar.jpg --show
```

Kemungkinan penyebab:

1. **Confidence threshold terlalu tinggi** – Turunkan `CONFIDENCE_THRESHOLD` di `yolo_service.py` dari `0.40` menjadi `0.25`

2. **Model belum dilatih** – Gunakan gambar dari kelas yang ada di dataset COCO (bukan kelas kendaraan prioritas Indonesia) untuk memverifikasi model berfungsi

3. **Ukuran gambar terlalu kecil** – YOLOv8 bekerja optimal pada gambar minimal 320×320 piksel

---

### 13.4 Dependency Error saat Install

**Gejala:**
```
ERROR: Could not find a version that satisfies the requirement torch>=2.2.0
```

**Solusi:**

```bash
# Pastikan pip sudah diupdate
pip install --upgrade pip

# Install torch terlebih dahulu (versi CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Lalu install sisa requirements
pip install -r requirements.txt
```

---

### 13.5 Error Port Sudah Digunakan

**Gejala:**
```
ERROR:    [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)
```

**Solusi:**

```bash
# Gunakan port lain
uvicorn app.main:app --reload --port 8001

# Atau cari dan hentikan proses yang menggunakan port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# Linux/macOS:
lsof -i :8000
kill -9 <PID>
```

---

### 13.6 EasyOCR Lambat Saat Pertama Kali

**Gejala:** Server baru jalan sangat lambat saat request pertama.

**Penjelasan:** EasyOCR mengunduh model bahasa dari internet saat pertama kali diinisialisasi. Proses ini normal dan hanya terjadi sekali. Model akan tersimpan di cache lokal (`~/.EasyOCR/`).

**Solusi:** Tunggu proses selesai. Untuk lingkungan tanpa internet, unduh model secara manual dan letakkan di folder cache.

---

### 13.7 Memory Error saat Training

**Gejala:**
```
RuntimeError: CUDA out of memory
```

**Solusi:**

```python
# Di scripts/train.py, turunkan BATCH_SIZE
BATCH_SIZE = 8   # atau 4
```

Atau gunakan CPU:

```python
DEVICE = "cpu"
```

---

*Dokumentasi ini dibuat sebagai bagian dari penelitian sistem deteksi kendaraan prioritas menggunakan deep learning. Seluruh komponen telah diuji dan dapat dijalankan sebagai sistem yang terintegrasi.*
