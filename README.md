# Deteksi Otomatis Kendaraan Prioritas Menggunakan YOLOv8n untuk Pencatatan Nomor Polisi

## A. Deskripsi Project

Project ini merupakan sistem penelitian berbasis deep learning untuk mendeteksi secara otomatis kendaraan-kendaraan prioritas (ambulance, mobil polisi, pemadam kebakaran) beserta plat nomornya menggunakan **YOLOv8n** (You Only Look Once versi 8 nano).

Sistem ini dibangun sebagai backend API end-to-end:

```
Upload Gambar → YOLO Deteksi Kendaraan + Plat Nomor → EasyOCR Baca Teks → Return JSON
```

### Teknologi yang Digunakan

| Komponen | Teknologi |
|---|---|
| Object Detection | YOLOv8n (Ultralytics) |
| OCR Plat Nomor | EasyOCR |
| API Backend | FastAPI + Uvicorn |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |

### Kelas yang Dideteksi

| Class ID | Nama Kelas | Keterangan |
|---|---|---|
| 0 | `ambulance` | Kendaraan ambulans |
| 1 | `police` | Kendaraan kepolisian |
| 2 | `fire_truck` | Kendaraan pemadam kebakaran |
| 3 | `license_plate` | Plat nomor kendaraan |

---

## B. Cara Install

### 1. Clone / Siapkan Project

```bash
git clone <repo-url>
cd Python-YOLO-PriorityNopol
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
```

Aktifkan virtual environment:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Catatan GPU (Opsional):** Jika menggunakan GPU NVIDIA, install PyTorch dengan CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## C. Cara Menambahkan Dataset

### Struktur Folder Dataset

```
dataset/
├── images/
│   ├── train/          ← Gambar training (.jpg / .png)
│   └── val/            ← Gambar validasi (.jpg / .png)
├── labels/
│   ├── train/          ← Label YOLO (.txt, satu file per gambar)
│   └── val/
└── dataset.yaml
```

### Format Label YOLO

Setiap gambar memiliki file `.txt` dengan nama yang sama. Setiap baris berisi satu objek:

```
<class_id> <x_center> <y_center> <width> <height>
```

- Semua nilai koordinat dalam format **normalized** (0.0 – 1.0) relatif terhadap ukuran gambar.
- `class_id` sesuai dengan urutan di `dataset.yaml`.

### Contoh File Label

Misal gambar `ambulance_001.jpg` memiliki file label `ambulance_001.txt`:

```
0 0.5134 0.4821 0.6230 0.5940
3 0.5020 0.7812 0.2100 0.0850
```

- Baris 1: kelas `0` (ambulance), bbox di tengah gambar.
- Baris 2: kelas `3` (license_plate), bbox di bawah kendaraan.

### Tips Anotasi Dataset

- Gunakan tool anotasi seperti **Roboflow**, **LabelImg**, atau **CVAT**.
- Export dalam format **YOLOv8 / YOLO Darknet**.
- Disarankan minimal **200 gambar per kelas** untuk hasil training yang baik.
- Pastikan distribusi data train : val = **80% : 20%**.

---

## D. Cara Training Model

### Jalankan Script Training

```bash
python scripts/train.py
```

Script akan:
1. Load pretrained `yolov8n.pt` dari Ultralytics
2. Fine-tune menggunakan dataset custom di `dataset/`
3. Simpan semua hasil ke `runs/train/priority_vehicle_detection/`
4. Otomatis copy `best.pt` ke `model/best.pt`

### Konfigurasi Training (di `scripts/train.py`)

```python
EPOCHS      = 100       # Jumlah epoch training
IMAGE_SIZE  = 640       # Ukuran gambar input (piksel)
BATCH_SIZE  = 16        # Turunkan ke 8 jika VRAM tidak cukup
DEVICE      = 0         # 0 = GPU; "cpu" = CPU only
```

### Monitoring Training

Setelah training selesai, buka hasil di:
```
runs/train/priority_vehicle_detection/
├── weights/
│   ├── best.pt         ← Model terbaik (berdasarkan mAP validasi)
│   └── last.pt         ← Checkpoint epoch terakhir
├── results.png         ← Grafik loss & mAP
├── confusion_matrix.png
└── ...
```

---

## E. Cara Menjalankan API

### Jalankan Server FastAPI

```bash
uvicorn app.main:app --reload
```

- `--reload` : Auto-restart saat ada perubahan kode (mode development)
- Default berjalan di `http://127.0.0.1:8000`

### Jalankan di Port Berbeda

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### Akses Dokumentasi API (Swagger UI)

Buka browser dan akses:
```
http://127.0.0.1:8000/docs
```

---

## F. Cara Test API

### Menggunakan curl

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detect" \
  -H "accept: application/json" \
  -F "file=@/path/to/gambar.jpg"
```

### Menggunakan Postman

1. Buka Postman → New Request
2. Method: `POST`
3. URL: `http://127.0.0.1:8000/api/v1/detect`
4. Pilih tab **Body** → **form-data**
5. Tambahkan key: `file`, Type: **File**, Value: pilih gambar
6. Klik **Send**

### Menggunakan Python (requests)

```python
import requests

url = "http://127.0.0.1:8000/api/v1/detect"

with open("gambar_ambulance.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json())
```

### Contoh Response JSON

**Success** – kendaraan dan plat terdeteksi:
```json
{
  "vehicle": "ambulance",
  "plate_number": "B1234XYZ",
  "confidence": 0.9512,
  "plate_detected": true
}
```

**Partial** – kendaraan terdeteksi tapi plat tidak terbaca:
```json
{
  "vehicle": "police",
  "plate_number": "",
  "confidence": 0.8731,
  "plate_detected": false
}
```

**Failure** – tidak ada kendaraan prioritas:
```json
{
  "vehicle": null,
  "plate_number": "",
  "confidence": 0,
  "plate_detected": false
}
```

### Integrasi Flutter (Dart)

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';

Future<Map<String, dynamic>> detectVehicle(File imageFile) async {
  final uri = Uri.parse('http://YOUR_SERVER_IP:8000/api/v1/detect');
  final request = http.MultipartRequest('POST', uri);

  request.files.add(
    await http.MultipartFile.fromPath('file', imageFile.path),
  );

  final response = await request.send();
  final body = await response.stream.bytesToString();
  return jsonDecode(body);
}

// Contoh penggunaan:
// final result = await detectVehicle(imageFile);
// print(result['vehicle']);       // "ambulance"
// print(result['plate_number']);  // "B1234XYZ"
// print(result['confidence']);    // 0.9512
// print(result['plate_detected']); // true
```

### Endpoint yang Tersedia

| Method | Endpoint | Deskripsi |
|---|---|---|
| GET | `/` | Health check dasar |
| GET | `/health` | Health check untuk monitoring |
| POST | `/api/v1/detect` | Deteksi kendaraan + OCR plat nomor |
| GET | `/docs` | Dokumentasi Swagger UI |
| GET | `/redoc` | Dokumentasi ReDoc |

---

## G. Flow Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                     FLOW SISTEM                             │
└─────────────────────────────────────────────────────────────┘

  [Client / Pengguna]
       │
       │  POST /api/v1/detect
       │  Content-Type: multipart/form-data
       │  Body: file = <gambar.jpg>
       ▼
  ┌─────────────────────┐
  │   FastAPI Router    │  ← Validasi file (JPEG/PNG)
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │   decode_image()    │  ← Bytes → NumPy array BGR (OpenCV)
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │   YOLOService       │  ← Load model/best.pt (singleton)
  │   .detect()         │  ← Inference → list bounding boxes
  └────────┬────────────┘
           │
           ├─── class 0,1,2 (kendaraan) → vehicle_detections[]
           │
           └─── class 3 (license_plate) → crop_region()
                                                │
                                                ▼
                                     ┌──────────────────┐
                                     │  OCRService      │
                                     │  .read_plate()   │
                                     │  preprocess →    │
                                     │  EasyOCR →       │
                                     │  clean text      │
                                     └────────┬─────────┘
                                              │
                                              ▼
                                    plate_number: "B 1234 XYZ"
           │
           ▼
  ┌─────────────────────────────────────────────────────────┐
  │                    Response JSON                        │
  │  {                                                      │
  │    "detections": [{"class": "ambulance", "conf": 0.95}] │
  │    "plate_number": "B 1234 XYZ",                        │
  │    "total_vehicles": 1,                                 │
  │    "plates_found": 1                                    │
  │  }                                                      │
  └─────────────────────────────────────────────────────────┘
```

---

## H. Tujuan Penelitian

Project ini dikembangkan sebagai bagian dari penelitian ilmiah dengan topik:

> **"Deteksi Otomatis Kendaraan Prioritas Menggunakan YOLOv8n untuk Pencatatan Nomor Polisi"**

### Latar Belakang

Kendaraan prioritas (ambulans, polisi, pemadam kebakaran) memerlukan akses cepat di jalan raya. Sistem deteksi otomatis berbasis computer vision dapat dimanfaatkan untuk:

- Monitoring lalu lintas kendaraan prioritas secara real-time
- Pencatatan otomatis nomor polisi kendaraan prioritas yang melintas
- Integrasi dengan sistem manajemen lalu lintas cerdas (Intelligent Traffic System)
- Dokumentasi data kendaraan prioritas untuk keperluan administratif

### Kontribusi Penelitian

1. **Model YOLOv8n** yang dioptimalkan khusus untuk deteksi kendaraan prioritas
2. **Pipeline OCR** terintegrasi untuk pembacaan nomor polisi secara otomatis
3. **REST API** yang dapat diintegrasikan dengan sistem kamera CCTV atau aplikasi lain
4. **Dataset** kendaraan prioritas berlabel untuk kebutuhan training model

---

## Struktur Project

```
Python-YOLO-PriorityNopol/
│
├── app/                          ← Aplikasi FastAPI
│   ├── main.py                   ← Entry point, setup CORS & router
│   ├── routes/
│   │   └── detect.py             ← Endpoint POST /api/v1/detect
│   ├── services/
│   │   ├── yolo_service.py       ← Load YOLO, jalankan inference
│   │   └── ocr_service.py        ← EasyOCR, preprocess, baca plat
│   └── utils/
│       └── image_utils.py        ← decode_image, crop_region, resize
│
├── model/
│   └── best.pt                   ← Model hasil training (di-generate)
│
├── dataset/
│   ├── images/
│   │   ├── train/                ← Gambar training
│   │   └── val/                  ← Gambar validasi
│   ├── labels/
│   │   ├── train/                ← Label YOLO training
│   │   └── val/                  ← Label YOLO validasi
│   └── dataset.yaml              ← Konfigurasi dataset
│
├── scripts/
│   ├── train.py                  ← Script training model
│   └── predict.py                ← Script testing inferensi (standalone)
│
├── requirements.txt              ← Daftar dependencies Python
└── README.md                     ← Dokumentasi project ini
```

---

## Troubleshooting

### Model belum ada (model/best.pt tidak ditemukan)

API akan otomatis menggunakan pretrained `yolov8n.pt` (COCO dataset) sebagai fallback.
Untuk hasil optimal, lakukan training terlebih dahulu:
```bash
python scripts/train.py
```

### Error CUDA / GPU

Ubah `DEVICE = "cpu"` di `scripts/train.py`, atau set di OCRService `gpu=False` (sudah default).

### EasyOCR lambat saat pertama kali

EasyOCR mengunduh model bahasa saat pertama dijalankan. Proses ini normal dan hanya terjadi sekali.

### Port 8000 sudah terpakai

```bash
uvicorn app.main:app --reload --port 8001
```
