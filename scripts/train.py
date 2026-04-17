"""
scripts/train.py — Entry point training YOLOv8n

Tersedia DUA pendekatan training:

──────────────────────────────────────────────────────────────────────────────
PENDEKATAN A — Dua tahap (DIREKOMENDASIKAN untuk nopol)
──────────────────────────────────────────────────────────────────────────────
Jalankan berurutan:

  1. python scripts/train_vehicle.py
     → Melatih deteksi kendaraan (ambulance, police, fire_truck, license_plate)
     → CPU-optimized: imgsz=640, batch=4, cache=disk, mixup=0
     → Output: model/vehicle_best.pt
     → Estimasi: ~2-4 jam CPU

  2. python scripts/train_nopol.py
     → Otomatis filter dataset: hanya gambar dengan license_plate
     → Melatih model KHUSUS nopol (1 kelas, augmentasi small-object agresif)
     → Output: model/nopol_best.pt
     → Estimasi: ~45-90 menit CPU

Keunggulan dua tahap:
  - Nopol mendapat 100% kapasitas model → mAP lebih tinggi
  - Stage 2 "memperbesar" plat (dari crop kendaraan): resolved small-object issue
  - Training lebih cepat per stage (lebih sedikit kompleksitas)

──────────────────────────────────────────────────────────────────────────────
PENDEKATAN B — Satu tahap (legacy, semua kelas sekaligus)
──────────────────────────────────────────────────────────────────────────────
Script ini (train.py) menjalankan satu model 4-kelas dengan parameter lama.
Gunakan hanya jika tidak ingin dua model terpisah.

  python scripts/train.py

──────────────────────────────────────────────────────────────────────────────
PERUBAHAN CPU OPTIMIZATION (berlaku di train_vehicle.py)
──────────────────────────────────────────────────────────────────────────────
  IMAGE_SIZE  960 → 640     : 2.25x lebih cepat (area ∝ px²)
  BATCH_SIZE  8   → 4       : hemat RAM, cache-friendly
  WORKERS     2   → 0       : Windows CPU, hindari multiprocessing overhead
  EPOCHS      120 → 80      : early stopping kompensasi
  patience    25  → 20      : stop lebih cepat saat plateau
  multi_scale True → False  : resize per-batch = overhead CPU murni
  mixup       0.15 → 0.0    : +1 forward pass/batch → skip di CPU
  cache       -   → 'disk'  : epoch 2+ skip disk read → 40-60% speedup
──────────────────────────────────────────────────────────────────────────────
"""

import shutil
from pathlib import Path

from ultralytics import YOLO


# ─── Konfigurasi Training (Pendekatan B — Legacy) ────────────────────────────

PRETRAINED_MODEL = "yolov8n.pt"
DATASET_CONFIG   = "dataset/dataset.yaml"
PROJECT_DIR      = "runs/train"
EXPERIMENT_NAME  = "priority_vehicle_detection"
OUTPUT_MODEL_DIR = Path("model")

# Hyperparameter (legacy — lihat train_vehicle.py untuk versi CPU-optimized)
EPOCHS      = 120
IMAGE_SIZE  = 960
BATCH_SIZE  = 8
WORKERS     = 2
DEVICE      = "cpu"

# resume: uncomment baris model di bawah dan set resume=True
RESUME = False

# ─────────────────────────────────────────────────────────────────────────────


def train():
    print("=" * 60)
    print("  YOLOv8n Training - Deteksi Kendaraan Prioritas (Legacy)")
    print("  Untuk performa lebih baik, gunakan train_vehicle.py")
    print("=" * 60)

    if not Path(DATASET_CONFIG).exists():
        raise FileNotFoundError(
            f"Dataset config tidak ditemukan: {DATASET_CONFIG}"
        )

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Loading pretrained model: {PRETRAINED_MODEL}")
    model = YOLO(PRETRAINED_MODEL)

    # Untuk resume, ganti ke:
    # model = YOLO("runs/train/priority_vehicle_detection/weights/last.pt")

    print(f"[2/3] Memulai training selama {EPOCHS} epoch...")
    print(f"      Dataset : {DATASET_CONFIG}")
    print(f"      Imgsz   : {IMAGE_SIZE}")
    print(f"      Batch   : {BATCH_SIZE}")
    print(f"      Device  : {DEVICE}")
    print()

    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        resume=RESUME,
        save_period=5,
        multi_scale=True,
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        scale=0.5,
        shear=2,
        patience=25,
        save=True,
        val=True,
        plots=True,
        verbose=True,
    )

    best_pt_src = Path(PROJECT_DIR) / EXPERIMENT_NAME / "weights" / "best.pt"
    best_pt_dst = OUTPUT_MODEL_DIR / "best.pt"

    print(f"\n[3/3] Menyalin model terbaik ke: {best_pt_dst}")
    if best_pt_src.exists():
        shutil.copy2(best_pt_src, best_pt_dst)
        print(f"  OK Model disimpan: {best_pt_dst}")
    else:
        print(f"  GAGAL best.pt tidak ada di: {best_pt_src}")

    print("\n" + "=" * 60)
    print("  Training selesai!")
    print(f"  Hasil lengkap : {PROJECT_DIR}/{EXPERIMENT_NAME}/")
    print(f"  Model terbaik : {best_pt_dst}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train()
