"""
scripts/train.py - Script untuk training model YOLOv8n

Cara menjalankan:
  python scripts/train.py

Hasil training disimpan di:
  runs/train/priority_vehicle_detectionX/

Model terbaik (best.pt) otomatis di-copy ke:
  model/best.pt
"""

import os
import shutil
from pathlib import Path

from ultralytics import YOLO


# ─── Konfigurasi Training ────────────────────────────────────────────────────

PRETRAINED_MODEL = "yolov8n.pt"       # Base model pretrained COCO
DATASET_CONFIG   = "dataset/dataset.yaml"
PROJECT_DIR      = "runs/train"
EXPERIMENT_NAME  = "priority_vehicle_detection"
OUTPUT_MODEL_DIR = Path("model")

# Hyperparameter training
# EPOCHS      = 100
# IMAGE_SIZE  = 640
# BATCH_SIZE  = 16
# WORKERS     = 4
# DEVICE      = "cpu"

# New hyperparam training (from cgpt) - BEING TRAINED
EPOCHS      = 120
IMAGE_SIZE  = 960
BATCH_SIZE  = 8
WORKERS     = 2
DEVICE      = "cpu"

# ─────────────────────────────────────────────────────────────────────────────


def train():
    print("=" * 60)
    print("  YOLOv8n Training - Deteksi Kendaraan Prioritas")
    print("=" * 60)

    # Pastikan dataset.yaml ada
    if not Path(DATASET_CONFIG).exists():
        raise FileNotFoundError(
            f"Dataset config tidak ditemukan: {DATASET_CONFIG}\n"
            "Pastikan kamu sudah meletakkan gambar & label di folder dataset/"
        )

    # Pastikan folder output model ada
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load model pretrained YOLOv8n
    print(f"\n[1/3] Loading pretrained model: {PRETRAINED_MODEL}")
    model = YOLO(PRETRAINED_MODEL)
    
    # resume model with uncomment this
    # model = YOLO("runs/train/priority_vehicle_detection/weights/last.pt")
    # or this
    # model = YOLO(r"C:\Users\YusnarSetiyadi\Me\technology\ilycode\Python-YOLO-PriorityNopol\runs\detect\runs\train\priority_vehicle_detection\weights\last.pt")

    # Mulai training
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
        exist_ok=True,           # Overwrite jika nama experiment sama

        save_period=10,          # Simpan checkpoint setiap 10 epoch
        resume=False,             # for resuming train model
        multi_scale=True,        # for tiny object (nopol)

        # 🔥 augment penting
        mosaic=1.0,
        mixup=0.15,

        # 🔥 warna
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # 🔥 tambahan biar robust
        degrees=5,
        scale=0.5,
        shear=2,

        # ⚠️ CPU friendly
        patience=25,             # Early stopping: berhenti jika 25 epoch tidak ada improvement
        save=True,               # Simpan checkpoint
        val=True,                # Validasi setiap epoch
        plots=True,              # Simpan grafik training
        verbose=True,
    )

    # Copy best.pt ke model/best.pt
    best_pt_src = Path(PROJECT_DIR) / EXPERIMENT_NAME / "weights" / "best.pt"
    best_pt_dst = OUTPUT_MODEL_DIR / "best.pt"

    print(f"\n[3/3] Menyalin model terbaik ke: {best_pt_dst}")

    if best_pt_src.exists():
        shutil.copy2(best_pt_src, best_pt_dst)
        print(f"  ✓ Model berhasil disimpan di: {best_pt_dst}")
    else:
        print(f"  ✗ File best.pt tidak ditemukan di: {best_pt_src}")
        print("    Periksa folder runs/train untuk hasil training.")

    print("\n" + "=" * 60)
    print("  Training selesai!")
    print(f"  Hasil lengkap : {PROJECT_DIR}/{EXPERIMENT_NAME}/")
    print(f"  Model API     : {best_pt_dst}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train()
