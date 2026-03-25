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
EPOCHS      = 100
IMAGE_SIZE  = 640
BATCH_SIZE  = 16   # Turunkan ke 8 jika RAM GPU tidak cukup
WORKERS     = 4    # Jumlah dataloader workers (0 = main thread, aman di Windows)
DEVICE      = 0    # 0 = GPU pertama; "cpu" jika tidak ada GPU

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
        patience=20,             # Early stopping: berhenti jika 20 epoch tidak ada improvement
        save=True,               # Simpan checkpoint
        save_period=10,          # Simpan checkpoint setiap 10 epoch
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
