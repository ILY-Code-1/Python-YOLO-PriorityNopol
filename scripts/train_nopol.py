"""
scripts/train_nopol.py — Stage 2: Deteksi nomor polisi (license plate)

Model ini KHUSUS untuk license plate (1 kelas).
Input di inference: crop kendaraan dari output Stage 1.

Alur inferensi dua tahap:
  Gambar asli
    → Stage 1 (vehicle_best.pt): deteksi ambulance/police/fire_truck
    → Crop setiap bbox kendaraan
    → Stage 2 (nopol_best.pt): deteksi license_plate di dalam crop
    → OCR (EasyOCR) untuk baca teks plat

Cara menjalankan:
  python scripts/train_nopol.py

Script ini otomatis:
  1. Memfilter label kelas 3 (license_plate) dari dataset utama
  2. Membuat dataset_nopol/ dengan struktur YOLO (kelas 0 = license_plate)
  3. Training model khusus nopol

Untuk RESUME:
  Ubah RESUME = True

Hasil:
  runs/train/nopol_detection/weights/best.pt
  model/nopol_best.pt

──────────────────────────────────────────────────────────────────────────────
KENAPA DUA TAHAP?
──────────────────────────────────────────────────────────────────────────────
Masalah model satu tahap untuk nopol:
  - License plate = small object (avg 27% × 12% image)
  - 33% objek plat < 15% lebar gambar → sangat kecil
  - Anchor head YOLOv8n tidak dioptimasi untuk objek kecil sekaligus besar
  - Model "bingung" antara fitur kendaraan besar vs plat kecil

Keuntungan dua tahap:
  - Stage 2 menerima crop kendaraan: plat mengisi ~20-40% lebar crop
  - Effectively meningkatkan resolusi apparent tanpa naikkan imgsz
  - Model Stage 2 fokus pada tekstur plat, bukan bentuk kendaraan
  - mAP nopol bisa naik dari ~0.4 ke >0.7 (estimasi)

──────────────────────────────────────────────────────────────────────────────
AUGMENTASI KHUSUS SMALL OBJECT
──────────────────────────────────────────────────────────────────────────────
scale=0.9    : Zoom lebih ekstrem → plat muncul di berbagai ukuran
translate=0.2: Geser lebih jauh → plat di pinggir frame tetap dilatih
degrees=10   : Rotasi lebih → plat miring (parkir, sudut kamera)
perspective=0.0003: Distorsi perspektif → foto dari sudut miring
erasing=0.4  : Random erase → nopol terpotong sebagian, model lebih robust
mosaic=1.0   : Tetap: 4-in-1 memberikan variasi background
──────────────────────────────────────────────────────────────────────────────
ESTIMASI WAKTU (CPU laptop)
  ~800 gambar dengan plat, batch=4 → 200 batch/epoch
  ~0.35s/batch → 70s = ±1.2 menit/epoch
  60 epoch penuh  : ±1.2 jam
  Dengan patience : berhenti ~30-40 epoch → ±45 menit
──────────────────────────────────────────────────────────────────────────────
"""

import os
import shutil
import time
from pathlib import Path

from ultralytics import YOLO


# ─── Konfigurasi ─────────────────────────────────────────────────────────────

SRC_IMAGES_TRAIN = Path("dataset/images/train")
SRC_IMAGES_VAL   = Path("dataset/images/val")
SRC_LABELS_TRAIN = Path("dataset/labels/train")
SRC_LABELS_VAL   = Path("dataset/labels/val")

NOPOL_DIR        = Path("dataset_nopol")   # Dataset khusus nopol
NOPOL_YAML       = NOPOL_DIR / "dataset_nopol.yaml"

PROJECT_DIR      = "runs/train"
EXPERIMENT_NAME  = "nopol_detection"
OUTPUT_MODEL_DIR = Path("model")

RESUME = True
LAST_PT = Path("runs/detect") / Path(PROJECT_DIR) / EXPERIMENT_NAME / "weights" / "last.pt"
PRETRAINED_MODEL = "yolov8n.pt"

# ─── Hyperparameter ───────────────────────────────────────────────────────────

EPOCHS      = 60      # Satu kelas, lebih cepat konvergen dari model 4-kelas
IMAGE_SIZE  = 640     # Setelah crop kendaraan, plat mengisi ~20-40% → cukup
BATCH_SIZE  = 4       # Hemat RAM, cache-friendly
WORKERS     = 0       # Windows CPU: workers=0 lebih stabil
DEVICE      = "cpu"

LICENSE_PLATE_CLASS_ID = 3   # Class ID di dataset utama

# ─────────────────────────────────────────────────────────────────────────────


def prepare_nopol_dataset():
    """
    Filter dataset utama: ambil hanya gambar dengan class 3 (license_plate).
    Remap class 3 → 0 di label output.
    Simpan ke dataset_nopol/{images,labels}/{train,val}/
    """
    print("\n[PREP] Menyiapkan dataset nopol...")

    copied = {"train": 0, "val": 0}
    skipped = {"train": 0, "val": 0}

    splits = [
        ("train", SRC_IMAGES_TRAIN, SRC_LABELS_TRAIN),
        ("val",   SRC_IMAGES_VAL,   SRC_LABELS_VAL),
    ]

    for split, img_dir, lbl_dir in splits:
        out_img = NOPOL_DIR / "images" / split
        out_lbl = NOPOL_DIR / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            text = lbl_path.read_text().strip()
            if not text:
                skipped[split] += 1
                continue

            # Filter baris yang merupakan license_plate (class 3)
            plate_lines = []
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) == 5 and int(parts[0]) == LICENSE_PLATE_CLASS_ID:
                    # Remap class 3 → 0
                    plate_lines.append("0 " + " ".join(parts[1:]))

            if not plate_lines:
                skipped[split] += 1
                continue

            # Copy gambar
            img_path = img_dir / (lbl_path.stem + ".jpg")
            if not img_path.exists():
                skipped[split] += 1
                continue

            dst_img = out_img / img_path.name
            dst_lbl = out_lbl / lbl_path.name

            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(plate_lines))
            copied[split] += 1

    print(f"  Train: {copied['train']} gambar dengan plat (skip {skipped['train']})")
    print(f"  Val  : {copied['val']} gambar dengan plat (skip {skipped['val']})")

    if copied["train"] == 0:
        raise RuntimeError(
            "Tidak ada gambar license_plate ditemukan. "
            "Pastikan dataset/labels/train/ berisi label kelas 3."
        )

    # Tulis dataset_nopol.yaml
    yaml_content = (
        "# dataset_nopol.yaml — License plate detection (Stage 2)\n"
        f"path: {NOPOL_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: license_plate\n"
    )
    NOPOL_YAML.write_text(yaml_content)
    print(f"  Dataset YAML: {NOPOL_YAML}")

    return copied["train"] + copied["val"]


def train():
    print("=" * 65)
    print("  YOLOv8n Stage 2 — Deteksi Nomor Polisi (License Plate)")
    print("=" * 65)

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Siapkan dataset nopol ───────────────────────────────────────────────
    if not NOPOL_YAML.exists():
        total = prepare_nopol_dataset()
    else:
        # Hitung ulang kalau sudah ada
        total = len(list((NOPOL_DIR / "images" / "train").glob("*.jpg"))) + \
                len(list((NOPOL_DIR / "images" / "val").glob("*.jpg")))
        print(f"\n[PREP] Dataset nopol sudah ada ({total} gambar). Skip persiapan.")
        print("       Hapus folder dataset_nopol/ untuk rebuild dari awal.")

    # ── Load model ─────────────────────────────────────────────────────────
    if RESUME and LAST_PT.exists():
        print(f"\n[RESUME] Loading checkpoint: {LAST_PT}")
        model = YOLO(str(LAST_PT))
    elif RESUME and not LAST_PT.exists():
        print(f"[RESUME] last.pt tidak ada, mulai dari pretrained.")
        model = YOLO(PRETRAINED_MODEL)
    else:
        print(f"\n[1/3] Loading pretrained model: {PRETRAINED_MODEL}")
        model = YOLO(PRETRAINED_MODEL)

    # ── Info training ───────────────────────────────────────────────────────
    batches_per_epoch = max(1, total * 4 // 5 // BATCH_SIZE)   # ~80% train
    est_sec = batches_per_epoch * 0.4 * EPOCHS
    est_h, rem = divmod(int(est_sec), 3600)
    est_m = rem // 60

    print(f"\n[2/3] Memulai training nopol...")
    print(f"      Dataset    : {NOPOL_YAML}")
    print(f"      Total img  : {total} (dengan plat)")
    print(f"      Image size : {IMAGE_SIZE}px")
    print(f"      Batch      : {BATCH_SIZE}")
    print(f"      Epochs     : {EPOCHS} (+ patience=15 early stop)")
    print(f"      Cache      : disk")
    print(f"      Estimasi   : ~{est_h}j {est_m}m (CPU, dengan early stop bisa lebih cepat)")
    print()

    t_start = time.time()

    results = model.train(
        data=str(NOPOL_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        resume=RESUME,

        # ── Checkpoint ──────────────────────────────────────────────────
        save=True,
        save_period=10,

        # ── Validation ──────────────────────────────────────────────────
        val=True,
        plots=True,
        verbose=True,

        # ── Early stopping ───────────────────────────────────────────────
        patience=15,                # Satu kelas → konvergen lebih cepat

        # ── Cache ────────────────────────────────────────────────────────
        cache="disk",               # Speedup epoch 2+ signifikan

        # ── Augmentasi khusus small object ───────────────────────────────
        mosaic=1.0,                 # 4-in-1: variasi ukuran dan posisi plat
        mixup=0.0,                  # Skip: overhead CPU tidak sepadan

        # Geometri — lebih agresif untuk melatih plat di berbagai kondisi
        scale=0.9,                  # 0.5→0.9: zoom jauh lebih ekstrem → plat berbagai ukuran
        translate=0.2,              # 0.1→0.2: plat sering di pinggir crop kendaraan
        degrees=10,                 # 5→10: plat miring, foto dari sudut berbeda
        perspective=0.0003,         # Distorsi perspektif: foto dari samping/atas
        shear=3,                    # Slight shear: sudut foto bervariasi

        # Warna — invariansi kondisi cahaya dan kamera
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,                  # Lebih tinggi: plat di bayangan/terang

        # Robustness
        erasing=0.4,                # Random erase: plat terpotong sebagian
        fliplr=0.5,

        # Nonaktifkan multi_scale — plat kecil, resize per-batch justru rugikan
        multi_scale=False,
    )

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)

    # ── Copy best.pt ────────────────────────────────────────────────────────
    best_src = Path(PROJECT_DIR) / EXPERIMENT_NAME / "weights" / "best.pt"
    best_dst = OUTPUT_MODEL_DIR / "nopol_best.pt"

    print(f"\n[3/3] Menyalin model terbaik ke: {best_dst}")
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"  OK Model disimpan: {best_dst}")
    else:
        print(f"  GAGAL best.pt tidak ada: {best_src}")

    # ── Summary ─────────────────────────────────────────────────────────────
    actual_epochs = results.epoch if hasattr(results, "epoch") else "?"
    print()
    print("=" * 65)
    print("  TRAINING SELESAI — Stage 2 Nopol Detection")
    print("=" * 65)
    print(f"  Waktu total       : {h}j {m}m {s}s")
    print(f"  Epoch selesai     : {actual_epochs} / {EPOCHS}")
    print(f"  Hasil lengkap     : {PROJECT_DIR}/{EXPERIMENT_NAME}/")
    print(f"  Model terbaik     : {best_dst}")
    print()
    print("  Cara pakai dua model:")
    print("    from ultralytics import YOLO")
    print("    v_model = YOLO('model/vehicle_best.pt')")
    print("    p_model = YOLO('model/nopol_best.pt')")
    print("    # 1. deteksi kendaraan")
    print("    vehicles = v_model(image)[0].boxes")
    print("    # 2. crop setiap kendaraan → deteksi plat")
    print("    for box in vehicles:")
    print("        crop = image[y1:y2, x1:x2]")
    print("        plates = p_model(crop)")
    print("=" * 65)

    return results


if __name__ == "__main__":
    train()
