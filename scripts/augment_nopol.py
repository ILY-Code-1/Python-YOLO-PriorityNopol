"""
scripts/augment_nopol.py — Augmentasi dataset khusus untuk nopol (license plate)

Menjalankan:
  pip install albumentations
  python scripts/augment_nopol.py

Output:
  dataset_aug/
  ├── images/
  │   ├── train/    ← gambar augmented
  │   └── val/
  └── labels/
      ├── train/    ← label YOLO (class_id xc yc w h)
      └── val/

Setelah selesai, merge ke dataset utama:
  Salin isi dataset_aug/images/train/ → dataset/images/train/
  Salin isi dataset_aug/labels/train/ → dataset/labels/train/

──────────────────────────────────────────────────────────────────────────────
DUA TAHAP AUGMENTASI
──────────────────────────────────────────────────────────────────────────────

TAHAP 1 — Copy-Paste Synthesis (mengatasi masalah KRITIS)
─────────────────────────────────────────────────────────
Masalah: 0% gambar punya KEDUA label kendaraan + plat bersamaan.
         Model tidak pernah melihat "plat di konteks kendaraan".

Solusi:
  - Kumpulkan crop plat dari gambar LP-only
  - Tempel ke gambar kendaraan di posisi realistis (bawah bbox kendaraan)
  - Scale plat = 18-28% lebar bbox kendaraan (proporsi realistis)
  - Hasilkan label YOLO gabungan (vehicle + license_plate)

TAHAP 2 — Standard Albumentations (meningkatkan robustness)
────────────────────────────────────────────────────────────
Diterapkan ke semua gambar LP-only (520 gambar):
  - Brightness / contrast variasi
  - Motion blur (kendaraan bergerak)
  - Gaussian noise
  - Perspective transform (sudut kamera)
  - Random shadow (bayangan di plat)
  - HSV shift (kondisi cahaya berbeda)
  - Random erase (plat terpotong sebagian)
──────────────────────────────────────────────────────────────────────────────
"""

import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.core.bbox_utils import convert_bboxes_to_albumentations

# ─── Konfigurasi ─────────────────────────────────────────────────────────────

DATASET     = Path("dataset")
OUTPUT_DIR  = Path("dataset_aug")
CLASS_NAMES = {0: "ambulance", 1: "police", 2: "fire_truck", 3: "license_plate"}

# Jumlah gambar synthetic copy-paste yang dihasilkan
COPYPASTE_TARGET    = 600   # per split (train)
COPYPASTE_VAL       = 100   # untuk val
AUGMENT_LP_COPIES   = 3     # berapa kopi per LP gambar (standard aug)

# Ukuran minimum plat setelah di-paste (px)
MIN_PLATE_PX = 30

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────


# Standard augmentation pipeline untuk LP images
AUG_PIPELINE = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
        A.MotionBlur(blur_limit=(3, 9), p=0.45),         # kendaraan bergerak
        A.GaussNoise(std_range=(0.01, 0.08), p=0.35),
        A.Perspective(scale=(0.04, 0.10), p=0.4),        # sudut kamera miring
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.3),  # bayangan di plat
        A.HueSaturationValue(
            hue_shift_limit=12, sat_shift_limit=35, val_shift_limit=25, p=0.5
        ),
        A.CoarseDropout(                                  # plat terpotong sebagian
            num_holes_range=(1, 3),
            hole_height_range=(0.05, 0.25),
            hole_width_range=(0.05, 0.25),
            fill=0,
            p=0.35,
        ),
        A.Rotate(limit=12, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.ShiftScaleRotate(
            shift_limit=0.08, scale_limit=0.2, rotate_limit=0,
            border_mode=cv2.BORDER_REFLECT_101, p=0.5
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.35,    # drop box jika < 35% terlihat setelah aug
    ),
)


def read_labels(lbl_path):
    """Return list of (class_id, xc, yc, w, h)."""
    lines = []
    text = lbl_path.read_text().strip()
    if not text:
        return lines
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                lines.append((int(parts[0]), float(parts[1]), float(parts[2]),
                               float(parts[3]), float(parts[4])))
            except ValueError:
                pass
    return lines


def write_labels(lbl_path, boxes):
    """Write list of (class_id, xc, yc, w, h) to YOLO label file."""
    lines = ["%d %.6f %.6f %.6f %.6f" % b for b in boxes]
    lbl_path.write_text("\n".join(lines))


def collect_plate_crops(img_dir, lbl_dir, max_crops=300):
    """
    Kumpulkan crop gambar plat dari file LP-only.
    Return list of numpy arrays (plate crops, BGR).
    """
    crops = []
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        text = lbl_path.read_text().strip()
        if not text:
            continue
        classes = [int(l.split()[0]) for l in text.splitlines() if len(l.split()) == 5]

        # Hanya ambil dari gambar yang HANYA punya LP (bukan campur kendaraan)
        if not classes or any(c in [0, 1, 2] for c in classes):
            continue

        img_path = img_dir / (lbl_path.stem + ".jpg")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5 or int(parts[0]) != 3:
                continue
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            px_w, px_h = bw * W, bh * H

            # Skip plat terlalu kecil atau terlalu besar untuk jadi template
            if px_w < 40 or px_h < 12 or bw > 0.85:
                continue

            x1 = max(0, int((xc - bw / 2) * W))
            y1 = max(0, int((yc - bh / 2) * H))
            x2 = min(W, int((xc + bw / 2) * W))
            y2 = min(H, int((yc + bh / 2) * H))
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

        if len(crops) >= max_crops:
            break

    print(f"  Collected {len(crops)} plate crop templates")
    return crops


def paste_plate_on_vehicle(img, vehicle_boxes, plate_crop):
    """
    Tempel plate_crop ke posisi bawah tengah salah satu bbox kendaraan.

    Args:
        img          : numpy BGR image
        vehicle_boxes: list of (class_id, xc, yc, bw, bh) — vehicle labels
        plate_crop   : numpy BGR plate image

    Returns:
        (augmented_img, plate_bbox) or (img, None) if failed
    """
    if not vehicle_boxes:
        return img, None

    H, W = img.shape[:2]
    # Pilih bbox kendaraan yang cukup besar
    usable = [(cid, xc, yc, bw, bh) for (cid, xc, yc, bw, bh) in vehicle_boxes
              if bw * W > 80 and bh * H > 80]
    if not usable:
        return img, None

    cid, xc, yc, bw, bh = random.choice(usable)

    vx1 = int((xc - bw / 2) * W)
    vy1 = int((yc - bh / 2) * H)
    vx2 = int((xc + bw / 2) * W)
    vy2 = int((yc + bh / 2) * H)
    v_w  = vx2 - vx1
    v_h  = vy2 - vy1

    # Target plate width: 18-28% of vehicle width
    scale_factor = random.uniform(0.18, 0.28)
    target_w = max(MIN_PLATE_PX, int(v_w * scale_factor))

    # Pertahankan aspect ratio plat asli (minimal 2.5:1)
    ph, pw = plate_crop.shape[:2]
    aspect = pw / max(ph, 1)
    aspect = max(2.5, min(aspect, 6.0))   # clamp ke range plat yang valid
    target_h = max(10, int(target_w / aspect))

    if target_w > v_w * 0.8 or target_h > v_h * 0.5:
        return img, None

    plate_resized = cv2.resize(plate_crop, (target_w, target_h),
                               interpolation=cv2.INTER_LINEAR)

    # Posisi: bawah tengah vehicle bbox, sedikit di atas tepi bawah
    offset_y = random.uniform(0.10, 0.25)  # 10-25% dari atas tepi bawah vehicle
    paste_x = vx1 + (v_w // 2) - (target_w // 2) + random.randint(-int(v_w * 0.05), int(v_w * 0.05))
    paste_y = vy2 - target_h - int(v_h * offset_y)

    # Clamp agar tidak keluar gambar
    paste_x = max(0, min(paste_x, W - target_w))
    paste_y = max(0, min(paste_y, H - target_h))

    # Paste
    result = img.copy()
    result[paste_y:paste_y + target_h, paste_x:paste_x + target_w] = plate_resized

    # Hitung normalized bbox LP
    new_xc = (paste_x + target_w / 2) / W
    new_yc = (paste_y + target_h / 2) / H
    new_w  = target_w / W
    new_h  = target_h / H

    # Validasi range 0-1
    if not all(0 < v < 1 for v in [new_xc, new_yc, new_w, new_h]):
        return img, None

    return result, (3, new_xc, new_yc, new_w, new_h)


def augment_standard(img, boxes):
    """Apply standard albumentations aug. Return (aug_img, aug_boxes)."""
    bboxes  = [(xc, yc, bw, bh) for (_, xc, yc, bw, bh) in boxes]
    classes = [cid for (cid, _, _, _, _) in boxes]

    try:
        result = AUG_PIPELINE(image=img, bboxes=bboxes, class_labels=classes)
        aug_boxes = [(c, xc, yc, bw, bh)
                     for c, (xc, yc, bw, bh) in zip(result["class_labels"], result["bboxes"])]
        return result["image"], aug_boxes
    except Exception:
        return img, boxes


def run_copypaste(split, plate_crops, n_target):
    """Generate copy-paste synthetic images for one split."""
    img_dir = DATASET / "images" / split
    lbl_dir = DATASET / "labels" / split
    out_img = OUTPUT_DIR / "images" / split
    out_lbl = OUTPUT_DIR / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # Collect vehicle-only images (have vehicle, no LP)
    vehicle_images = []
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        text = lbl_path.read_text().strip()
        if not text:
            continue
        rows = [r for r in text.splitlines() if len(r.split()) == 5]
        classes = [int(r.split()[0]) for r in rows]
        if any(c in [0, 1, 2] for c in classes) and 3 not in classes:
            vehicle_images.append(lbl_path)

    print(f"  [{split}] Vehicle-only images: {len(vehicle_images)}")

    if not vehicle_images or not plate_crops:
        print(f"  [{split}] Skipping copy-paste (insufficient data)")
        return 0

    generated = 0
    attempts  = 0
    max_att   = n_target * 4

    while generated < n_target and attempts < max_att:
        attempts += 1
        lbl_path    = random.choice(vehicle_images)
        img_path    = img_dir / (lbl_path.stem + ".jpg")
        plate_crop  = random.choice(plate_crops)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        boxes = read_labels(lbl_path)
        v_boxes = [(cid, xc, yc, bw, bh) for (cid, xc, yc, bw, bh) in boxes
                   if cid in [0, 1, 2]]

        aug_img, lp_bbox = paste_plate_on_vehicle(img, v_boxes, plate_crop)
        if lp_bbox is None:
            continue

        # Apply color aug after paste (realistic blend)
        combined_boxes = boxes + [lp_bbox]
        aug_img, combined_boxes = augment_standard(aug_img, combined_boxes)
        if not combined_boxes:
            continue

        name = f"cp_{split}_{generated:05d}"
        cv2.imwrite(str(out_img / (name + ".jpg")), aug_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        write_labels(out_lbl / (name + ".txt"), combined_boxes)
        generated += 1

    print(f"  [{split}] Copy-paste generated: {generated}")
    return generated


def run_standard_aug(split):
    """Apply standard augmentations to all LP-only images."""
    img_dir = DATASET / "images" / split
    lbl_dir = DATASET / "labels" / split
    out_img = OUTPUT_DIR / "images" / split
    out_lbl = OUTPUT_DIR / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    lp_only_files = []
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        text = lbl_path.read_text().strip()
        if not text:
            continue
        classes = [int(l.split()[0]) for l in text.splitlines() if len(l.split()) == 5]
        if 3 in classes and not any(c in [0, 1, 2] for c in classes):
            lp_only_files.append(lbl_path)

    print(f"  [{split}] LP-only images for standard aug: {len(lp_only_files)}")

    generated = 0
    n_copies = AUGMENT_LP_COPIES if split == "train" else 1

    for lbl_path in lp_only_files:
        img_path = img_dir / (lbl_path.stem + ".jpg")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = read_labels(lbl_path)
        if not boxes:
            continue

        for i in range(n_copies):
            aug_img, aug_boxes = augment_standard(img, boxes)
            if not aug_boxes:
                continue

            name = f"lpaug_{lbl_path.stem}_{i}"
            cv2.imwrite(str(out_img / (name + ".jpg")), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            write_labels(out_lbl / (name + ".txt"), aug_boxes)
            generated += 1

    print(f"  [{split}] Standard aug generated: {generated}")
    return generated


def main():
    print("=" * 65)
    print("  Augmentasi Dataset Nopol")
    print("=" * 65)

    if OUTPUT_DIR.exists():
        print(f"\nFolder {OUTPUT_DIR} sudah ada.")
        ans = input("Hapus dan buat ulang? [y/N] ").strip().lower()
        if ans == "y":
            shutil.rmtree(OUTPUT_DIR)
        else:
            print("Dibatalkan.")
            return

    # ── Tahap 1: Copy-Paste Synthesis ──────────────────────────────────────
    print("\n[TAHAP 1] Copy-Paste Synthesis")
    print("  Mengumpulkan template plat...")

    train_lp_crops = collect_plate_crops(
        DATASET / "images" / "train",
        DATASET / "labels" / "train",
        max_crops=500,
    )
    val_lp_crops = collect_plate_crops(
        DATASET / "images" / "val",
        DATASET / "labels" / "val",
        max_crops=150,
    )
    # Fallback: gunakan train crops untuk val jika val crops kurang
    if len(val_lp_crops) < 30:
        val_lp_crops = train_lp_crops

    cp_train = run_copypaste("train", train_lp_crops, COPYPASTE_TARGET)
    cp_val   = run_copypaste("val",   val_lp_crops,   COPYPASTE_VAL)

    # ── Tahap 2: Standard Albumentations ───────────────────────────────────
    print("\n[TAHAP 2] Standard Augmentations (albumentations)")
    sa_train = run_standard_aug("train")
    sa_val   = run_standard_aug("val")

    # ── Summary ─────────────────────────────────────────────────────────────
    total_new = cp_train + cp_val + sa_train + sa_val
    print()
    print("=" * 65)
    print("  SELESAI")
    print("=" * 65)
    print(f"  Copy-paste train : {cp_train}")
    print(f"  Copy-paste val   : {cp_val}")
    print(f"  Standard aug train: {sa_train}")
    print(f"  Standard aug val  : {sa_val}")
    print(f"  Total baru       : {total_new}")
    print()
    print("  Untuk merge ke dataset utama:")
    print(f"    cp {OUTPUT_DIR}/images/train/*.jpg  dataset/images/train/")
    print(f"    cp {OUTPUT_DIR}/labels/train/*.txt  dataset/labels/train/")
    print(f"    cp {OUTPUT_DIR}/images/val/*.jpg    dataset/images/val/")
    print(f"    cp {OUTPUT_DIR}/labels/val/*.txt    dataset/labels/val/")
    print()
    print("  Setelah merge, jalankan ulang: python scripts/train_vehicle.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
