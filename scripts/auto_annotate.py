"""
scripts/auto_annotate.py
────────────────────────
Auto-Annotation Engine untuk Dataset Kendaraan Prioritas
Menggunakan YOLOv8n pretrained (COCO) + Plate Estimator

Strategi per kelas:
  ambulance     → deteksi COCO [car, bus, truck] → map ke class_id 0
  police        → deteksi COCO [car, truck]       → map ke class_id 1
  fire_truck    → deteksi COCO [truck, bus, car]  → map ke class_id 2
  license_plate → estimasi plat dari kendaraan terdeteksi + fallback pusat gambar
                  → map ke class_id 3

Cara pakai:
    python scripts/auto_annotate.py
"""

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ─── Konfigurasi path ────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
IMG_TRAIN = BASE_DIR / "dataset" / "images" / "train"
IMG_VAL   = BASE_DIR / "dataset" / "images" / "val"
LBL_TRAIN = BASE_DIR / "dataset" / "labels" / "train"
LBL_VAL   = BASE_DIR / "dataset" / "labels" / "val"

# ─── Model & threshold ────────────────────────────────────────────────────────
MODEL_PATH  = BASE_DIR / "model" / "best.pt"
FALLBACK    = "yolov8n.pt"
CONF        = 0.18   # Threshold rendah agar kendaraan khusus (ambulans, dll.) tetap terdeteksi

# ─── Mapping COCO class ID → tipe kendaraan ──────────────────────────────────
# COCO: 2=car, 3=motorcycle, 5=bus, 7=truck, 1=bicycle
COCO_VEHICLE_IDS = {2, 3, 5, 7}

# Untuk setiap kelas dataset, COCO class mana yang diterima sebagai kandidat
CLASS_COCO_ACCEPT = {
    "ambulance"    : {2, 5, 7},       # car, bus, truck
    "police"       : {2, 7, 5},       # car, truck, bus
    "fire_truck"   : {7, 5, 2},       # truck, bus, car (priority order)
    "license_plate": {2, 7, 5, 3},    # semua kendaraan untuk estimasi plat
}

# Our class ID
OUR_CLASS_ID = {
    "ambulance"    : 0,
    "police"       : 1,
    "fire_truck"   : 2,
    "license_plate": 3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_class_from_filename(stem: str) -> str:
    """
    Ekstrak nama kelas dari stem filename.
    ambulance_001 → 'ambulance'
    police_val_001 → 'police'
    fire_truck_val_023 → 'fire_truck'
    """
    stem_lower = stem.lower()
    if stem_lower.startswith("ambulance"):
        return "ambulance"
    if stem_lower.startswith("police"):
        return "police"
    if stem_lower.startswith("fire_truck"):
        return "fire_truck"
    if stem_lower.startswith("license_plate"):
        return "license_plate"
    return "unknown"


def bbox_area(box) -> float:
    """Luas bounding box (piksel^2)."""
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1) * (y2 - y1))


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h) -> tuple:
    """Konversi piksel absolut ke format YOLO (normalized 0–1)."""
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    # Clamp ke [0,1]
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    bw = max(0.001, min(1.0, bw))
    bh = max(0.001, min(1.0, bh))
    return xc, yc, bw, bh


def estimate_plate_from_vehicle(vx1, vy1, vx2, vy2, img_w, img_h) -> tuple:
    """
    Estimasi posisi plat nomor dari bounding box kendaraan.
    Plat umumnya berada di:
      - Horizontal: tengah kendaraan (sekitar 30% lebar kendaraan)
      - Vertikal  : 80-95% dari tinggi kendaraan (bagian bawah)
    Ini adalah heuristik, BUKAN deteksi sesungguhnya.
    """
    vw = vx2 - vx1
    vh = vy2 - vy1

    # Lebar plat ≈ 30% lebar kendaraan, tinggi ≈ 10%
    plate_w = vw * 0.32
    plate_h = vh * 0.12

    # Posisi horizontal: tengah kendaraan
    px_center = vx1 + vw * 0.50
    # Posisi vertikal: 87% dari tinggi (area bumper bawah)
    py_center = vy1 + vh * 0.87

    px1 = px_center - plate_w / 2
    py1 = py_center - plate_h / 2
    px2 = px_center + plate_w / 2
    py2 = py_center + plate_h / 2

    # Pastikan tidak keluar batas gambar
    px1 = max(0, px1)
    py1 = max(0, py1)
    px2 = min(img_w, px2)
    py2 = min(img_h, py2)

    return xyxy_to_yolo(px1, py1, px2, py2, img_w, img_h)


def fallback_plate_bbox(img_w, img_h) -> tuple:
    """
    Fallback: jika tidak ada kendaraan terdeteksi di gambar license_plate,
    asumsikan plat berada di tengah gambar (60% x 35% coverage).
    Digunakan untuk gambar close-up plat nomor.
    """
    return (0.500, 0.500, 0.680, 0.420)


def detect_plate_by_contour(img_bgr) -> tuple | None:
    """
    Deteksi plat nomor menggunakan analisis kontur.
    Cari persegi panjang horizontal dengan aspek rasio 2:1 hingga 5:1.
    Returns (xc, yc, bw, bh) normalized, atau None jika tidak ditemukan.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Bilateral filter untuk mengurangi noise tapi jaga tepi
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged    = cv2.Canny(filtered, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for c in contours:
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            x, y, cw, ch = cv2.boundingRect(approx)
            aspect_ratio = cw / ch if ch > 0 else 0

            # Plat nomor: aspek rasio 1.5:1 hingga 6:1, ukuran minimal
            if 1.5 <= aspect_ratio <= 6.0 and cw > w * 0.05 and ch > h * 0.02:
                return xyxy_to_yolo(x, y, x + cw, y + ch, w, h)

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# ANNOTATE SINGLE IMAGE
# ═══════════════════════════════════════════════════════════════════════════════

def annotate_image(img_path: Path, lbl_path: Path, model: YOLO, stats: dict) -> int:
    """
    Proses satu gambar → tulis file label YOLO.

    Returns jumlah objek yang dianotasi.
    """
    stem      = img_path.stem
    cls_name  = get_class_from_filename(stem)
    class_id  = OUR_CLASS_ID.get(cls_name, -1)
    accepted_coco = CLASS_COCO_ACCEPT.get(cls_name, set())

    if class_id == -1:
        lbl_path.write_text("")
        stats["unknown_class"] = stats.get("unknown_class", 0) + 1
        return 0

    # Baca gambar
    img = cv2.imread(str(img_path))
    if img is None:
        lbl_path.write_text("")
        stats["unreadable"] = stats.get("unreadable", 0) + 1
        return 0
    img_h, img_w = img.shape[:2]

    # Jalankan YOLO inference
    results = model.predict(source=img, conf=CONF, verbose=False)
    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            coco_id = int(box.cls[0])
            conf    = float(box.conf[0])
            if coco_id in accepted_coco:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "coco_id": coco_id,
                    "conf"   : conf,
                    "bbox"   : (x1, y1, x2, y2),
                    "area"   : bbox_area((x1, y1, x2, y2)),
                })

    label_lines = []

    # ── Kelas kendaraan (ambulance, police, fire_truck) ───────────────────────
    if cls_name in ("ambulance", "police", "fire_truck"):
        if detections:
            # Pilih deteksi terbaik: skor gabungan confidence × area
            best = max(detections, key=lambda d: d["conf"] * math.sqrt(d["area"]))
            x1, y1, x2, y2 = best["bbox"]
            xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            stats[cls_name] = stats.get(cls_name, 0) + 1
            stats["total_objects"] = stats.get("total_objects", 0) + 1
        else:
            # Tidak terdeteksi → label kosong (jangan asal-asalan)
            stats["no_detection"] = stats.get("no_detection", 0) + 1

    # ── Kelas license_plate ───────────────────────────────────────────────────
    elif cls_name == "license_plate":
        if detections:
            # Gunakan kendaraan terbesar → estimasi posisi plat
            best = max(detections, key=lambda d: d["area"])
            x1, y1, x2, y2 = best["bbox"]
            xc, yc, bw, bh = estimate_plate_from_vehicle(x1, y1, x2, y2, img_w, img_h)
            label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            stats["license_plate_from_vehicle"] = stats.get("license_plate_from_vehicle", 0) + 1
        else:
            # Fallback 1: coba deteksi kontur plat
            plate = detect_plate_by_contour(img)
            if plate:
                xc, yc, bw, bh = plate
                label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                stats["license_plate_from_contour"] = stats.get("license_plate_from_contour", 0) + 1
            else:
                # Fallback 2: center bounding box untuk close-up plat
                xc, yc, bw, bh = fallback_plate_bbox(img_w, img_h)
                label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                stats["license_plate_fallback"] = stats.get("license_plate_fallback", 0) + 1

        stats["license_plate"] = stats.get("license_plate", 0) + 1
        stats["total_objects"] = stats.get("total_objects", 0) + 1

    # Tulis file label
    lbl_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
    return len(label_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  AUTO-ANNOTATION ENGINE - Kendaraan Prioritas")
    print("=" * 62)

    # Load model
    if MODEL_PATH.exists():
        print(f"[Model] Loading trained model: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
    else:
        print(f"[Model] Using pretrained: {FALLBACK}")
        model = YOLO(FALLBACK)

    global_stats = {
        "total_images"  : 0,
        "total_labeled" : 0,
        "total_objects" : 0,
        "no_detection"  : 0,
    }

    # Proses train dan val
    pairs = [
        (IMG_TRAIN, LBL_TRAIN, "train"),
        (IMG_VAL,   LBL_VAL,   "val"),
    ]

    for img_dir, lbl_dir, split in pairs:
        imgs = sorted(img_dir.glob("*.jpg"))
        print(f"\n[{split.upper()}] Processing {len(imgs)} images...")

        split_stats: dict = {}
        labeled = 0

        for i, img_path in enumerate(imgs, 1):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            n = annotate_image(img_path, lbl_path, model, split_stats)
            if n > 0:
                labeled += 1
            if i % 50 == 0 or i == len(imgs):
                print(f"  [{split}] {i}/{len(imgs)} processed...", flush=True)

        global_stats["total_images"]   += len(imgs)
        global_stats["total_labeled"]  += labeled
        global_stats["total_objects"]  += split_stats.get("total_objects", 0)
        global_stats["no_detection"]   += split_stats.get("no_detection", 0)

        print(f"  [{split}] Labeled  : {labeled}/{len(imgs)}")
        print(f"  [{split}] Objects  : {split_stats.get('total_objects', 0)}")
        print(f"  [{split}] No-detect: {split_stats.get('no_detection', 0)}")

    return global_stats


if __name__ == "__main__":
    stats = main()
    print(f"\nDone. Total: {stats['total_images']} images, "
          f"{stats['total_labeled']} labeled, "
          f"{stats['total_objects']} objects.")
