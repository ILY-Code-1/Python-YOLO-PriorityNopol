"""
scripts/resume_annotate.py
Resume annotation - hanya proses file yang BELUM ada labelnya.
Menghindari EasyOCR yang lambat dengan mode FAST (vehicle-only + contour plate).
"""

import json
import math
import re
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CLEAN   = Path("dataset/_clean")
LBL_OUT = Path("dataset/_new_labels")
LBL_OUT.mkdir(parents=True, exist_ok=True)

OUR_CLS  = {"ambulance": 0, "police": 1, "fire_truck": 2, "license_plate": 3}
COCO_VEH = {2, 5, 7}
CONF_VEH = 0.18

print("Loading YOLO...", flush=True)
model = YOLO("yolov8n.pt")
print("YOLO ready. (Fast mode - no EasyOCR)", flush=True)


def xyxy2yolo(x1, y1, x2, y2, W, H):
    xc = max(0.0, min(1.0, ((x1 + x2) / 2) / W))
    yc = max(0.0, min(1.0, ((y1 + y2) / 2) / H))
    bw = max(0.001, min(1.0, (x2 - x1) / W))
    bh = max(0.001, min(1.0, (y2 - y1) / H))
    return xc, yc, bw, bh


def compute_iou(b1, b2):
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def detect_plate_contour(img_bgr, W, H):
    """Plate detection via contour analysis."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bi   = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bi, 30, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:40]
    candidates = []
    for c in cnts:
        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, cw, ch = cv2.boundingRect(approx)
            if ch <= 0:
                continue
            ar = cw / ch
            if 1.5 <= ar <= 7.0 and cw > W * 0.04 and cw < W * 0.55:
                area = cw * ch
                candidates.append((x, y, x + cw, y + ch, area, ar))
    if candidates:
        best = max(candidates, key=lambda c: c[4])
        x1, y1, x2, y2 = best[:4]
        return xyxy2yolo(x1, y1, x2, y2, W, H)
    return None


def estimate_plate_heuristic(vx1, vy1, vx2, vy2, W, H):
    vw = vx2 - vx1
    vh = vy2 - vy1
    pw = vw * 0.30
    ph = vh * 0.12
    pcx = vx1 + vw * 0.50
    pcy = vy1 + vh * 0.87
    p1x = max(0, pcx - pw / 2)
    p1y = max(0, pcy - ph / 2)
    p2x = min(W, pcx + pw / 2)
    p2y = min(H, pcy + ph / 2)
    return xyxy2yolo(p1x, p1y, p2x, p2y, W, H)


total = 0
labeled = 0
skipped = 0
no_det  = 0
cls_counts = {"ambulance": 0, "police": 0, "fire_truck": 0, "license_plate": 0}

for cls_dir in sorted(CLEAN.iterdir()):
    if not cls_dir.is_dir():
        continue
    cls_name = cls_dir.name
    cls_id   = OUR_CLS.get(cls_name, -1)
    if cls_id < 0:
        continue
    lbl_dir = LBL_OUT / cls_name
    lbl_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(cls_dir.glob("*.jpg"))

    # Only process files WITHOUT existing label
    missing = [f for f in files if not (lbl_dir / (f.stem + ".txt")).exists()]
    print("[%s] %d total, %d missing labels" % (cls_name, len(files), len(missing)), flush=True)

    for i, img_path in enumerate(missing, 1):
        total += 1
        img = cv2.imread(str(img_path))
        if img is None:
            (lbl_dir / (img_path.stem + ".txt")).write_text("")
            skipped += 1
            continue
        H, W = img.shape[:2]
        lines = []

        # YOLO detect
        res = model.predict(source=img, conf=CONF_VEH, verbose=False)
        veh_dets = []
        if res and res[0].boxes is not None:
            for box in res[0].boxes:
                cid = int(box.cls[0])
                if cid in COCO_VEH:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)
                    veh_dets.append({"bbox": (x1, y1, x2, y2),
                                     "conf": float(box.conf[0]), "area": area})

        if cls_name != "license_plate":
            if veh_dets:
                best = max(veh_dets, key=lambda d: d["conf"] * math.sqrt(d["area"]))
                xc, yc, bw, bh = xyxy2yolo(*best["bbox"], W, H)
                lines.append("%d %.6f %.6f %.6f %.6f" % (cls_id, xc, yc, bw, bh))
                labeled += 1
                cls_counts[cls_name] += 1
            else:
                no_det += 1
        else:
            # License plate: contour → vehicle heuristic → center fallback
            plate = detect_plate_contour(img, W, H)
            if plate:
                xc, yc, bw, bh = plate
                lines.append("3 %.6f %.6f %.6f %.6f" % (xc, yc, bw, bh))
                labeled += 1
                cls_counts["license_plate"] += 1
            elif veh_dets:
                best = max(veh_dets, key=lambda d: d["area"])
                xc, yc, bw, bh = estimate_plate_heuristic(*best["bbox"], W, H)
                lines.append("3 %.6f %.6f %.6f %.6f" % (xc, yc, bw, bh))
                labeled += 1
                cls_counts["license_plate"] += 1
            else:
                lines.append("3 0.500000 0.500000 0.680000 0.420000")
                labeled += 1
                cls_counts["license_plate"] += 1

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        if i % 50 == 0 or i == len(missing):
            print("  %d/%d" % (i, len(missing)), flush=True)

print("RESUME DONE: processed=%d labeled=%d skipped=%d no_det=%d" % (
    total, labeled, skipped, no_det))
print("cls_counts: %s" % cls_counts)

# Final label count
print("\nFinal label counts:")
for cls_dir in sorted(LBL_OUT.iterdir()):
    if cls_dir.is_dir():
        n = len(list(cls_dir.glob("*.txt")))
        print("  %s: %d" % (cls_dir.name, n))
