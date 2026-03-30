"""
scripts/multi_stage_annotate.py
Multi-stage auto annotation:
  Stage 1: YOLOv8 vehicle detection
  Stage 2: EasyOCR license plate detection (full + crop)
  Stage 3: Heuristic fallback
"""

import json
import math
import re
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr

CLEAN   = Path("dataset/_clean")
LBL_OUT = Path("dataset/_new_labels")
LBL_OUT.mkdir(parents=True, exist_ok=True)

OUR_CLS  = {"ambulance": 0, "police": 1, "fire_truck": 2, "license_plate": 3}
COCO_VEH = {2, 5, 7}
CONF_VEH = 0.18

print("Loading YOLO...", flush=True)
model = YOLO("yolov8n.pt")
print("Loading EasyOCR...", flush=True)
ocr = easyocr.Reader(["en", "id"], gpu=False, verbose=False)
print("Models ready.", flush=True)


def xyxy2yolo(x1, y1, x2, y2, W, H):
    xc = max(0.0, min(1.0, ((x1 + x2) / 2) / W))
    yc = max(0.0, min(1.0, ((y1 + y2) / 2) / H))
    bw = max(0.001, min(1.0, (x2 - x1) / W))
    bh = max(0.001, min(1.0, (y2 - y1) / H))
    return xc, yc, bw, bh


def is_valid_plate_text(txt):
    txt = re.sub(r"[^A-Z0-9]", "", txt.upper())
    has_alpha = any(c.isalpha() for c in txt)
    has_digit = any(c.isdigit() for c in txt)
    return len(txt) >= 4 and has_alpha and has_digit


def compute_iou(b1, b2):
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def run_ocr_plate(img_bgr, W, H, offset_x=0, offset_y=0, conf_min=0.45):
    results = ocr.readtext(img_bgr, detail=1, paragraph=False)
    plates = []
    for (bbox_pts, text, conf) in results:
        if conf < conf_min:
            continue
        if not is_valid_plate_text(text):
            continue
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        x1r, y1r = min(xs), min(ys)
        x2r, y2r = max(xs), max(ys)
        gx1 = x1r + offset_x
        gy1 = y1r + offset_y
        gx2 = x2r + offset_x
        gy2 = y2r + offset_y
        pw = gx2 - gx1
        ph = gy2 - gy1
        if ph <= 0:
            continue
        ar = pw / ph
        if not (1.3 <= ar <= 7.0):
            continue
        if pw < W * 0.02:
            continue
        plates.append((gx1, gy1, gx2, gy2, conf, text))
    return plates


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
    return p1x, p1y, p2x, p2y


total = 0
labeled = 0
ocr_success = 0
heuristic_count = 0
no_det = 0
cls_counts = {"ambulance": 0, "police": 0, "fire_truck": 0, "license_plate": 0}
ann_data = []

for cls_dir in sorted(CLEAN.iterdir()):
    if not cls_dir.is_dir():
        continue
    cls_name = cls_dir.name
    cls_id = OUR_CLS.get(cls_name, -1)
    if cls_id < 0:
        continue
    lbl_dir = LBL_OUT / cls_name
    lbl_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(cls_dir.glob("*.jpg"))
    print("[%s] %d files" % (cls_name, len(files)), flush=True)

    for i, img_path in enumerate(files, 1):
        total += 1
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        lines = []

        # YOLO vehicle detection
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
            # Stage 1: OCR full image
            plate_found = run_ocr_plate(img, W, H, 0, 0, conf_min=0.45)

            # Stage 2: OCR on vehicle crops (upscaled)
            if veh_dets:
                for vdet in veh_dets[:3]:
                    vx1, vy1, vx2, vy2 = vdet["bbox"]
                    pad_x = (vx2 - vx1) * 0.15
                    pad_y = (vy2 - vy1) * 0.15
                    cx1 = max(0, int(vx1 - pad_x))
                    cy1 = max(0, int(vy1 - pad_y))
                    cx2 = min(W, int(vx2 + pad_x))
                    cy2 = min(H, int(vy2 + pad_y))
                    crop = img[cy1:cy2, cx1:cx2]
                    if crop.size == 0:
                        continue
                    cH, cW = crop.shape[:2]
                    scale_back = 1.0
                    if cW < 300:
                        scale_f = 300 / cW
                        scale_back = cW / 300
                        crop = cv2.resize(crop, (int(cW * scale_f), int(cH * scale_f)),
                                          interpolation=cv2.INTER_CUBIC)
                    crop_plates = run_ocr_plate(crop, W, H, cx1, cy1, conf_min=0.45)
                    if scale_back != 1.0:
                        scaled = []
                        for (gx1r, gy1r, gx2r, gy2r, conf, txt) in crop_plates:
                            new_gx1 = cx1 + (gx1r - cx1) * scale_back
                            new_gy1 = cy1 + (gy1r - cy1) * scale_back
                            new_gx2 = cx1 + (gx2r - cx1) * scale_back
                            new_gy2 = cy1 + (gy2r - cy1) * scale_back
                            scaled.append((new_gx1, new_gy1, new_gx2, new_gy2, conf, txt))
                        crop_plates = scaled
                    plate_found.extend(crop_plates)

            # Filter and quality check
            valid_plates = []
            for (px1, py1, px2, py2, conf, txt) in plate_found:
                pw = px2 - px1
                ph = py2 - py1
                if ph <= 0:
                    continue
                ar = pw / ph
                if not (1.3 <= ar <= 7.0):
                    continue
                if pw > W * 0.60:
                    continue
                if pw < W * 0.025:
                    continue
                if ph > pw:
                    continue
                valid_plates.append((px1, py1, px2, py2, conf, txt))

            # IoU dedup
            final_plates = []
            for plate in sorted(valid_plates, key=lambda x: -x[4]):
                ab = plate[:4]
                if not any(compute_iou(ab, ex[:4]) > 0.85 for ex in final_plates):
                    final_plates.append(plate)

            if final_plates:
                for (px1, py1, px2, py2, conf, txt) in final_plates[:3]:
                    xc, yc, bw, bh = xyxy2yolo(px1, py1, px2, py2, W, H)
                    lines.append("%d %.6f %.6f %.6f %.6f" % (3, xc, yc, bw, bh))
                labeled += 1
                cls_counts["license_plate"] += len(final_plates)
                ocr_success += 1
            else:
                # Heuristic fallback
                if veh_dets:
                    best = max(veh_dets, key=lambda d: d["area"])
                    px1, py1, px2, py2 = estimate_plate_heuristic(*best["bbox"], W, H)
                    xc, yc, bw, bh = xyxy2yolo(px1, py1, px2, py2, W, H)
                    lines.append("%d %.6f %.6f %.6f %.6f" % (3, xc, yc, bw, bh))
                else:
                    lines.append("%d %.6f %.6f %.6f %.6f" % (3, 0.500, 0.500, 0.680, 0.420))
                labeled += 1
                heuristic_count += 1
                cls_counts["license_plate"] += 1

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        ann_data.append({"file": img_path.name, "class": cls_name, "objects": len(lines)})

        if i % 80 == 0 or i == len(files):
            print("  %d/%d" % (i, len(files)), flush=True)

print("DONE: total=%d labeled=%d no_det=%d" % (total, labeled, no_det))
print("ocr_success=%d heuristic=%d" % (ocr_success, heuristic_count))
print("cls_counts: %s" % cls_counts)

with open("dataset/_ann_data2.json", "w") as f:
    json.dump({"total": total, "labeled": labeled, "no_det": no_det,
               "ocr_success": ocr_success, "heuristic": heuristic_count,
               "cls_counts": cls_counts, "ann_data": ann_data}, f, indent=2)
print("ann_data2 saved.")
