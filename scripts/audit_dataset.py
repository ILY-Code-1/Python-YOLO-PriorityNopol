"""
scripts/audit_dataset.py — Full YOLOv8 dataset quality audit

Menjalankan:
  python scripts/audit_dataset.py

Output:
  - Audit report dicetak ke terminal
  - dataset/audit_report.json (machine-readable)
"""

import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


# ─── Config ───────────────────────────────────────────────────────────────────

DATASET     = Path("dataset")
CLASS_NAMES = {0: "ambulance", 1: "police", 2: "fire_truck", 3: "license_plate"}
SPLITS      = ["train", "val"]

# ─────────────────────────────────────────────────────────────────────────────


def scan_split(split):
    img_dir = DATASET / "images" / split
    lbl_dir = DATASET / "labels" / split

    img_paths = sorted(img_dir.glob("*.jpg"))
    img_stems  = {p.stem for p in img_paths}
    lbl_stems  = {p.stem for p in lbl_dir.glob("*.txt")}

    class_boxes   = {i: 0 for i in range(4)}   # total bbox count per class
    class_images  = {i: 0 for i in range(4)}   # images that contain class
    resolutions   = []
    empty_labels  = []
    tiny_boxes    = []   # < 10×10 px
    huge_boxes    = []   # bbox_w > 80% AND bbox_h > 80%
    bad_class     = []
    lp_bboxes     = []   # (px_w, px_h, norm_w, norm_h)

    co_v_only  = 0   # vehicle classes only, no LP
    co_lp_only = 0   # LP only, no vehicle
    co_both    = 0   # both vehicle + LP
    co_empty   = 0

    no_label    = img_stems - lbl_stems
    label_no_img = lbl_stems - img_stems

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        resolutions.append((W, H))

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        text = lbl_path.read_text().strip()
        if not text:
            empty_labels.append(img_path.stem)
            co_empty += 1
            continue

        classes_in_img = set()
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                bad_class.append((img_path.stem, "bad_format", line))
                continue
            try:
                cid  = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                bad_class.append((img_path.stem, "parse_error", line))
                continue

            if cid not in CLASS_NAMES:
                bad_class.append((img_path.stem, "unknown_class", cid))
                continue

            px_w = bw * W
            px_h = bh * H
            classes_in_img.add(cid)
            class_boxes[cid]  += 1

            if px_w < 10 or px_h < 10:
                tiny_boxes.append({"file": img_path.stem, "class": CLASS_NAMES[cid],
                                   "px_w": round(px_w, 1), "px_h": round(px_h, 1)})
            if bw > 0.8 and bh > 0.8:
                huge_boxes.append({"file": img_path.stem, "class": CLASS_NAMES[cid],
                                   "norm_w": round(bw, 3), "norm_h": round(bh, 3)})
            if cid == 3:
                lp_bboxes.append((px_w, px_h, bw, bh))

        for cid in classes_in_img:
            class_images[cid] += 1

        has_v  = bool(classes_in_img & {0, 1, 2})
        has_lp = 3 in classes_in_img
        if has_v and has_lp:
            co_both += 1
        elif has_v:
            co_v_only += 1
        elif has_lp:
            co_lp_only += 1

    ws = [r[0] for r in resolutions]
    hs = [r[1] for r in resolutions]

    lp_too_small = sum(1 for x in lp_bboxes if x[0] < 32 or x[1] < 32)
    lp_pws = [x[0] for x in lp_bboxes]
    lp_phs = [x[1] for x in lp_bboxes]
    lp_nws = [x[2] for x in lp_bboxes]
    lp_nhs = [x[3] for x in lp_bboxes]

    huge_by_class = {}
    for h in huge_boxes:
        cn = h["class"]
        huge_by_class[cn] = huge_by_class.get(cn, 0) + 1

    return {
        "split":           split,
        "total_images":    len(img_paths),
        "no_label_count":  len(no_label),
        "label_no_img":    len(label_no_img),
        "empty_labels":    len(empty_labels),
        "class_boxes":     {CLASS_NAMES[k]: v for k, v in class_boxes.items()},
        "class_images":    {CLASS_NAMES[k]: v for k, v in class_images.items()},
        "co_occur": {
            "vehicle_only":     co_v_only,
            "lp_only":          co_lp_only,
            "vehicle_and_lp":   co_both,
            "empty":            co_empty,
        },
        "resolution": {
            "w_min": min(ws) if ws else 0,
            "w_max": max(ws) if ws else 0,
            "w_avg": int(np.mean(ws)) if ws else 0,
            "w_med": int(np.median(ws)) if ws else 0,
            "h_min": min(hs) if hs else 0,
            "h_max": max(hs) if hs else 0,
            "h_avg": int(np.mean(hs)) if hs else 0,
            "h_med": int(np.median(hs)) if hs else 0,
        },
        "tiny_boxes":      {"count": len(tiny_boxes), "samples": tiny_boxes[:5]},
        "huge_boxes":      {"count": len(huge_boxes), "by_class": huge_by_class, "samples": huge_boxes[:5]},
        "bad_class":       {"count": len(bad_class),  "samples": bad_class[:5]},
        "lp_quality": {
            "total_boxes":    len(lp_bboxes),
            "too_small":      lp_too_small,
            "too_small_pct":  round(100 * lp_too_small / len(lp_bboxes), 1) if lp_bboxes else 0,
            "px_w_min":       round(min(lp_pws), 1)           if lp_pws else 0,
            "px_w_max":       round(max(lp_pws), 1)           if lp_pws else 0,
            "px_w_avg":       round(float(np.mean(lp_pws)), 1) if lp_pws else 0,
            "px_w_med":       round(float(np.median(lp_pws)), 1) if lp_pws else 0,
            "px_h_min":       round(min(lp_phs), 1)           if lp_phs else 0,
            "px_h_max":       round(max(lp_phs), 1)           if lp_phs else 0,
            "px_h_avg":       round(float(np.mean(lp_phs)), 1) if lp_phs else 0,
            "px_h_med":       round(float(np.median(lp_phs)), 1) if lp_phs else 0,
            "norm_w_avg":     round(float(np.mean(lp_nws)), 4) if lp_nws else 0,
            "norm_h_avg":     round(float(np.mean(lp_nhs)), 4) if lp_nhs else 0,
        },
    }


def print_report(results):
    total_imgs = sum(r["total_images"] for r in results)
    train_r = next(r for r in results if r["split"] == "train")
    val_r   = next(r for r in results if r["split"] == "val")
    split_pct = 100 * train_r["total_images"] // total_imgs

    SEP  = "=" * 70
    SEP2 = "-" * 70

    print()
    print(SEP)
    print("  DATASET AUDIT REPORT — YOLOv8 Priority Vehicle Detection")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(SEP)

    # ── 1. Overview ────────────────────────────────────────────────────────
    print()
    print("1. OVERVIEW")
    print(SEP2)
    print(f"  Total images   : {total_imgs}")
    print(f"  Train / Val    : {train_r['total_images']} / {val_r['total_images']}"
          f"  ({split_pct}% / {100 - split_pct}%)")

    target = 80
    ok = abs(split_pct - target) <= 5
    print(f"  Split ratio    : {'OK (80/20)' if ok else 'WARNING: deviates from 80/20'}")

    # ── 2. Class Distribution ─────────────────────────────────────────────
    print()
    print("2. CLASS DISTRIBUTION")
    print(SEP2)
    print(f"  {'Class':15s}  {'Train boxes':>12}  {'Val boxes':>10}  {'Total':>7}  {'Rec. min':>9}  {'Status':>8}")
    print("  " + "-" * 65)
    mins = {"ambulance": 300, "police": 300, "fire_truck": 300, "license_plate": 600}
    for cn in ["ambulance", "police", "fire_truck", "license_plate"]:
        tr = train_r["class_boxes"][cn]
        vl = val_r["class_boxes"][cn]
        tot = tr + vl
        rec = mins[cn]
        status = "OK" if tot >= rec else "LOW"
        flag = " <-- NEEDS MORE DATA" if status == "LOW" else ""
        print(f"  {cn:15s}  {tr:>12}  {vl:>10}  {tot:>7}  {rec:>9}  {status:>8}{flag}")

    # ── 3. Co-Occurrence Analysis ─────────────────────────────────────────
    print()
    print("3. CO-OCCURRENCE ANALYSIS (train)")
    print(SEP2)
    co = train_r["co_occur"]
    total_files = sum(co.values())
    print(f"  Vehicle only (no plate)  : {co['vehicle_only']:4d}  ({100*co['vehicle_only']//total_files}%)")
    print(f"  Plate only (no vehicle)  : {co['lp_only']:4d}  ({100*co['lp_only']//total_files}%)")
    print(f"  Vehicle AND plate        : {co['vehicle_and_lp']:4d}  ({100*co['vehicle_and_lp']//total_files}%)")
    print(f"  Empty (background)       : {co['empty']:4d}  ({100*co['empty']//total_files}%)")
    print()
    if co["vehicle_and_lp"] == 0:
        print("  !! CRITICAL: 0 images have BOTH vehicle + plate annotations.")
        print("     Model cannot learn to detect plates IN CONTEXT of vehicles.")
        print("     Action required: run scripts/augment_nopol.py (copy-paste augmentation)")

    # ── 4. Image Quality ──────────────────────────────────────────────────
    print()
    print("4. IMAGE QUALITY")
    print(SEP2)
    for r in results:
        res = r["resolution"]
        print(f"  [{r['split'].upper()}] Resolution W: min={res['w_min']}  max={res['w_max']}"
              f"  avg={res['w_avg']}  median={res['w_med']}")
        print(f"         Resolution H: min={res['h_min']}  max={res['h_max']}"
              f"  avg={res['h_avg']}  median={res['h_med']}")
    print()
    print("  Note: images with short-side < 640 will be UPSCALED during training")
    print("        (may reduce label accuracy for small objects like nopol)")

    # ── 5. Label Quality ──────────────────────────────────────────────────
    print()
    print("5. LABEL QUALITY")
    print(SEP2)
    for r in results:
        spl = r["split"].upper()
        print(f"  [{spl}]")
        print(f"    No label file        : {r['no_label_count']}")
        print(f"    Label without image  : {r['label_no_img']}")
        print(f"    Empty label files    : {r['empty_labels']}  (valid background samples for YOLO)")
        print(f"    Tiny boxes (<10px)   : {r['tiny_boxes']['count']}")
        print(f"    Huge boxes (>80%)    : {r['huge_boxes']['count']}"
              f"  by class: {r['huge_boxes']['by_class']}")
        print(f"    Bad class IDs        : {r['bad_class']['count']}")

    # ── 6. License Plate (Nopol) Quality ──────────────────────────────────
    print()
    print("6. LICENSE PLATE (NOPOL) QUALITY")
    print(SEP2)
    for r in results:
        lp = r["lp_quality"]
        spl = r["split"].upper()
        print(f"  [{spl}]")
        print(f"    Total LP boxes       : {lp['total_boxes']}")
        print(f"    Too small (<32px)    : {lp['too_small']}  ({lp['too_small_pct']}%)")
        print(f"    Pixel W  (min/avg/med/max) : {lp['px_w_min']:.0f} / {lp['px_w_avg']:.0f}"
              f" / {lp['px_w_med']:.0f} / {lp['px_w_max']:.0f}")
        print(f"    Pixel H  (min/avg/med/max) : {lp['px_h_min']:.0f} / {lp['px_h_avg']:.0f}"
              f" / {lp['px_h_med']:.0f} / {lp['px_h_max']:.0f}")
        print(f"    Norm W avg           : {lp['norm_w_avg']:.3f}")
        print(f"    Norm H avg           : {lp['norm_h_avg']:.3f}")
    print()
    print("  Recommendation: YOLOv8n anchor grid starts at 32px.")
    print("  LP boxes < 32px on any side will NOT be learned reliably.")
    print("  Action: use imgsz=1280 OR two-stage detection (train_nopol.py)")

    # ── 7. Recommendations ────────────────────────────────────────────────
    print()
    print("7. RECOMMENDATIONS (priority order)")
    print(SEP2)

    recs = []

    if train_r["co_occur"]["vehicle_and_lp"] == 0:
        recs.append(("CRITICAL", "Co-annotation missing",
                     "Run scripts/augment_nopol.py to synthesize vehicle+plate images\n"
                     "     via copy-paste augmentation. This is the #1 issue."))

    lp_tr = train_r["lp_quality"]
    if lp_tr["too_small_pct"] > 20:
        recs.append(("HIGH", f"Nopol too-small: {lp_tr['too_small_pct']}% boxes < 32px",
                     "Use two-stage detection (train_nopol.py) OR increase imgsz to 1280\n"
                     "     OR filter out annotations where px_h < 16px (unlearnable)"))

    fire = train_r["class_boxes"]["fire_truck"] + val_r["class_boxes"]["fire_truck"]
    if fire < 300:
        recs.append(("HIGH", f"fire_truck underrepresented ({fire} boxes, need 300+)",
                     "Collect 100+ more fire_truck images with scripts/collect_dataset.py"))

    if train_r["empty_labels"] > 100:
        pct = 100 * train_r["empty_labels"] // train_r["total_images"]
        recs.append(("MEDIUM", f"Empty labels: {train_r['empty_labels']} ({pct}% of train)",
                     "These are valid background samples but at high % they reduce\n"
                     "     positive sample diversity. Consider removing images with\n"
                     "     no annotations that don't provide useful negatives."))

    if train_r["huge_boxes"]["count"] > 50:
        recs.append(("MEDIUM", f"Huge bboxes: {train_r['huge_boxes']['count']} (>80% image area)",
                     "Review and remove. Close-up shots where vehicle fills frame\n"
                     "     provide little background context for the model."))

    train_res = train_r["resolution"]
    if train_res["w_min"] < 400:
        recs.append(("LOW", f"Low-res images exist (min W={train_res['w_min']}px)",
                     "Consider removing images with short-side < 300px:\n"
                     "     upscaling artifacts hurt annotation accuracy"))

    for priority, issue, action in recs:
        print(f"  [{priority}] {issue}")
        print(f"     Action: {action}")
        print()

    # ── 8. Labeling Guidelines for Nopol ──────────────────────────────────
    print()
    print("8. NOPOL LABELING GUIDELINES")
    print(SEP2)
    print("  Tool recommendation : Roboflow (web, free tier) or CVAT (self-host)")
    print("                        DO NOT use LabelImg (outdated, no smart-label)")
    print()
    print("  Bounding box rules for license plate:")
    print("  - Tight fit: box edge should be 1-3px outside plate border")
    print("  - Include full plate text area (do not cut off characters)")
    print("  - Do NOT include mounting frame or surroundings")
    print("  - For angle/perspective plates: follow the visible plate edges")
    print("  - Minimum annotatable size: 32x12 pixels in the training image")
    print("  - Skip plates smaller than 20x8px (unlearnable at imgsz=640)")
    print()
    print("  CRITICAL for two-stage detection:")
    print("  - Annotate plates ON vehicle images (not standalone plate crops)")
    print("  - Each vehicle image should have BOTH the vehicle bbox AND plate bbox")
    print(SEP)


def main():
    print("Scanning dataset... (this may take 1-3 minutes)")
    results = [scan_split(s) for s in SPLITS]
    print_report(results)

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "splits": results,
    }
    out = DATASET / "audit_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAudit report saved: {out}")


if __name__ == "__main__":
    main()
