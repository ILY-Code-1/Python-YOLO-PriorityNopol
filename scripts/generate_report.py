"""
scripts/generate_report.py
Generate dataset_report.json and bbox visualization samples.
"""

import json
import random
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

DATASET = Path("dataset")
IMAGES_TRAIN = DATASET / "images" / "train"
IMAGES_VAL   = DATASET / "images" / "val"
LABELS_TRAIN = DATASET / "labels" / "train"
LABELS_VAL   = DATASET / "labels" / "val"
VIZ_OUT      = DATASET / "visualizations"
VIZ_OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "ambulance", 1: "police", 2: "fire_truck", 3: "license_plate"}
COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 128, 255), 3: (0, 0, 255)}

# ── Load split stats ───────────────────────────────────────────────────────────
split_stats = json.load(open(DATASET / "_split_stats.json"))["split_stats"]
aug_stats   = json.load(open(DATASET / "_aug_stats.json"))

# ── Count labels per class in train/val ───────────────────────────────────────
def count_labels(labels_dir):
    counts = {name: 0 for name in CLASS_NAMES.values()}
    bbox_widths = {name: [] for name in CLASS_NAMES.values()}
    bbox_heights = {name: [] for name in CLASS_NAMES.values()}
    empty = 0
    bad_fmt = 0
    out_of_range = 0
    for lf in labels_dir.glob("*.txt"):
        text = lf.read_text().strip()
        if not text:
            empty += 1
            continue
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                bad_fmt += 1
                continue
            try:
                cid = int(parts[0])
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                bad_fmt += 1
                continue
            if any(v < 0 or v > 1 for v in vals):
                out_of_range += 1
                continue
            name = CLASS_NAMES.get(cid)
            if name:
                counts[name] += 1
                bbox_widths[name].append(vals[2])
                bbox_heights[name].append(vals[3])
    return counts, bbox_widths, bbox_heights, empty, bad_fmt, out_of_range

train_counts, train_bw, train_bh, t_empty, t_bad, t_oor = count_labels(LABELS_TRAIN)
val_counts,   val_bw,   val_bh,   v_empty, v_bad, v_oor   = count_labels(LABELS_VAL)

total_counts = {n: train_counts[n] + val_counts[n] for n in CLASS_NAMES.values()}

# ── Compute bbox stats ─────────────────────────────────────────────────────────
def bbox_stats(widths, heights):
    if not widths:
        return {}
    w = np.array(widths)
    h = np.array(heights)
    return {
        "count": len(w),
        "avg_width":  round(float(w.mean()), 4),
        "avg_height": round(float(h.mean()), 4),
        "min_width":  round(float(w.min()),  4),
        "max_width":  round(float(w.max()),  4),
    }

all_bw = {}
all_bh = {}
for n in CLASS_NAMES.values():
    all_bw[n] = train_bw[n] + val_bw[n]
    all_bh[n] = train_bh[n] + val_bh[n]

bbox_distribution = {n: bbox_stats(all_bw[n], all_bh[n]) for n in CLASS_NAMES.values()}

# ── Validation checks ──────────────────────────────────────────────────────────
total_imgs  = len(list(IMAGES_TRAIN.glob("*.jpg"))) + len(list(IMAGES_VAL.glob("*.jpg")))
total_train = len(list(IMAGES_TRAIN.glob("*.jpg")))
total_val   = len(list(IMAGES_VAL.glob("*.jpg")))

targets = {"ambulance": 300, "police": 300, "fire_truck": 300, "license_plate": 600}
targets_met = {n: total_counts[n] >= targets[n] for n in targets}

# ── Pool manifest summary ──────────────────────────────────────────────────────
manifest = json.load(open(DATASET / "_pool_manifest.json"))
pool_total = manifest["total"]
pool_counters = manifest["counters"]

# ── Build report ───────────────────────────────────────────────────────────────
report = {
    "project": "Deteksi Otomatis Kendaraan Prioritas - YOLOv8n",
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "pipeline_stages": [
        "Stage 1: Image collection via BingImageCrawler (4 classes, 760 clean images)",
        "Stage 2: Multi-stage auto-annotation (YOLOv8n + EasyOCR + contour fallback)",
        "Stage 3: QC filtering (blur/dark/oversized bbox removal)",
        "Stage 4: Data augmentation (brightness, contrast, blur, noise, rotation, perspective)",
        "Stage 5: Pool merge (clean + augmented = 2166 images)",
        "Stage 6: Stratified 80/20 train/val split",
    ],
    "dataset_summary": {
        "total_images": total_imgs,
        "total_train":  total_train,
        "total_val":    total_val,
        "split_ratio":  "%.1f%% / %.1f%%" % (total_train / total_imgs * 100, total_val / total_imgs * 100),
        "pool_size":    pool_total,
        "augmented_added": aug_stats["total_augmented"],
    },
    "class_distribution": {
        "total":  total_counts,
        "train":  train_counts,
        "val":    val_counts,
        "targets": targets,
        "targets_met": targets_met,
    },
    "bbox_distribution": bbox_distribution,
    "augmentation_stats": aug_stats,
    "annotation_status": {
        "method": "Auto-annotation: YOLOv8n (vehicle detection) + EasyOCR multi-stage (license plate) + contour fallback",
        "empty_label_files_train": t_empty,
        "empty_label_files_val":   v_empty,
        "bad_format_train":  t_bad,
        "bad_format_val":    v_bad,
        "out_of_range_train": t_oor,
        "out_of_range_val":   v_oor,
    },
    "validation_checks": {
        "total_images_gte_2000":     total_imgs >= 2000,
        "train_val_split_ok":        abs(total_train / total_imgs - 0.8) < 0.05,
        "empty_labels_note":         "%d empty label files (valid background/hard-negative samples for YOLO)" % (t_empty + v_empty),
        "no_bad_format":             (t_bad + v_bad) == 0,
        "no_out_of_range":           (t_oor + v_oor) == 0,
        "all_class_targets_met":     all(targets_met.values()),
    },
    "retrain_config": {
        "model":    "yolov8n.pt",
        "epochs":   100,
        "imgsz":    640,
        "batch":    16,
        "patience": 20,
        "command":  "python scripts/train.py",
        "dataset_yaml": "dataset/dataset.yaml",
    },
    "label_format": "YOLO: class_id x_center y_center width height (all normalized 0-1)",
    "class_mapping": {v: k for k, v in CLASS_NAMES.items()},
    "ready_for_training": all([
        (t_bad + v_bad) == 0,
        (t_oor + v_oor) == 0,
        all(targets_met.values()),
    ]),
}

with open(DATASET / "dataset_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("dataset_report.json saved.")
print("ready_for_training:", report["ready_for_training"])
print("total_images:", total_imgs, "| train:", total_train, "| val:", total_val)
print("class totals:", total_counts)

# ── Visualizations: 4 images per class, draw bbox ─────────────────────────────
print("\nGenerating visualizations...")

# Build index: class -> list of (img_path, lbl_path)
class_samples = {n: [] for n in CLASS_NAMES.values()}
for split, img_dir, lbl_dir in [("train", IMAGES_TRAIN, LABELS_TRAIN),
                                  ("val",   IMAGES_VAL,   LABELS_VAL)]:
    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        text = lbl_path.read_text().strip()
        if not text:
            continue
        cids = set()
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cids.add(int(parts[0]))
                except ValueError:
                    pass
        for cid in cids:
            name = CLASS_NAMES.get(cid)
            if name and len(class_samples[name]) < 20:
                class_samples[name].append((img_path, lbl_path))

def draw_bboxes(img_path, lbl_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    H, W = img.shape[:2]
    text = lbl_path.read_text().strip()
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cid = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        x1 = int((xc - bw / 2) * W)
        y1 = int((yc - bh / 2) * H)
        x2 = int((xc + bw / 2) * W)
        y2 = int((yc + bh / 2) * H)
        color = COLORS.get(cid, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES.get(cid, str(cid))
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img

saved = 0
for cls_name, samples in class_samples.items():
    random.seed(42)
    picks = random.sample(samples, min(4, len(samples)))
    for idx, (img_path, lbl_path) in enumerate(picks, 1):
        vis = draw_bboxes(img_path, lbl_path)
        if vis is None:
            continue
        # Resize to max 640px wide for display
        H, W = vis.shape[:2]
        if W > 640:
            scale = 640 / W
            vis = cv2.resize(vis, (640, int(H * scale)))
        out_name = "%s_%d.jpg" % (cls_name, idx)
        cv2.imwrite(str(VIZ_OUT / out_name), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved += 1

print("Saved %d visualization images to dataset/visualizations/" % saved)
print("\nDone.")
